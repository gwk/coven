#!/usr/bin/env python3

# Note: any modules imported prior to the calls to install_trace and run_path
# will not report coverage fully, because their <module> code objects will not be captured.
# Therefore, we only use stdlib modules.
import sys; assert sys.version_info >= (3, 6, 0)
import marshal
import os
import os.path
import re
from collections import defaultdict
from dis import Instruction, findlinestarts, get_instructions, hasjabs, hasjrel, opname, opmap
from argparse import ArgumentParser
from inspect import getmodule
from itertools import chain
from os.path import abspath as abs_path, join as path_join, normpath as normalize_path
from runpy import run_path
from sys import exc_info, settraceinst, stderr, stdout
from types import CodeType


def main():
  arg_parser = ArgumentParser(description='coven: code coverage harness.')
  arg_parser.add_argument('-targets', nargs='*', default=[])
  arg_parser.add_argument('-dbg')
  arg_parser.add_argument('-show-all', action='store_true')
  arg_parser.add_argument('-color-on', dest='color', action='store_true', default=stdout.isatty())
  arg_parser.add_argument('-color-off', dest='color', action='store_false')
  excl = arg_parser.add_mutually_exclusive_group()
  excl.add_argument('-coalesce', nargs='+')
  trace_group = excl.add_argument_group('trace')
  trace_group.add_argument('-output')
  trace_group.add_argument('cmd', nargs='*')
  args = arg_parser.parse_args()
  arg_targets = expand_targets(args.targets)
  if args.coalesce:
    coalesce(trace_paths=args.coalesce, arg_targets=arg_targets, args=args)
  else:
    if not args.cmd:
      arg_parser.error('please specify a command.')
    trace_cmd(cmd=args.cmd, arg_targets=arg_targets, output_path=args.output, args=args)


def expand_targets(arg_targets):
  targets = set()
  for arg in arg_targets:
    targets.add(expand_module_name_or_path(arg))
  return targets


def expand_module_name_or_path(word):
  if word.endswith('.py') or '/' in word:
    return expand_module_path(word)
  else:
    return word


def expand_module_path(path):
  slash_pos = path.find('/')
  if slash_pos == -1: slash_pos = 0
  dot_pos = path.find('.', slash_pos)
  if dot_pos == -1: dot_pos = len(path)
  stem = path[:dot_pos]
  return stem.replace('/', '.')


def trace_cmd(cmd, arg_targets, output_path, args):
  'NOTE: this must be called before importing any module that we might wish to trace with coven.'
  cmd_head = abs_path(cmd[0])
  targets = set(arg_targets or ['__main__'])
  # although run_path alters and restores sys.argv[0],
  # we need to replace all of argv to provide the correct arguments to the command getting traced.
  orig_argv = sys.argv.copy()
  sys.argv = cmd.copy()
  # also need to fix the search path to imitate the regular interpreter.
  orig_path = sys.path
  sys.path = orig_path.copy()
  sys.path[0] = os.path.dirname(cmd[0]) # not sure if this is right in all cases.
  exit_code = 0
  code_edges = install_trace(targets, dbg=args.dbg)
  #if dbg: errSL('coven untraceable modules (imported prior to `install_trace`):', sorted(sys.modules.keys()))
  try:
    run_path(cmd_head, run_name='__main__')
  except FileNotFoundError as e:
    exit(f'coven error: could not find command to run: {cmd_head!r}')
  except SystemExit as e:
    exit_code = e.code
  except BaseException:
    from traceback import TracebackException
    exit_code = 1 # exit code that Python returns when an exception raises to toplevel.
    # format the traceback exactly as it would appear when run without coven.
    tbe = TracebackException(*exc_info())
    scrub_traceback(tbe)
    print(*tbe.format(), sep='', end='', file=stderr)
  finally:
    settraceinst(None, 0)
    stdout.flush()
    stderr.flush()
  sys.argv = orig_argv
  target_paths = gen_target_paths(targets, cmd_head, dbg=args.dbg)
  if output_path:
    write_coverage(output_path=output_path, target_paths=target_paths, code_edges=code_edges)
  else:
    report(target_paths=target_paths, code_edges=code_edges, args=args)
  exit(exit_code)


def scrub_traceback(tbe):
  'Remove frames from TracebackException object that refer to coven, rather than the child process under examination.'
  stack = tbe.stack # StackSummary is a subclass of list.
  if not stack or 'coven' not in stack[0].filename: return # not the root exception.
  #^ TODO: verify that the above is sufficiently strict,
  #^ while also covering both the installed entry_point and the local dev cases.
  del stack[0]
  while stack and stack[0].filename.endswith('runpy.py'): del stack[0] # remove coven runpy.run_path frames.


# Fake instruction/line offsets.
LINE_BEGIN  = OFF_BEGIN  = OP_BEGIN  = -1
LINE_RAISED = OFF_RAISED = OP_RAISED = -2
LINE_RETURN = OFF_RETURN = OP_RETURN = -3


def install_trace(targets, dbg):
  'NOTE: this must be called before importing any module that we might wish to trace with coven.'

  code_edges = defaultdict(set)
  file_name_filter = {}

  def is_code_path_targeted(code):
    module = getmodule(code)
    if dbg:
      stderr.flush()
      errSL('coven.is_code_path_targeted: {}:{} -> {} -> {}'.format(
        code.co_filename, code.co_name, module and module.__name__,
        (module and module.__name__) in targets))
    if module is None: return False # probably a python builtin; not traceable.
    # note: the module filename may not equal the code filename.
    # example: .../python3.5/collections/abc.py != .../python3.5/_collections_abc.py
    # thus the following check sometimes fires, but it seems acceptable.
    #if dbg and module.__file__ != code.co_filename:
    #  errSL(f'  note: module file: {module.__file__} != code file: {code.co_filename}')
    is_target = (module.__name__ in targets)
    return is_target

  def cove_global_tracer(g_frame, g_event, _g_arg_is_none):
    #errSL('GTRACE:', g_event, g_frame.f_lineno, g_frame.f_code.co_name)
    if g_event != 'call': return None
    code = g_frame.f_code
    path = code.co_filename
    try:
      is_target = file_name_filter[path]
    except KeyError:
      is_target = is_code_path_targeted(code)
      file_name_filter[path] = is_target

    if not is_target: return None # do not trace this scope.

    # the local tracer lives only as long as execution continues within the code block.
    # for a generator, this can be less than the lifetime of the frame,
    # which is saved and restored when resuming from a `yield`.
    edges = code_edges[code]
    prev_off  = OFF_BEGIN
    def cove_local_tracer(frame, event, arg):
      nonlocal prev_off
      line = frame.f_lineno
      off = frame.f_lasti
      #errSL(f'LTRACE: {event} {prev_off} -> {off} ({line}) {code.co_name}')
      if event in ('instruction', 'line'):
        edges.add((prev_off, off, line))
        prev_off = off
      return cove_local_tracer # local tracer keeps itself in place during its local scope.

    return cove_local_tracer # global tracer installs a new local tracer for every call.

  settraceinst(cove_global_tracer, 1)
  return code_edges


def gen_target_paths(targets, cmd_head, dbg):
  '''
  Given a list of target module names/paths and the path that serves as __main__,
  return a dictionary mapping targets to sets of paths.
  The values are sets to allow __main__ to map to multiple paths,
  which can occur during coalescing.
  Empty sets have meaning: they indicate a target which has no coverage.
  '''
  target_paths = {}
  for target in targets:
    if target == '__main__': # sys.modules['__main__'] points to coven; we want cmd_head.
      target_paths['__main__'] = {cmd_head}
    else:
      try: module = sys.modules[target]
      except KeyError: target_paths[target] = set()
      else: target_paths[target] = {module.__file__}
  if dbg:
    for t, p in sorted(target_paths.items()):
      errSL(f'gen_target_paths: {t} -> {p}')
  return target_paths


def write_coverage(output_path, target_paths, code_edges):
  data = {
    'target_paths': target_paths,
    'code_edges': dict(code_edges), # convert from defaultdict.
  }
  with open(output_path, 'wb') as f:
    marshal.dump(data, f)


def coalesce(trace_paths, arg_targets, args):
  target_paths = defaultdict(set)
  for arg_target in arg_targets:
    target_paths[arg_target]
  all_code_edges = defaultdict(set)
  for trace_path in trace_paths:
    try: f = open(trace_path, 'rb')
    except FileNotFoundError:
      exit(f'coven error: trace file not found: {trace_path}')
    with f: data = marshal.load(f)
    target_paths = data['target_paths']
    code_edges = data['code_edges']
    for target, paths in target_paths.items():
      if arg_targets and target not in arg_targets: continue
    for code, edges in code_edges.items():
      all_code_edges[code].update(edges)
  report(target_paths, all_code_edges, args=args)


def report(target_paths, code_edges, args):
  print('----------------')
  print('Coverage Report:')
  path_code_edges = gen_path_code_edges(code_edges)
  totals = Stats()
  for target, paths in sorted(target_paths.items()):
    if not paths:
      print(f'\n{target}: NEVER IMPORTED.')
      continue
    for path in sorted(paths):
      coverage = calculate_coverage(path=path, code_edges=path_code_edges[path], dbg=args.dbg)
      report_path(target=target, path=path, coverage=coverage, totals=totals, args=args)
  if len(target_paths) > 1:
    totals.describe('\nTOTAL', True if args.color else '')


def gen_path_code_edges(code_edges):
  'Group the traced code_edges by path.'
  d = defaultdict(dict)
  for code, edges in code_edges.items():
    d[code.co_filename][code] = edges
  return d


def calculate_coverage(path, code_edges, dbg):
  '''
  Calculate and return the coverage data structure,
  Which maps line numbers to (required, matched) tuples of sets of (src, dst, code).
  Each set contains Edge tuples.
  An Edge is (prev_offset, offset, code).
  A line is fully covered if (required <= traced).
  '''
  if dbg: errSL(f'\ncalculate_coverage: {path}:')

  all_codes = list(visit_nodes(start_nodes=code_edges, visitor=sub_codes))
  if dbg: all_codes.sort(key=lambda c: c.co_name)

  coverage = defaultdict(lambda: (set(), set()))
  def add_edges(edge_lines, code, cov_idx):
    for edge, lines in edge_lines.items():
      for line in lines:
        assert line > 0
        coverage[line][cov_idx].add((edge[0], edge[1], code))

  for code in all_codes:
    traced = code_edges.get(code, {})
    # infer all possible edges.
    # TODO: optimization: if not traces, do not bother analyzing code; instead just add fake required edges for each line start in code.
    req, opt = crawl_code_insts(path=path, code=code, dbg_name=dbg)
    if dbg == code.co_name:
      for edge in sorted(traced): err_edge('traced', edge, code)
    # match traced to inferred edges.
    raise_reqs = { edge[1] : (edge, lines) for edge, lines in req.items() if edge[0] == OFF_RAISED }
    raise_opts = { edge[1] for edge in opt if edge[0] == OFF_RAISED }
    matched = defaultdict(set) # expected exception edges that matched an actual traced edge.
    for src, dst, line in traced:
      edge = (src, dst)
      if edge in req:
        matched[edge].add(line)
      elif dst in raise_reqs:
        e, l = raise_reqs[dst]
        matched[e].update(l) # add all the lines (really just one line?) implied by the exception edge.
      elif not (edge in opt or dst in raise_opts):
        err_edge('UNEXPECTED:', edge, code)
        errSL(*raise_reqs)
    # assemble final coverage data by line.
    add_edges(req, code, COV_REQ)
    add_edges(matched, code, COV_MATCHED)
  return coverage


COV_REQ, COV_MATCHED = range(2)


def visit_nodes(start_nodes, visitor):
  remaining = set(start_nodes)
  visited = set()
  while remaining:
    node = remaining.pop()
    assert node not in visited
    visited.add(node)
    discovered = visitor(node)
    remaining.update(n for n in discovered if n not in visited)
  return visited


def sub_codes(code):
  return [c for c in code.co_consts if isinstance(c, CodeType)]


def enhance_inst(inst, off, line, is_line_start, stack):
  '''
  Add some useful fields to Instruction:
  * off: logical offset; the address of the first EXTENDED_ARG for instructions with preceding EXTENDED_ARG.
  * line: the preceding start_line if this instruction does not have one.
  * is_line_start: bool of starts_line, but also True when preceded by EXTENDED_ARG that has starts_line set.
  * stack: represents the approximate static scope of all live frames on the block stack.
  The remaining fields are flags that are set in certain cases.
  '''
  inst.off = off
  inst.line = line
  inst.is_line_start = is_line_start
  inst.stack = stack
  inst.is_SF_exc_opt = False
  inst.is_exc_match = False
  inst.is_exc_match_jmp_src = False
  inst.is_exc_match_jmp_dst = False


_begin_inst = Instruction(opname='_BEGIN', opcode=OP_BEGIN, arg=None, argval=None, argrepr=None, offset=OFF_BEGIN, starts_line=LINE_BEGIN, is_jump_target=False)
enhance_inst(_begin_inst, off=OFF_BEGIN, line=LINE_BEGIN, is_line_start=False, stack=())

_raised_inst = Instruction(opname='_RAISED', opcode=OP_RAISED, arg=None, argval=None, argrepr=None, offset=OFF_RAISED, starts_line=LINE_RAISED, is_jump_target=False)
enhance_inst(_raised_inst, off=OFF_RAISED, line=LINE_RAISED, is_line_start=False, stack=())


def crawl_code_insts(path, code, dbg_name):
  name = code.co_name
  dbg = (name == dbg_name)
  if dbg: errSL(f'\ncrawl code: {path}:{name}')

  insts = { OFF_BEGIN : _begin_inst, OFF_RAISED : _raised_inst } # offsets (accounting for EXTENDED_ARG) to Instructions.
  nexts = {} # instructions to prev instruction.
  prevs = {} # instructions to next instruction.

  # Step 1: scan raw instructions and assemble structure.

  blocks = [] # (op, dst) pairs.
  prev = _begin_inst
  exc_match_jmp_dsts = set()
  ext = None # first preceding EXTENDED_ARG.
  #^ EXTENDED_ARG (or several) can precede an actual instruction.
  #^ In this case, we use the first offset and starts_line but the final instruction.
  for inst in get_instructions(code):
    op = inst.opcode
    if op == EXTENDED_ARG and ext is None:
      ext = inst
    off = ext.offset if ext else inst.offset
    starts_line = ext.starts_line if ext else inst.starts_line
    is_line_start = bool(starts_line)
    line = starts_line or prev.line

    while blocks and blocks[-1][1] == off:
      blocks.pop()
      #^ According to cpython compile.c,
      #^ each block lifespan is terminated by POP_BLOCK, POP_EXCEPT, or END_FINALLY.
      #^ However there might be multiple pop instructions for a single block (in different branches),
      #^ So it is difficult te reconstruct.
      #^ Instead we just pretend that blocks span to their jump destination.
      #^ This is good enough, since the instructions between the actual terminator and the destination
      #^ are concerned with block management.

    if op in setup_opcodes:
      dst = inst.argval
      assert all(dst <= d for _, d in blocks)
      blocks.append((op, dst))
    enhance_inst(inst, off=off, line=line, is_line_start=is_line_start, stack=tuple(blocks))
    if op == EXTENDED_ARG:
      #if dbg: err_inst(inst)
      continue
    ext = None

    insts[off] = inst
    nexts[prev] = inst
    prevs[inst] = prev

    if op == COMPARE_OP and inst.argrepr == 'exception match':
      # This instruction is doing an exception match,
      # which will lead to a jump that results in the exception getting reraised.
      inst.is_exc_match = True
    if prev.is_exc_match:
      assert op == POP_JUMP_IF_FALSE
      inst.is_exc_match_jmp_src = True
      exc_match_jmp_dsts.add(inst.argval)
    if off in exc_match_jmp_dsts:
      inst.is_exc_match_jmp_dst = True

    prev = inst
    #if dbg: err_inst(inst)

  # Step 2: scan again to assemble dsts and add additional info.

  dsts = defaultdict(set)

  for inst in insts.values():
    op = inst.opcode
    if op in (OP_BEGIN, OP_RAISED): continue

    if inst.off == 0:
      dsts[_begin_inst].add(inst)

    if op not in stop_opcodes:
      dsts[inst].add(nexts[inst])

    if op in jump_opcodes:
      dsts[inst].add(insts[inst.argval])

    elif op in setup_exc_opcodes:
      dsts[_raised_inst].add(insts[inst.argval])
      #^ Enter the exception handler from an unknown exception source.
      #^ This makes matching harder because while initial raises are labeled with src=OFF_RAISED,
      #^ reraises do not get traced and so they have src offset of the END_FINALLY that reraises.
      #^ The solution is in the custom matching logic in calculate_coverage.
      # TODO: Perhaps it is possible to emit optional edges from reraising END_FINALLY?

    if op == BREAK_LOOP:
      dst_off = find_block_dst_off(inst, (SETUP_LOOP,))
      if not dst: raise Exception(f'{path}:{line}: off:{off}; BREAK_LOOP stack has no SETUP_LOOP block')
      dsts[inst].add(insts[dst_off])

    elif op == END_FINALLY:
      # END_FINALLY can either reraise an exception, or advance to the next instruction.
      # It advances in just two cases:
      # * a `with` __exit__ might return True, silencing an exception.
      #   In this case END_FINALLY is always preceded by WITH_CLEANUP_FINISH.
      # * TOS is None.
      # This does not appear sufficient for static analysis; I got only as far as:
      # * TOS is never None for an exception compare, which always returns True/False.
      # * In compilation of SETUP_FINALLY, a None is pushed,
      #   but I'm not sure that it is always TOS when END_FINALLY is reached.
      # For now, we just assume it can always advance, and add the jump dst where applicable.
      dst_off = find_block_dst_off(inst, (SETUP_ASYNC_WITH, SETUP_FINALLY, SETUP_WITH))
      if dst_off:
        dsts[inst].add(insts[dst_off])

    elif op == RAISE_VARARGS:
      dst_off = find_block_dst_off(inst, (SETUP_EXCEPT, SETUP_FINALLY))
      if dst_off: dsts[_raised_inst].add(insts[dst_off])

    elif op == RETURN_VALUE:
      dst_off = find_block_dst_off(inst, (SETUP_ASYNC_WITH, SETUP_FINALLY, SETUP_WITH))
      if dst_off: dsts[inst].add(insts[dst_off])

    elif op == SETUP_FINALLY:
      if is_SF_exc_opt(nexts[inst], insts, path, name):
        insts[inst.argval].is_SF_exc_opt = True

    elif op == YIELD_VALUE:
      dsts[_begin_inst].add(nexts[inst])

    elif op == YIELD_FROM:
      dsts[_begin_inst].add(inst)
      dsts[_raised_inst].add(nexts[inst])
      #^ Use the same hack as FOR_ITER to accommodate generators that emit raise instead of advance.
      #^ See emit_edges for explanation.



  srcs = defaultdict(set)
  for src, dst_set in dsts.items():
    for dst in dst_set:
      srcs[dst].add(src)

  # Step 3: find all arcs.
  starts_to_arcs = {} # maps first instructions to arcs.

  def find_arcs(inst):
    arc = [inst]
    dst_set = dsts[inst]
    while len(dst_set) == 1:
      for inst in dst_set: pass # advance; best way to get the single element out.
      if len(srcs[inst]) != 1: break
      arc.append(inst)
      dst_set = dsts[inst]
    a = tuple(arc)
    assert a
    assert a[0] not in starts_to_arcs
    starts_to_arcs[a[0]] = a
    return dst_set

  visit_nodes(start_nodes=(dsts[_begin_inst] | dsts[_raised_inst]), visitor=find_arcs)

  if dbg:
    for arc in sorted(starts_to_arcs.values(), key=arc_key):
      src_opts = [f'{src.off}:{"o" if is_arc_opt(src, arc, srcs, dsts) else "r"}' for src in sorted(srcs[arc[0]])]
      dst_offs = sorted(inst.off for inst in dsts[arc[-1]])
      errSL(TXT_D, 'arc:', ', '.join(src_opts), '=->', dst_offs, RST)
      for inst in arc:
        err_inst(inst)

  # Step 4: emit edges for each arc, taking care to represent lines as they will be traced.
  req = defaultdict(set) # maps edges to sets of lines.
  opt = defaultdict(set) # ditto.
  def add_edge(edge, line, is_opt):
    if dbg: err_edge(f'   {"opt" if is_opt else "req"}', edge, code)
    (opt if is_opt else req)[edge].add(line)

  def emit_edges(triple):
    src, src_line, is_src_opt = triple
    for start in dsts[src]:
      arc = starts_to_arcs[start]
      is_opt = is_arc_opt(src, arc, srcs, dsts)
      prev_line = src_line
      for prev, inst in zip((src,) + arc, arc):
        line = next_line(prev, inst, prev_line)
        assert line > 0
        prev_off = prev.off
        if src == prev and src.opcode == FOR_ITER and src.argval == inst.off:
          #^ We might get a normal edge when the loop ends, or a StopIteration exception edge.
          #^ The StopIteration exception lands at the FOR_ITER dst, not the SETUP_LOOP dst.
          #^ This is why setup_exc_opcodes excludes SETUP_LOOP; it's not the actual destination.
          #^ Since calculate_coverage can convert traced normal edges to expected exception edges,
          #^ emit an exception edge here to cover both cases,
          #^ but preserve the actual line of the FOR_ITER or else it will look confusing.
          prev_off = OFF_RAISED
        edge = (prev_off, inst.off)
        if is_opt:
          pass # TODO: switch back to required when we see "content" instructions.
        else:
          if prev.opcode == END_FINALLY and nexts[prev] == inst:
            # Because END_FINALLY is so hard to analyze, for now we treat any step to next as optional.
            is_opt = True
        add_edge(edge, line, (is_opt or (is_src_opt and src == prev)))
        prev_line = line
      yield (inst, line, is_opt)

  visit_nodes(start_nodes=[(_begin_inst, LINE_BEGIN, False), (_raised_inst, LINE_RAISED, False)], visitor=emit_edges)

  return req, opt


def inst_key(inst): return inst.off

def arc_key(arc): return tuple(inst.off for inst in arc)


def next_line(inst, nxt, line):
  '''
  Our interpretation of the line number tricks described in lnotabs_notes.txt.
  Note: the last clause is not an lnotabs rule, but necessary for exception and yield resume edges.
  '''
  return nxt.line if (nxt.is_line_start or nxt.off < inst.off or line < 0) else line


def err_inst(inst, prefix=''):
  op = inst.opcode
  line = inst.starts_line or ('^' if inst.is_line_start else '')
  off = inst.off
  sym = ' '
  if inst.is_SF_exc_opt: sym = '~'
  if inst.is_exc_match_jmp_src: sym = '^' # jump.
  if inst.is_exc_match_jmp_dst: sym = '_' # land.
  dst = ('DST' if inst.is_jump_target else '   ')
  if op == END_FINALLY:     stop = 'end?'
  elif op == RETURN_VALUE:  stop = 'ret?'
  elif op in stop_opcodes:  stop = 'stop'
  else: stop = '    '
  if op in jump_opcodes:
    target = f'jump {inst.argval:4}'
  elif op in setup_opcodes:
    target = f'push {inst.argval:4}'
  else: target = ''
  stack = ''.join(push_abbrs[op] for op, _ in inst.stack)
  arg = f'to {inst.arg} (abs)' if inst.opcode in hasjabs else inst.argrepr
  errSL(f'{prefix}  line:{line:>4}  off:{off:>4} {dst:4} {sym} {stop} {target:9}  {stack:8}  {inst.opname:{onlen}} {arg}')


def find_block_dst_off(inst, match_ops):
  for block_op, dst_off in reversed(inst.stack):
    if block_op in match_ops:
      return dst_off
  return None


def is_SF_exc_opt(nxt, insts, path, code_name):
  '''
  Some SETUP_FINALLY imply a required exception edge, but others do not.

  For a try/except/finally, we only expect coverage to raise an exception in the try clause.
  We do not expect an exception in the except clause, or an unhandled exception.
  In other words, typical code catches some but not all possible exceptions.
  In this case SETUP_FINALLY should not emit an exception edge,
  because the SETUP_EXCEPT that immediately follows will emit its own exception edge.

  However, for a try/finally, there is no SETUP_EXCEPT, so SETUP_FINALLY does emit.

  Unfortunately for us there is a pathological case, "TF-TE-R":
  try:
    try: ...
    except: ...
    <code>
  finally: ...
  The compiler emits consecutive SETUP_FINALLY, SETUP_EXCEPT for this code as well,
  but unlike TEF, here we *do* want a second exception edge in case <code> raises.

  This heuristic attempts to detect TEF, as distinct from TF-TE.
  '''
  if nxt.opcode == SETUP_EXCEPT: # looks like TEF, but might be TF-TE.
    # Inspect the destination of nested SETUP_EXCEPT.
    exc_dst_inst = insts[nxt.argval]
    op = exc_dst_inst.opcode
    if op == DUP_TOP: return True # TEF; exception is optional.
    if op == POP_TOP: return False # TF-TE; exception is required.
    errSL(f'coven WARNING: is_SETUP_FINALLY_exc_opt: {path}:{code_name}: heuristic failed on exc_dst_inst opcode: {exc_dst_inst}')
  return False


def is_arc_opt(src, arc, srcs, dsts):
  return (
    is_arc_opt_SF_raise(src, arc) or
    is_arc_unhandled_exc_reraise(src, arc) or
    is_arc_exc_as_cleanup(src, arc) or
    is_arc_with_cleanup(arc) or
    is_arc_join_return_none(arc, srcs)
  )


def is_arc_opt_SF_raise(src, arc):
  return src == _raised_inst and arc[0].is_SF_exc_opt


def is_arc_unhandled_exc_reraise(src, arc):
  '''
  Track jumps predicated on a preceding COMPARE_OP performing 'exception match'.
  The failure branch treated as optional, because all possible exceptions are not usually covered.
  '''
  return (
    (src.is_exc_match_jmp_src and arc[0].is_exc_match_jmp_dst) or
    src.is_exc_match_jmp_dst
  )


def is_arc_exc_as_cleanup(src, arc):
  '''
  CPython's compile.c:compiler_try_except emits a nested SETUP_FINALLY for `except _ as <name>`,
  and generates finally code to delete <name>.
  This instruction sequence is easy to recognize, but is just a heuristic that may fail.
  Note: for further discrimination, the final three instructions in the preceding block are:
    POP_BLOCK,
    POP_EXCEPT,
    (LOAD_CONST, None).
  '''
  return src == _raised_inst and match_insts(arc, (
    (LOAD_CONST, None),
    STORE_FAST,
    DELETE_FAST,
    END_FINALLY))


def is_arc_with_cleanup(arc):
  '''
  With statements do not typically get the exception case exercised.
  TODO: this heuristic may need to be broadened.
  '''
  return match_insts(arc, (
    WITH_CLEANUP_START,
    WITH_CLEANUP_FINISH,
    END_FINALLY))


def is_arc_join_return_none(arc, srcs):
  '''
  `return None` is implied by functions without an explicit final return statement,
  which causes the last line of a branch to represent the implicit join-and-return arc.
  This looks confusing, because it shows partial coverage of the branch when there is None.
  '''
  return len(arc) == 2 and len(srcs[arc[0]]) > 1 and match_insts(arc, (
    (LOAD_CONST, None),
    RETURN_VALUE))


def match_insts(insts, exps):
  if len(insts) < len(exps): return False
  return all(match_inst(*p) for p in zip(insts, exps))


def match_inst(inst, exp):
  if isinstance(exp, tuple):
    op, arg = exp
    return inst.opcode == op and inst.argval == arg
  else:
    return inst.opcode == exp


class Stats:

  def __init__(self):
    self.lines = 0
    self.trivial = 0
    self.traceable = 0
    self.covered = 0
    self.ignored = 0
    self.ignored_but_covered = 0
    self.not_covered = 0

  def add(self, stats):
    self.lines += stats.lines
    self.trivial += stats.trivial
    self.traceable += stats.traceable
    self.covered += stats.covered
    self.ignored += stats.ignored
    self.ignored_but_covered += stats.ignored_but_covered
    self.not_covered += stats.not_covered

  def describe_stat(self, name, val, c):
    colors = {
      'trivial' : c and TXT_L,
      'ignored' : c and TXT_C,
      'ignored_but_covered' : c and TXT_Y,
      'not_covered' : c and TXT_R,
    }
    color = colors.get(name, '') if val > 0 else ''
    rst = RST if color else ''
    display_name = name.replace('_', ' ')
    return f'{color}{val} {display_name}{rst}'

  def describe(self, label, c):
    s = self
    print(label, ': ', '; '.join(self.describe_stat(name, val, c) for name, val in self.__dict__.items()), '.', sep='')


def report_path(target, path, coverage, totals, args):

  line_texts = [text.rstrip() for text in open(path).readlines()]
  ignored_lines, explicitly_ignored_lines = calc_ignored_lines(line_texts)

  covered_lines = set() # line indices that are perfectly covered.
  ign_cov_lines = set()
  not_cov_lines = set()

  for line, (required, matched) in coverage.items():
    if matched >= required:
      if line in explicitly_ignored_lines:
        ign_cov_lines.add(line)
      else:
        covered_lines.add(line)
    elif line not in ignored_lines:
      not_cov_lines.add(line)

  problem_lines = ign_cov_lines | not_cov_lines

  length = len(line_texts)
  stats = Stats()
  stats.lines = length
  stats.trivial = max(0, length - len(coverage))
  stats.traceable = len(coverage)
  stats.covered = len(covered_lines)
  stats.ignored_but_covered = len(ign_cov_lines)
  stats.not_covered = len(not_cov_lines)
  stats.ignored = len(ignored_lines - covered_lines - ign_cov_lines - not_cov_lines)
  totals.add(stats)

  c = True if args.color else ''
  rel_path = path_rel_to_current_or_abs(path)
  label = f'\n{target}: {rel_path}'
  if not problem_lines:
    stats.describe(label, c)
    return

  RST1 = c and RST
  TXT_B1 = c and TXT_B
  TXT_C1 = c and TXT_C
  TXT_D1 = c and TXT_D
  TXT_G1 = c and TXT_G
  TXT_L1 = c and TXT_L
  TXT_M1 = c and TXT_M
  TXT_R1 = c and TXT_R
  TXT_Y1 = c and TXT_Y
  print(label, ':', sep='')
  if args.show_all:
    reported_lines = range(1, length + 1) # entire document, 1-indexed.
  else:
    reported_lines = sorted(problem_lines)
  ranges = line_ranges(reported_lines, before=4, after=1, terminal=length+1)
  for r in ranges:
    if r is None:
      print(f'{TXT_D1} ...{RST1}')
      continue
    for line in r:
      text = line_texts[line - 1] # line is a 1-index.
      color = RST1
      sym = ' '
      needs_dbg = False
      if line not in coverage: # trivial.
        color = TXT_L1
      else:
        required, matched = coverage[line]
        if line in ign_cov_lines:
          color = TXT_Y1
          sym = '?'
        elif line in ignored_lines:
          color = TXT_C1
          sym = '|'
        elif line in not_cov_lines:
          color = TXT_R1
          if matched:
            sym = '%'
            needs_dbg = True
          else: # no coverage.
            sym = '!'
        else: assert line in covered_lines
      print(f'{TXT_D1}{line:4} {color}{sym} {text}{RST1}'.rstrip())
      if args.dbg and needs_dbg:
        #print(f'     {TXT_B1}^ required:{len(required)} traced:{len(traced)}.{RST1}')
        err_cov_set(f'{TXT_D1}{line:4} {TXT_B1}-', required - matched, args.dbg)
        err_cov_set(f'{TXT_D1}{line:4} {TXT_B1}=', matched, args.dbg)
  stats.describe(label, c)


def path_rel_to_current_or_abs(path: str) -> str:
  ap = abs_path(path)
  ac = abs_path('.')
  comps = path_comps(ap)
  prefix = path_comps(ac)
  if comps == prefix:
    return '.'
  if prefix == comps[:len(prefix)]:
    return path_join(*comps[len(prefix):])
  return ap

def path_comps(path: str):
  np = normalize_path(path)
  if np == '/': return ['/']
  assert not np.endswith('/')
  return [comp or '/' for comp in np.split(os.sep)]


indent_and_ignored_re = re.compile(r'''(?x:
^ (\s*) # capture leading space.
( .* (?P<directive> \#!cov-ignore )
| assert\b
| if \s+ __name__ \s* == \s* ['"]__main__['"] \s* :
)?
)''')

def calc_ignored_lines(line_texts):
  explicit = set()
  implicit = set()
  indent = -1
  is_directive = False
  for line, text in enumerate(line_texts, 1):
    m = indent_and_ignored_re.match(text)
    ind = m.end(1) - m.start(1)
    if m.lastindex == 2: # matched one of the ignore triggers.
      is_directive = bool(m.group('directive')) # explicit ignore.
      (explicit if is_directive else implicit).add(line)
      indent = ind
    elif -1 < indent < ind:
      (explicit if is_directive else implicit).add(line)
    else:
      indent = -1
  return (explicit | implicit), explicit


def line_ranges(iterable, before, after, terminal):
  'Group individual line numbers (1-indexed) into chunks.'
  assert terminal > 0
  it = iter(iterable)
  try:
    i = next(it)
    assert i > 0
  except StopIteration: return
  start = i - before
  end = i + after + 1
  for i in it:
    assert i > 0
    # +1 bridges chunks that would otherwise elide a single line, appearing replaced by '...'.
    if end + 1 < i - before:
      yield range(max(1, start), min(end, terminal))
      yield None # interstitial causes '...' to be printed.
      start = i - before
    end = i + after + 1
  yield range(max(1, start), min(end, terminal))


def fmt_edge(edge, code):
  src, dst = edge[:2]
  return f'off: {src:4} -> {dst:4}  {code.co_name}'


def errSL(*items): print(*items, file=stderr)

def errLSSL(*items): print(*items, sep='\n  ', file=stderr)


def err_edge(label, edge, code):
  errSL(label, fmt_edge(edge, code))


def err_cov_set(label, cov_set, dbg_name):
  for src, dst, code in sorted(cov_set, key=lambda t: (t[2].co_name, t[0], t[1])):
    if not dbg_name or dbg_name == code.co_name:
      errSL(f'{label} {src:4} -> {dst:4}  {code.co_name}')


# Opcode information.

onlen = max(len(name) for name in opname)

# absolute jump codes.
#errSL('JMP ABS:', *sorted(opname[op] for op in hasjabs))
CONTINUE_LOOP         = opmap['CONTINUE_LOOP']
JUMP_ABSOLUTE         = opmap['JUMP_ABSOLUTE']
JUMP_IF_FALSE_OR_POP  = opmap['JUMP_IF_FALSE_OR_POP']
JUMP_IF_TRUE_OR_POP   = opmap['JUMP_IF_TRUE_OR_POP']
POP_JUMP_IF_FALSE     = opmap['POP_JUMP_IF_FALSE']
POP_JUMP_IF_TRUE      = opmap['POP_JUMP_IF_TRUE']

# relative jump codes.
#errSL('JMP REL:', *sorted(opname[op] for op in hasjrel))
FOR_ITER              = opmap['FOR_ITER']
JUMP_FORWARD          = opmap['JUMP_FORWARD']
SETUP_ASYNC_WITH      = opmap['SETUP_ASYNC_WITH']
SETUP_EXCEPT          = opmap['SETUP_EXCEPT']
SETUP_FINALLY         = opmap['SETUP_FINALLY']
SETUP_LOOP            = opmap['SETUP_LOOP']
SETUP_WITH            = opmap['SETUP_WITH']

# other opcodes of interest.
BREAK_LOOP            = opmap['BREAK_LOOP']
COMPARE_OP            = opmap['COMPARE_OP']
DELETE_FAST           = opmap['DELETE_FAST']
DUP_TOP               = opmap['DUP_TOP']
END_FINALLY           = opmap['END_FINALLY']
EXTENDED_ARG          = opmap['EXTENDED_ARG']
LOAD_CONST            = opmap['LOAD_CONST']
POP_BLOCK             = opmap['POP_BLOCK']
POP_EXCEPT            = opmap['POP_EXCEPT']
POP_TOP               = opmap['POP_TOP']
RAISE_VARARGS         = opmap['RAISE_VARARGS']
RETURN_VALUE          = opmap['RETURN_VALUE']
STORE_FAST            = opmap['STORE_FAST']
WITH_CLEANUP_FINISH   = opmap['WITH_CLEANUP_FINISH']
WITH_CLEANUP_START    = opmap['WITH_CLEANUP_START']
YIELD_FROM            = opmap['YIELD_FROM']
YIELD_VALUE           = opmap['YIELD_VALUE']

# `hasjrel` includes the SETUP_* ops, which do not actually branch on execution.
jump_opcodes = {
  CONTINUE_LOOP,
  FOR_ITER,
  JUMP_ABSOLUTE,
  JUMP_IF_FALSE_OR_POP,
  JUMP_IF_TRUE_OR_POP,
  POP_JUMP_IF_FALSE,
  POP_JUMP_IF_TRUE,
  JUMP_FORWARD,
}

# the following opcodes never advance to the next instruction.
stop_opcodes = {
  BREAK_LOOP,
  CONTINUE_LOOP,
  JUMP_ABSOLUTE,
  JUMP_FORWARD,
  RAISE_VARARGS,
  RETURN_VALUE,
  YIELD_VALUE,
  YIELD_FROM,
}

# the following opcodes trigger 'return' events.
return_opcodes = {
  RETURN_VALUE,
  YIELD_FROM,
  YIELD_VALUE,
}

# These codes push a block that specifies a block destination.
setup_opcodes = {
  SETUP_ASYNC_WITH,
  SETUP_EXCEPT,
  SETUP_FINALLY,
  SETUP_LOOP,
  SETUP_WITH,
}

# These codes expect an exception edge that lands at the block destination.
setup_exc_opcodes = {
  SETUP_ASYNC_WITH,
  SETUP_EXCEPT,
  SETUP_FINALLY,
  SETUP_WITH
}

push_abbrs = {
  SETUP_ASYNC_WITH: 'A',
  SETUP_EXCEPT:     'E',
  SETUP_FINALLY:    'F',
  SETUP_LOOP:       'L',
  SETUP_WITH:       'W',
}

pop_block_opcodes = {
  BREAK_LOOP,
  END_FINALLY,
  POP_BLOCK,
  POP_EXCEPT,
}

RST = '\x1b[0m'
TXT_B = '\x1b[34m'
TXT_C = '\x1b[36m'
TXT_D = '\x1b[30m'
TXT_G = '\x1b[32m'
TXT_L = '\x1b[37m'
TXT_M = '\x1b[35m'
TXT_R = '\x1b[31m'
TXT_Y = '\x1b[33m'

if __name__ == '__main__': main()

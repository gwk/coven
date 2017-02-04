#!/usr/bin/env python3

# Note: any modules imported prior to the calls to install_trace and run_path
# will not report coverage fully, because their <module> code objects will not be captured.
# Therefore, we import only stdlib modules that we need.
import sys; assert sys.version_info >= (3, 6, 0)
import marshal
import os
import os.path
import re
from collections import defaultdict
from dis import Instruction, findlinestarts, get_instructions, hasjabs, hasjrel, opname, opmap
from argparse import ArgumentParser
from inspect import getmodule
from os.path import abspath as abs_path, join as path_join, normpath as normalize_path
from runpy import run_path
from sys import exc_info, settrace, stderr, stdout
from types import CodeType


def main():
  arg_parser = ArgumentParser(description='cove: code coverage harness.')
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
  'NOTE: this must be called before importing any module that we might wish to trace with cove.'
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
  #if dbg: errSL('cove untraceable modules (imported prior to `install_trace`):', sorted(sys.modules.keys()))
  try:
    run_path(cmd_head, run_name='__main__')
  except FileNotFoundError as e:
    exit('cove error: could not find command to run: {!r}'.format(cmd_head))
  except SystemExit as e:
    exit_code = e.code
  except BaseException:
    from traceback import TracebackException
    exit_code = 1 # exit code that Python returns when an exception raises to toplevel.
    # format the traceback exactly as it would appear when run without cove.
    tbe = TracebackException(*exc_info())
    scrub_traceback(tbe)
    print(*tbe.format(), sep='', end='', file=stderr)
  finally:
    settrace(None)
    stdout.flush()
    stderr.flush()
  sys.argv = orig_argv
  target_paths = gen_target_paths(targets, cmd_head, dbg=args.dbg)
  if output_path:
    write_coverage(output_path=output_path, target_paths=target_paths, code_edges=code_edges)
    exit(exit_code) # TODO: move out of if?
  else:
    report(target_paths=target_paths, code_edges=code_edges, args=args)


def scrub_traceback(tbe):
  'Remove frames from TracebackException object that refer to cove, rather than the child process under examination.'
  stack = tbe.stack # StackSummary is a subclass of list.
  if not stack or 'cove' not in stack[0].filename: return # not the root exception.
  #^ TODO: verify that the above is sufficiently strict,
  #^ while also covering both the installed entry_point and the local dev cases.
  del stack[0]
  while stack and stack[0].filename.endswith('runpy.py'): del stack[0] # remove cove runpy.run_path frames.


# Fake instruction/line offsets.
OFF_BEGIN = -1
OFF_RETURN = -2
OFF_RAISED = -3


def install_trace(targets, dbg):
  'NOTE: this must be called before importing any module that we might wish to trace with cove.'

  code_edges = defaultdict(set)
  file_name_filter = {}

  def is_code_path_targeted(code):
    module = getmodule(code)
    if dbg:
      stderr.flush()
      errSL('cove.is_code_path_targeted: {}:{} -> {} -> {}'.format(
        code.co_filename, code.co_name, module and module.__name__,
        (module and module.__name__) in targets))
    if module is None: return False # probably a python builtin; not traceable.
    # note: the module filename may not equal the code filename.
    # example: .../python3.5/collections/abc.py != .../python3.5/_collections_abc.py
    # thus the following check sometimes fires, but it seems acceptable.
    #if dbg and module.__file__ != code.co_filename:
    #  errSL('  note: module file: {} != code file: {}'.format(module.__file__, code.co_filename))
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
    prev_line = OFF_BEGIN
    prev_off  = OFF_BEGIN
    def cove_local_tracer(frame, event, arg):
      nonlocal prev_line, prev_off
      line = frame.f_lineno
      off = frame.f_lasti
      #errSL(f'LTRACE: {event} {prev_off} -> {off} ({prev_line} -> {line}) {code.co_name}')
      if event in ('instruction', 'line'):
        edges.add((prev_off, off, prev_line, line))
        prev_line = line
        prev_off = off
      elif event == 'return':
        prev_line = OFF_RETURN
        prev_off  = OFF_RETURN
      elif event == 'exception':
        prev_line = OFF_RAISED
        prev_off  = OFF_RAISED
      else: raise ValueError(event)
      return cove_local_tracer # local tracer keeps itself in place during its local scope.

    return cove_local_tracer # global tracer installs a new local tracer for every call.

  try: settrace(cove_global_tracer, 'instruction')
  except TypeError: exit('cove error: sys.settrace does not support instruction tracing (private patch)')

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
    if target == '__main__': # sys.modules['__main__'] points to cove; we want cmd_head.
      target_paths['__main__'] = {cmd_head}
    else:
      try: module = sys.modules[target]
      except KeyError: target_paths[target] = set()
      else: target_paths[target] = {module.__file__}
  if dbg:
    for t, p in sorted(target_paths.items()):
      errSL('gen_target_paths: {} -> {}'.format(t, p))
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
      exit('cove error: trace file not found: {}'.format(trace_path))
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
      print('\n{}: NEVER IMPORTED.'.format(target))
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
  Which maps line numbers to (required, optional, traced) triples of sets of (src, dst, code).
  Each set contains Edge tuples.
  An Edge is (prev_offset, offset, code).
  A line is fully covered if (required <= traced <= required&optional).
  However there are additional relaxation semantics.
  '''
  if dbg: errSL(f'\calculate_coverage: {path}:')

  all_codes = list(visit_nodes(start_nodes=code_edges, visitor=sub_codes))
  if dbg: all_codes.sort(key=lambda c: c.co_name)

  coverage = defaultdict(lambda: (set(), set(), set()))
  def add_edges(edges, code, cov_idx):
    for edge in edges:
      line = edge[3]
      assert line >= 0
      coverage[line][cov_idx].add((edge[0], edge[1], code))

  for code in all_codes:
    traced = code_edges.get(code, {})
    # generate all possible traces.
    # TODO: optimization: if not traces, do not bother analyzing code; instead just add fake required edges for each line start in code.
    req, opt = crawl_code_insts(path=path, code=code, dbg_name=dbg)
    if dbg == code.co_name:
      for edge in sorted(traced): err_edge('traced', edge, code)
    reduce_edges(req, opt, traced)
    # assemble final coverage data by line.
    add_edges(req,    code, COV_REQ)
    add_edges(opt,    code, COV_OPT)
    add_edges(traced, code, COV_TRACED)

  return coverage


COV_REQ, COV_OPT, COV_TRACED = range(3)


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


def enhance_inst(inst, off, line):
  '''
  Add some useful fields to Instruction:
  * off: logical offset; the address of the first EXTENDED_ARG for instructions with preceding EXTENDED_ARG.
  * line: the preceding start_line if this instruction does not have one.
  The remaining fields are flags that are set in certain cases.
  '''
  inst.off = off
  inst.line = line
  inst.is_req = False
  inst.is_exc_match = False
  inst.is_exc_jmp_src = False
  inst.is_exc_jmp_dst = False


_begin_inst = Instruction(opname='_START', opcode=-1, arg=None, argval=None, argrepr=None, offset=OFF_BEGIN, starts_line=OFF_BEGIN, is_jump_target=False)
enhance_inst(_begin_inst, off=OFF_BEGIN, line=OFF_BEGIN)

def crawl_code_insts(path, code, dbg_name):
  name = code.co_name
  dbg = (name == dbg_name)
  if dbg: errSL(f'\ncrawl code: {path}:{name}')

  insts = {} # offsets (accounting for EXTENDED_ARG) to Instructions.
  nexts = {} # to prev Instruction.
  prevs = {} # to next Instruction.
  #^ track jumps predicated on a preceding COMPARE_OP performing 'exception match'.
  #^ the jump dst is treated as optional, because all possible exceptions cannot be reasonably covered.
  entry_offs = [0] # crawl starting points.

  # Step 1: scan over instructions and assemble structure.

  blocks = [] # (op, dst) pairs.
  prev = _begin_inst
  exc_jmp_dsts = set()
  ext_off = None # address of first EXTENDED_ARG.
  #^ EXTENDED_ARG (or several) can precede an actual instruction.
  #^ In this case, we use the first offset but the final instruction.
  for inst in get_instructions(code):
    op = inst.opcode
    if op == EXTENDED_ARG:
      if ext_off is None:
        ext_off = inst.offset
      continue
    if ext_off:
      off = ext_off # change the offset to represent "logical offset".
      ext_off = None
    else:
      off = inst.offset
    enhance_inst(inst, off=off, line=(inst.starts_line or prev.line))
    insts[off] = inst
    nexts[prev.off] = inst
    prevs[off] = prev

    if prev.opcode == YIELD_VALUE:
      entry_offs.append(off)
    if op == YIELD_FROM:
      entry_offs.append(off)

    # calculate if this instruction is doing an exception match,
    # which will lead to a jump that results in the exception getting reraised.
    if op == COMPARE_OP and inst.argrepr == 'exception match':
      inst.is_exc_match = True
      inst.is_req = True # any exception match written in the source should be covered.
    if prev.is_exc_match:
      assert op == POP_JUMP_IF_FALSE
      inst.is_exc_jmp_src = True
      exc_jmp_dsts.add(inst.argval)
    if off in exc_jmp_dsts:
      inst.is_exc_jmp_dst = True

    prev = inst # update; no longer valid for remainder of loop body.

    while blocks:
      # We assume here that the runtime lifespan of a block is equivalent to its static span.
      # According to cpython compile.c it's slightly more complicated;
      # each block lifespan is terminated by POP_BLOCK, POP_EXCEPT, or END_FINALLY.
      # however there might be multiple pop instructions for a single block (in different branches),
      # so it is difficult te reconstruct.
      # this heuristic is the best we can do for now.
      # TODO: should this be an if instead of a while? verify pre and post condition.
      o, d = blocks[-1]
      assert d >= off
      if d > off: break
      blocks.pop()

    if op in push_block_opcodes:
      dst = inst.argval
      assert all(dst < d for _, d in blocks)
      blocks.append((op, dst))
    inst.stack = tuple(blocks)

    if dbg: err_inst(inst)

  # Step 2: walk the code graph.
  req = set()
  opt = set()

  visited = set()
  def find_traceable_edges(prev_line, prev_off, off, is_req):
    inst = insts[off]
    assert inst.off == off
    is_req = is_req or inst.is_req
    args = (prev_line, prev_off, off, is_req)
    if args in visited or (not is_req and (prev_line, prev_off, off, True) in visited): return
    visited.add(args)
    assert off or not inst.is_jump_target # otherwise boolean tests on jump dsts are unsafe.

    op = inst.opcode
    starts_line = inst.starts_line
    if starts_line or off < prev_off or prev_line < 0:
      #^ Our interpretation of the line number tricks described in lnotabs_notes.txt.
      #^ Note: the last clause is not an lnotabs rule, but necessary for exception and yield resume edges.
      line = inst.line
    else:
      line = prev_line # jumped forward to an instruction that is not a line start, so display as prev line.

    edge = (prev_off, off, prev_line, line)
    if dbg: err_edge(f'   {"req" if is_req else "opt"}', edge, code)
    (req if is_req else opt).add(edge)

    if op == BREAK_LOOP:
      for block_op, dst in reversed(inst.stack):
        if block_op == SETUP_LOOP:
          find_traceable_edges(line, off, dst, is_req)
          return
      else: raise Exception(f'{path}:{line}: off:{off}; BREAK_LOOP stack has no SETUP_LOOP block')
    elif op == FOR_ITER:
      find_traceable_edges(line, OFF_RAISED, inst.argval, is_req)
      #^ We might get a normal edge when the loop ends, or an exception edge.
      #^ Since reduce_edges can convert normal edges to exception edges, just emit the latter.
      #^ Note: unlike other exception edges we specify prev_line.
    elif op == SETUP_EXCEPT:
      # Enter the exception handler from an unknown exception source.
      # This makes matching harder because while we can trace raises with src=OFF_RAISED,
      # eraises do not get traced and so they have src offset of the END_FINALLY that reraises.
      # The solution is to use reduce_exc_edges().
      find_traceable_edges(OFF_RAISED, OFF_RAISED, inst.argval, is_req)
    elif op == SETUP_FINALLY:
      if is_SETUP_FINALLY_exc_pad(insts, prevs, nexts, inst, path, name):
        # omit outer exception edge for code that looks like TEF, because inner EXCEPT block will catch it.
        find_traceable_edges(OFF_RAISED, OFF_RAISED, inst.argval, is_req)
    elif op == END_FINALLY:
      # The semantics of END_FINALLY are complicated.
      # END_FINALLY can either reraise an exception
      # or in two cases continue execution to the next instruction:
      # * a `with` __exit__ might return True, silencing an exception.
      #   In this case END_FINALLY is always preceded by WITH_CLEANUP_FINISH.
      # * TOS is None.
      #   * Never None for an exception compare, which always returns True/False.
      #   * Beyond that, hard to say.
      #   * In compilation of SETUP_FINALLY, a None is pushed, but might not remain as TOS.
      for block_op, block_dst in reversed(inst.stack):
        if block_op == SETUP_FINALLY: # create an edge to next handler dst.
          find_traceable_edges(line, off, block_dst, is_req)
          return # TODO: it may be that this case can also step to next.
      # TODO: handle WITH_CLEANUP_FINISH case.
      if inst.is_exc_jmp_dst: # can never step to next.
        return

    # note: we currently use recursion to explore the control flow graph.
    # this could hit the recursion limit for large code objects, and is probably slow.
    # alternatively we could:
    # * turn the second call into a fake tail call using a while loop around this whole function body;
    # * use visit_nodes, which would break ordering of dbg output.
    if op not in stop_opcodes: # normal interpreter step forward.
      find_traceable_edges(line, off, nexts[off].off, is_req)
    if op in jump_opcodes:
      jmp = inst.argval # argval accounts for absolute vs relative offsets.
      find_traceable_edges(line, off, jmp, (is_req and not inst.is_exc_jmp_src))

  for off in entry_offs:
    find_traceable_edges(-1, -1, off, is_req=True)

  return req, opt


def err_inst(inst):
  op = inst.opcode
  line = inst.starts_line or ''
  off = inst.offset
  exc_match = ' '
  if inst.is_exc_jmp_src: exc_match = '^' # jump.
  if inst.is_exc_jmp_dst: exc_match = '_' # land.
  dst = ('DST' if inst.is_jump_target else '   ')
  stop = 'stop' if op in stop_opcodes else '    '
  if op in jump_opcodes:
    target = f'jump {inst.argval:4}'
  elif op in push_block_opcodes:
    target = f'push {inst.argval:4}'
  else: target = ''
  stack = ''.join(push_abbrs[op] for op, _ in inst.stack)
  arg = 'to {} (abs)'.format(inst.arg) if inst.opcode in hasjabs else inst.argrepr
  errSL(f'  line:{line:>4}  off:{off:>4} {dst:4} {exc_match} {stop} {target:9}  {stack:8}  {inst.opname:{onlen}} {arg}')


def is_SETUP_FINALLY_exc_pad(insts, prevs, nexts, inst, path, code_name):
  '''
  Some SETUP_FINALLY imply an expected exception edge, but others do not.

  For a try/except/finally, we do not expect coverage for the case where
  an exception is raised but does not match the except clause,
  because typical code catches some but not all possible exceptions.
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
  If it fails, than any exception raised by TF-TE's <code> will be flagged as an impossible edge.

  Separately, compile.c:compiler_try_except emits a nested SETUP_FINALLY for `except _ as <name>`.
  It generates finally code to delete <name>.
  This instruction sequence appears easy to recognize, but is again just a heuristic that may fail.
  '''
  assert inst.opcode == SETUP_FINALLY
  dst_off = inst.argval
  next_inst = nexts[inst.off]
  if next_inst.opcode == SETUP_EXCEPT:
    # looks like TEF, but might be TF-TE.
    # TODO: this heuristic may need work!
    # Inspect the destination of nested SETUP_EXCEPT.
    exc_dst_inst = insts[next_inst.argval]
    op = exc_dst_inst.opcode
    if op == DUP_TOP: return False # TEF.
    if op == POP_TOP: return True # TF-TE.
    errSL(f'cove WARNING: is_SETUP_FINALLY_exc_pad: {path}:{code_name}: heuristic failed on exc_dst_inst opcode: {exc_dst_inst}')
    return False # if the code does actually raise it will be flagged as impossible.

  # `except _ as <name>` heuristic looks for a particular cleanup sequence.
  dst_inst = insts[dst_off]
  if match_insts(dst_inst, prevs, nexts,
    exp_prev=(
      POP_BLOCK,
      POP_EXCEPT,
      (LOAD_CONST, None)),
    expected=(
      (LOAD_CONST, None), # inst.
      STORE_FAST,
      DELETE_FAST,
      END_FINALLY)):
    return False # not an exception that would be reasonably covered.

  return True


def match_insts(inst, prevs, nexts, exp_prev, expected):
  p = prevs[inst.off]
  for exp in reversed(exp_prev):
    if not match_inst(p, exp): return False
    try: p = prevs[p.off]
    except KeyError: return False
  n = inst
  for exp in expected:
    if not match_inst(n, exp): return False
    try: n = nexts[n.off]
    except KeyError: return False
  return True


def match_inst(inst, exp):
  if isinstance(exp, tuple):
    op, arg = exp
    return inst.opcode == op and inst.argval == arg
  else:
    return inst.opcode == exp


def reduce_edges(req, opt, traced):
  '''
  Transform the three sets as necessary so that later set comparison operations will calculate
  per-line coverage appropriately.
  '''
  opt.difference_update(req) # might have overlap?
  possible = req | opt
  raise_dsts = { edge[1] for edge in possible if edge[0] == OFF_RAISED }
  def reduce_edge(edge):
    src, dst, _, dst_line = edge
    if edge not in possible and dst in raise_dsts:
      return (OFF_RAISED, dst, OFF_RAISED, dst_line)
    return edge
  traced_ = [reduce_edge(edge) for edge in traced]
  traced.clear()
  traced.update(traced_)


class Stats:

  def __init__(self):
    self.lines = 0
    self.trivial = 0
    self.traceable = 0
    self.covered = 0
    self.relaxed = 0
    self.ignored = 0
    self.ignored_but_covered = 0
    self.not_covered = 0

  def add(self, stats):
    self.lines += stats.lines
    self.trivial += stats.trivial
    self.traceable += stats.traceable
    self.covered += stats.covered
    self.relaxed += stats.relaxed
    self.ignored += stats.ignored
    self.ignored_but_covered += stats.ignored_but_covered
    self.not_covered += stats.not_covered

  def describe_stat(self, name, val, c):
    colors = {
      'trivial' : c and TXT_L,
      'relaxed' : c and TXT_G,
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

  covered_lines = set() # line indices that are perfectly covered.
  relaxed_lines = set() # line indices that are not perfectly covered but meet the relaxed requirement.
  not_cov_lines = set() # line indices that are not well covered.
  impossible_lines = set()

  for line, (required, optional, traced) in coverage.items():
    possible = required | optional
    unexpected = traced - possible
    if traced == possible:
      covered_lines.add(line)
    elif traced >= required and not unexpected:
      relaxed_lines.add(line)
    else:
      not_cov_lines.add(line)
      if unexpected:
        impossible_lines.add(line)

  ignored_lines = calc_ignored_lines(line_texts)
  ign_cov_lines = ignored_lines & covered_lines

  length = len(line_texts)
  stats = Stats()
  stats.lines = length
  stats.trivial = max(0, length - len(coverage))
  stats.traceable = len(coverage)
  stats.covered = len(covered_lines)
  stats.relaxed = len(relaxed_lines)
  stats.ignored = len(ignored_lines)
  stats.ignored_but_covered = len(ign_cov_lines)
  stats.not_covered = len(not_cov_lines - ignored_lines)
  totals.add(stats)

  c = True if args.color else ''
  rel_path = path_rel_to_current_or_abs(path)
  label = f'\n{target}: {rel_path}'
  if not (args.show_all or ign_cov_lines or not_cov_lines):
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
    reported_lines = sorted(impossible_lines | (not_cov_lines ^ ignored_lines))
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
        required, optional, traced = coverage[line]
        if line in covered_lines:
          if line in ign_cov_lines:
            color = TXT_Y1
            sym = '?'
          # else default symbol / color.
        elif line in relaxed_lines:
          color = TXT_G1
          sym = '~'
          needs_dbg = True
        else:
          assert line in not_cov_lines
          if line in impossible_lines:
            color = TXT_M1
            sym = '*'
            needs_dbg = True
          elif line in ignored_lines:
            color = TXT_C1
            sym = '|'
          else:
            color = TXT_R1
            if traced:
              sym = '%'
              needs_dbg = True
            else: # no coverage.
              sym = '!'
      print(f'{TXT_D1}{line:4} {color}{sym} {text}{RST1}'.rstrip())
      if args.dbg and needs_dbg:
        suffix = f'required:{len(required)} optional:{len(optional)} traced:{len(traced)}.'
        print(f'     {TXT_B1}^ {suffix}{RST1}')
        possible = required | optional
        err_cov_set(f'{TXT_D1}{line:4} {TXT_B1}-', required - traced)
        err_cov_set(f'{TXT_D1}{line:4} {TXT_B1}o', optional - traced)
        err_cov_set(f'{TXT_D1}{line:4} {TXT_B1}=', possible & traced)
        err_cov_set(f'{TXT_D1}{line:4} {TXT_B1}+', traced - possible)
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
( assert\b        # ignore assertions.
| .* \#no-cov! $  # ignore directive.
| if \s+ __name__ \s* == \s* ['"]__main__['"] \s* :
)?
)''')

def calc_ignored_lines(line_texts):
  ignored = set()
  ignored_indent = 0
  for line, text in enumerate(line_texts, 1):
    m = indent_and_ignored_re.match(text)
    indent = m.end(1) - m.start(1)
    if m.lastindex == 2: # matched one of the ignore triggers.
      ignored.add(line)
      ignored_indent = indent
    elif 0 < ignored_indent < indent:
      ignored.add(line)
    else:
      ignored_indent = 0
  return ignored


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
  src, dst, sl, dl = edge
  line = sl if (sl == dl) else f'{sl:4} -> {dl:4}'
  return f'off: {src:4} -> {dst:4}    line: {line:>12}  {code.co_name}'


def errSL(*items): print(*items, file=stderr)


def err_edge(label, edge, code, *tail):
  errSL(label, fmt_edge(edge, code), *tail)


def err_cov_set(label, cov_set):
  for src, dst, code in sorted(cov_set, key=lambda t: (t[2].co_name, t[0], t[1])):
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
YIELD_FROM            = opmap['YIELD_FROM']
YIELD_VALUE           = opmap['YIELD_VALUE']

# `hasjrel` includes the SETUP_* ops, which do not actually branch on execution.
# FOR_ITER is also omitted because we handle it manually.
jump_opcodes = {
  CONTINUE_LOOP,
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
}

# the following opcodes trigger 'return' events.
return_opcodes = {
  RETURN_VALUE,
  YIELD_FROM,
  YIELD_VALUE,
}

# These codes push a block that specifies a jump destination for pop instructions.
push_block_opcodes = {
  SETUP_ASYNC_WITH,
  SETUP_EXCEPT,
  SETUP_FINALLY,
  SETUP_LOOP,
  SETUP_WITH,
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

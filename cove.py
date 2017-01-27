#!/usr/bin/env python3

# Note: any modules imported prior to the calls to install_trace and run_path
# will not report coverage fully, because their <module> code objects will not be captured.
# Therefore, we import only stdlib modules that we need.
import marshal
import sys
import os
from collections import defaultdict
from dis import findlinestarts, get_instructions, hasjabs, hasjrel, opname, opmap
from argparse import ArgumentParser
from inspect import getmodule
from os.path import abspath as abs_path, join as path_join, normpath as normalize_path
from runpy import run_path
from sys import exc_info, settrace, stderr, stdout
from types import CodeType


def main():
  arg_parser = ArgumentParser(description='cove: code coverage harness.')
  arg_parser.add_argument('-targets', nargs='*', default=[])
  arg_parser.add_argument('-dbg', action='store_true')
  excl = arg_parser.add_mutually_exclusive_group()
  excl.add_argument('-coalesce', nargs='+')
  trace_group = excl.add_argument_group('trace')
  trace_group.add_argument('-output')
  trace_group.add_argument('cmd', nargs='*')
  args = arg_parser.parse_args()

  arg_targets = expand_targets(args.targets)
  if args.coalesce:
    coalesce(args.coalesce, arg_targets=arg_targets, dbg=args.dbg)
  else:
    if not args.cmd:
      arg_parser.error('please specify a command.')
    trace_cmd(cmd=args.cmd, arg_targets=arg_targets, output_path=args.output, dbg=args.dbg)


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


def trace_cmd(cmd, arg_targets, output_path, dbg):
  'NOTE: this must be called before importing any module that we might wish to trace with cove.'
  cmd_head = abs_path(cmd[0])
  targets = set(arg_targets or ['__main__'])
  # although run_path alters and restores sys.argv[0],
  # we need to replace all of argv to provide the correct arguments to the command getting traced.
  orig_argv = sys.argv.copy()
  sys.argv = cmd.copy()
  exit_code = 0
  trace_set = install_trace(targets, dbg=dbg)
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
  sys.argv = orig_argv
  target_paths = gen_target_paths(targets, cmd_head, dbg=dbg)
  if output_path:
    write_coverage(output_path=output_path, target_paths=target_paths, trace_set=trace_set)
    exit(exit_code)
  else:
    report(target_paths=target_paths, trace_sets=[trace_set], dbg=dbg)


def scrub_traceback(tbe):
  'Remove frames from TracebackException object that refer to cove, rather than the child process under examination.'
  stack = tbe.stack # StackSummary is a subclass of list.
  if not stack or stack[0].filename.find('cove') == -1: return # not the root exception.
  # TODO: verify that the above find('cove') is sufficiently strict,
  # while also covering both the installed entry_point and the local dev cases.
  del stack[0]
  while stack and stack[0].filename.endswith('runpy.py'): del stack[0] # remove cove runpy.run_path frames.


def install_trace(targets, dbg):
  'NOTE: this must be called before importing any module that we might wish to trace with cove.'

  traces = set()
  file_name_filter = {}

  def is_code_path_targeted(code):
    module = getmodule(code)
    if dbg:
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
    prev_line = -1
    prev_off = -1
    def cove_local_tracer(frame, event, arg):
      nonlocal prev_line, prev_off
      #errSL('LTRACE:', event, prev_line, prev_off, frame.f_lineno, frame.f_lasti, frame.f_code.co_name)
      traces.add((prev_line, prev_off, frame.f_lineno, frame.f_lasti, frame.f_code))
      prev_line = frame.f_lineno
      prev_off = frame.f_lasti
      return cove_local_tracer # local tracer keeps itself in place during its local scope.

    return cove_local_tracer # global tracer installs a new local tracer for every call.

  settrace(cove_global_tracer)
  return traces


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


def write_coverage(output_path, target_paths, trace_set):
  data = {
    'target_paths': target_paths,
    'trace_set': trace_set
  }
  with open(output_path, 'wb') as f:
    marshal.dump(data, f)


def coalesce(trace_paths, arg_targets, dbg):
  target_paths = defaultdict(set)
  for arg_target in arg_targets:
    target_paths[arg_target]
  trace_sets = []
  for trace_path in trace_paths:
    try:
      with open(trace_path, 'rb') as f:
        data = marshal.load(f)
        for target, paths in data['target_paths'].items():
          if arg_targets and target not in arg_targets: continue
          target_paths[target].update(paths)
        trace_sets.append(data['trace_set'])
    except FileNotFoundError:
      exit('cove error: trace file not found: {}'.format(trace_path))
  report(target_paths, trace_sets, dbg=dbg)


class Totals:
  def __init__(self):
    self.lines = 0
    self.traceable = 0
    self.ignored = 0
    self.ignored_covered = 0
    self.uncovered = 0


def report(target_paths, trace_sets, dbg):
  print('cove report:')
  path_traces = gen_path_traces(trace_sets)
  totals = Totals()
  for target, paths in sorted(target_paths.items()):
    if not paths:
      print('\n{}: NEVER IMPORTED.'.format(target))
      continue
    for path in sorted(paths):
      coverage = calculate_coverage(path=path, traces=path_traces[path], dbg=dbg)
      report_path(target=target, path=path, coverage=coverage, totals=totals, dbg=dbg)
  if len(target_paths) > 1:
    print('\nTOTAL: {} lines; {} traceable; {} ignored; {} IGNORED but covered; {} UNCOVERED.'.format(
      totals.lines, totals.traceable, totals.ignored, totals.ignored_covered, totals.uncovered))


def gen_path_traces(trace_sets):
  'Group the traces by path.'
  path_traces = defaultdict(list)
  for trace_set in trace_sets:
    for trace in trace_set:
      code = trace[-1]
      path_traces[code.co_filename].append(trace)
  return path_traces


def calculate_coverage(path, traces, dbg):
  '''
  Calculate and return the coverage data structure,
  Which maps line numbers to (traced:Set[Loc], traceable:Set[Loc]) pairs.
  Each set contains Loc tuples.
  A Loc is (line, offset, code).
  A line is fully covered if (traced == traceable).
  NOTE: if analysis proves to be imperfect, we could use issuperset instead of equality.
  '''
  if dbg:
    errSL('\ntrace: {}:'.format(path))
    traces = sorted_traces(traces)
  traced_codes = (t[-1] for t in traces)
  all_codes = list(visit_nodes(start_nodes=traced_codes, visitor=sub_codes))
  if dbg: all_codes.sort(key=lambda code: code.co_name)
  coverage = {} # (traced, traceable). Do not use defaultdict because reporting logic depends on KeyError.
  # generate all possible traces.
  for code in all_codes:
    crawl_code_insts(path=path, code=code, coverage=coverage, dbg=dbg)
  # then fill in the traced sets.
  for trace in traces:
    if dbg: err_trace('+', trace)
    src_line, src_off, dst_line, dst_off, code = trace
    coverage[dst_line][0].add((dst_line, dst_off, code))
  return coverage


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


def crawl_code_insts(path, code, coverage, dbg):
  if dbg: errSL('\ncrawl code: {}:{}'.format(path, code.co_name))

  insts = list(get_instructions(code))
  if dbg:
    for inst in insts: err_inst(inst)

  visited = set()
  def find_traceable_edges(prev_line, prev_off, off, blocks):
    args = (prev_line, prev_off, off, blocks)
    #errSL('FIND', args)
    if args in visited: return
    visited.add(args)
    index = off // 2 # this is incorrect prior to python3.6.
    inst = insts[index]
    op = inst.opcode
    starts_line = inst.starts_line
    line = starts_line or prev_line # note: this is the line trick described in lnotabs_notes.txt.
    assert inst.offset == off
    assert off or not inst.is_jump_target # this would make boolean tests on jump dsts unsafe.
    if starts_line or (off < prev_off) or (op in traced_opcodes):
      # add loc.
      try: tt = coverage[line]
      except KeyError:
        tt = (set(), set())
        coverage[line] = tt
      loc = (line, off, code)
      #errSL('LOC', loc)
      #assert loc not in tt[1]
      tt[1].add(loc)
      if dbg: err_trace('-', (prev_line, prev_off, line, off, code))
    # block management.
    # this is a guess, probably wrong.
    if op in push_block_opcodes:
      block_dst = None
      blocks = blocks + (inst.argval,)
    elif op in pop_block_opcodes:
      block_dst = blocks[-1]
      blocks = blocks[:-1]
    # note: we currently use recursion to explore the control flow graph.
    # this could hit the recursion limit for large code objects, and is probably slow.
    # alternatively we could:
    # * turn the second call into a fake tail call using a while loop around this whole function body;
    # * use visit_nodes, which would break ordering of dbg output.
    if op == BREAK_LOOP:
      find_traceable_edges(line, off, block_dst, blocks)
      return
    if op in jump_opcodes:
      jmp = inst.argval # argval accounts for absolute vs relative offsets.
      find_traceable_edges(line, off, jmp, blocks)
    if op not in stop_opcodes: # normal interpreter step forward.
      nxt = off + 2
      #^ as of python3.6, all opcodes take 2 bytes.
      # previously the next instruction was off + (1 if op < HAVE_ARGUMENT else 3).
      find_traceable_edges(line, off, nxt, blocks)

  find_traceable_edges(-1, -1, 0, ())


def err_inst(inst):
  op = inst.opcode
  line = inst.starts_line or ''
  off = inst.offset
  dst = ('DST' if inst.is_jump_target else '   ')
  stop = 'stop' if op in stop_opcodes else '    '
  if op in jump_opcodes:
    target = f'jump {inst.argval:4}'
  elif op in push_block_opcodes:
    target = f'push {inst.argval:4}'
  else: target = ''
  arg = 'to {} (abs)'.format(inst.arg) if inst.opcode in hasjabs else inst.argrepr
  errSL(f'  line:{line:>4}  off:{off:>4} {dst:4}  {stop} {target:9}  {inst.opname:{onl}} {arg}')



def report_path(target, path, coverage, totals, dbg):
  rel_path = path_rel_to_current_or_abs(path)

  uncovered_lines = set() # lines that not are perfectly covered.
  for line, (traceable, traced) in coverage.items():
    if traceable != traced:
      uncovered_lines.add(line)

  line_texts = [text.rstrip() for text in open(path).readlines()]
  ignored_lines = calc_ignored_lines(line_texts)
  ignored_covered = len(ignored_lines - uncovered_lines)
  uncovered_count = len(uncovered_lines - ignored_lines)

  totals.lines += len(line_texts)
  totals.traceable += len(coverage)
  totals.ignored += len(ignored_lines)
  totals.ignored_covered += ignored_covered
  totals.uncovered += uncovered_count
  if ignored_covered == 0 and uncovered_count == 0:
    print('\n{}: {}: {} lines; {} traceable; {} ignored.'.format(
      target, rel_path, len(line_texts), len(coverage), len(ignored_lines)))
    return

  print(f'\n{target}: {rel_path}:')
  ctx_lead = 4
  ctx_tail = 1
  intervals = closed_int_intervals(sorted(uncovered_lines ^ ignored_lines))

  def next_interval():
    i = next(intervals)
    return (i[0] - ctx_lead, i[0], i[1], i[1] + ctx_tail)

  lead, start, last, tail_last = next_interval()
  for line, text in enumerate(line_texts, 1):
    if line < lead: continue
    assert line <= tail_last
    sym = ' '
    color = RST
    suffix = None
    try: traced, traceable = coverage[line]
    except KeyError: pass
    else:
      if traced == traceable: # fully covered.
        if line in ignored_lines:
          sym = '?'
          color = TXT_Y
      elif traceable.issuperset(traced): # ignored, partially covered or uncovered.
        if line in ignored_lines:
          sym = '|'
          color = TXT_C
        else:
          sym = '%' if traced else '!'
          color = TXT_R
          suffix = f'{len(traced)} of {len(traceable)} possible edges covered.'
      else: # confused. traceable is missing something.
        sym = '\\'
        color = TXT_M
        suffix = f'{len(traced)} of {len(traceable)} possible edges covered.'
    print(f'{TXT_D}{line:4} {color}{sym} {text}{RST}')
    if suffix and dbg:
      print(f'     {TXT_B}^ {suffix}{RST}')
      err_traces(f'{TXT_D}{line:4} {TXT_B}-', traceable - traced)
      err_traces(f'{TXT_D}{line:4} {TXT_B}=', traceable & traced)
      err_traces(f'{TXT_D}{line:4} {TXT_B}+', traced - traceable)
    if line == tail_last:
      try: lead, start, last, tail_last = next_interval()
      except StopIteration: break
      else:
        if line + 1 < lead: print(f'{TXT_D}   …{RST}')
  print('{}: {}: {} lines; {} traceable; {} ignored; {} IGNORED but covered; {} NOT COVERED.'.format(
    target, rel_path, len(line_texts), len(coverage), len(ignored_lines), ignored_covered, uncovered_count))


def path_rel_to_current_or_abs(path: str) -> str:
  ap = abs_path(path)
  ac = abs_path('.')
  comps = path_comps(ap)
  prefix = path_comps(ac)
  if comps == prefix:
    return '.'
  if prefix == comps[:len(prefix)]:
    return path_join(*comps[len(prefix):])
  errSL(comps)
  errSL(prefix)
  return ap

def path_comps(path: str):
  np = normalize_path(path)
  if np == '/': return ['/']
  assert not np.endswith('/')
  return [comp or '/' for comp in np.split(os.sep)]


def calc_ignored_lines(line_texts):
  from re import compile
  indent_re = compile(r' *')
  ignored = set()
  ignored_indent = 0
  for line, text in enumerate(line_texts, 1):
    m = indent_re.match(text)
    indent = m.end() - m.start()
    if text.endswith('#no-cov!'):
      ignored.add(line)
      ignored_indent = indent
    elif 0 < ignored_indent < indent:
      ignored.add(line)
    else:
      ignored_indent = 0
  return ignored


def closed_int_intervals(iterable):
  '''
  Note: lifted from pithy.iterable. Please send changes upstream.
  Given `iterable` of integers, yield a sequence of closed intervals.
  '''
  it = iter(iterable)
  try: first = next(it)
  except StopIteration: return
  if not isinstance(first, int):
    raise TypeError('closed_int_intervals requires a sequence of int elements; received first element: {!r}', first)
  interval = (first, first)
  for i in it:
    l, h = interval
    if i < h:
      raise ValueError('closed_int_intervals requires monotonically increasing elements')
    if i == h: continue
    if i == h + 1:
      interval = (l, i)
    else:
      yield interval
      interval = (i, i)
  yield interval


def sorted_traces(traces):
  return sorted(traces, key=lambda t: (t[:-1], t[-1].co_name))


def err_trace(label, trace):
  errSL(label, *(f'{el:4}' for el in trace[:-1]), '  ', trace[-1].co_name)


def err_traces(label, traces):
  for t in sorted_traces(traces):
    err_trace(label, t)


def errSL(*items): print(*items, file=stderr)


# Opcode information.

onl = max(len(name) for name in opname)

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

# other opcodes that affect control flow.
BREAK_LOOP            = opmap['BREAK_LOOP']
END_FINALLY           = opmap['END_FINALLY']
POP_BLOCK             = opmap['POP_BLOCK']
POP_EXCEPT            = opmap['POP_EXCEPT']
RETURN_VALUE          = opmap['RETURN_VALUE']
YIELD_FROM            = opmap['YIELD_FROM']
YIELD_VALUE           = opmap['YIELD_VALUE']

# `hasjrel` includes the SETUP_* ops, which do not actually branch.
jump_opcodes = set(hasjabs) | {
  FOR_ITER,
  JUMP_FORWARD
}

# These codes push a block that specifies a jump destination for pop instructions.
push_block_opcodes = {
  SETUP_ASYNC_WITH,
  SETUP_EXCEPT,
  SETUP_FINALLY,
  SETUP_LOOP,
  SETUP_WITH,
}

pop_block_opcodes = {
  BREAK_LOOP,
  END_FINALLY,
  POP_BLOCK,
  POP_EXCEPT,
}

# the following opcodes never advance to the next instruction.
stop_opcodes = {
  BREAK_LOOP,
  CONTINUE_LOOP,
  JUMP_ABSOLUTE,
  JUMP_FORWARD,
  RETURN_VALUE,
  YIELD_FROM, # ??
  YIELD_VALUE, # ??
}

# the following opcodes always trigger tracing due to 'return' trace.
traced_opcodes = {
  RETURN_VALUE,
  YIELD_FROM,
  YIELD_VALUE,
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

#!/usr/bin/env python3

# NOTE: no other top-level imports allowed.
# any modules imported prior to the calls to install_trace and run_path
# will not report coverage fully, because their <module> code objects will not be captured.
# therefore, we import only stdlib modules that we absolutely need.
import sys
from argparse import ArgumentParser
from inspect import getmodule
from os.path import abspath
from runpy import run_path
from sys import stderr, stdout, settrace


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
    trace(cmd=args.cmd, arg_targets=arg_targets, output_path=args.output, dbg=args.dbg)


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


def trace(cmd, arg_targets, output_path, dbg):
  'NOTE: this function must not import or depend on any library that we might wish to trace with cove.'
  cmd_head = abspath(cmd[0])
  targets = set(arg_targets or ['__main__'])
  # although run_path alters sys.argv[0],
  # we need to replace all of argv to provide the correct arguments.
  orig_argv = sys.argv.copy()
  sys.argv = cmd.copy()
  exit_code = 0
  trace_set = install_trace(targets, dbg=dbg)
  #if dbg: print('cove untraceable modules:', sorted(sys.modules.keys()), file=stderr)
  try:
    run_path(cmd_head, run_name='__main__')
  except FileNotFoundError as e:
    exit('cove error: could not find command to run: {!r}'.format(cmd_head))
  except SystemExit as e:
    exit_code = e.code
  finally:
    settrace(None)
  sys.argv = orig_argv
  target_paths = gen_target_paths(targets, cmd_head, dbg=dbg)
  if output_path:
    write_coverage(output_path=output_path, target_paths=target_paths, trace_set=trace_set)
    exit(exit_code)
  else:
    report(target_paths=target_paths, trace_sets=[trace_set], dbg=dbg)


def install_trace(targets, dbg):
  'NOTE: this function must not import or depend on any library that we might wish to trace with cove.'

  traces = set()
  file_name_filter = {}

  def is_code_path_targeted(code):
    module = getmodule(code)
    if dbg:
      print('cove.is_code_path_targeted: {}:{} -> {} -> {}'.format(
        code.co_filename, code.co_name, module and module.__name__,
        (module and module.__name__) in targets), file=stderr)
    if module is None: return False # probably a python builtin; not traceable.
    # note: the module filename may not equal the code filename.
    # example: .../python3.5/collections/abc.py != .../python3.5/_collections_abc.py
    # thus the following check sometimes fires, but it seems acceptable.
    #if dbg and module.__file__ != code.co_filename:
    #  print('  note: module file: {} != code file: {}'.format(module.__file__, code.co_filename), file=stderr)
    is_target = (module.__name__ in targets)
    return is_target

  def cove_tracer(frame, event, arg):
    code = frame.f_code
    path = code.co_filename
    try:
      is_target = file_name_filter[path]
    except KeyError:
      is_target = is_code_path_targeted(code)
      file_name_filter[path] = is_target
    if is_target:
      #print('trace_callback: d:{} e:{} p:{} l:{} i:{}'.format(stack_depth, event, code.co_filename, frame.f_lineno, frame.f_lasti), file=stderr)
      traces.add((frame.f_code, frame.f_lasti, event))
      return cove_tracer
    else:
      return None # do not trace this scope.

  settrace(cove_tracer)
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
      print('gen_target_paths: {} -> {}'.format(t, p), file=stderr)
  return target_paths


def report(target_paths, trace_sets, dbg):
  print('\ncove report:')
  path_code_insts = gen_path_code_insts(trace_sets)
  for target, paths in sorted(target_paths.items()):
    if not paths:
      print('\n{}: NEVER IMPORTED.'.format(target))
      continue
    for path in sorted(paths):
      code_insts = path_code_insts[path]
      report_path(target, path, code_insts, dbg)


def gen_path_code_insts(trace_sets):
  from collections import defaultdict
  path_code_insts = defaultdict(lambda: defaultdict(list)) # path -> code -> inst offsets.
  for trace_set in trace_sets:
    for code, inst, event in trace_set:
      path = code.co_filename
      path_code_insts[path][code].append((inst, event))
  return path_code_insts


def calculate_module_coverage(path, code_insts, dbg):
  from pithy.io import errFL
  traceable_lines = set()
  traced_lines = set()
  code_inst_lines  = visit_codes(path, set(code_insts.keys()), traceable_lines, dbg=dbg)
  for code, values in code_insts.items():
    if dbg: errFL('\ncode: {}:{}', path, code.co_name)
    inst_lines = code_inst_lines[code]
    for inst, event in sorted(values):
      if dbg: errFL('  trace inst: {:3}; event: {}', inst, event)
      if event != 'line': continue
      line = inst_lines[inst]
      traced_lines.add(line)
  return len(traceable_lines), (traceable_lines - traced_lines)


def visit_codes(path, codes, traceable_lines, dbg):
  from dis import get_instructions, hasjabs, hasjrel
  from types import CodeType
  from pithy.io import errFL
  jmp_opcodes = set(hasjabs + hasjrel)
  code_inst_lines = {}
  while codes:
    code = codes.pop()
    if dbg: errFL('\ncode: {}:{}', path, code.co_name)
    assert code not in code_inst_lines
    inst_lines = {}
    code_inst_lines[code] = inst_lines
    line = None
    for inst in get_instructions(code):
      l = inst.starts_line
      if l is not None:
        line = l
        traceable_lines.add(line)
      inst_lines[inst.offset] = line # could do this more sparsely, since most inst offsets are never traced.
      if isinstance(inst.argval, CodeType):
        sub = inst.argval
        if sub not in code_inst_lines: # not yet visited.
          codes.add(sub)
      if dbg:
        jmp = ('DST' if inst.is_jump_target else '')
        if inst.opcode in jmp_opcodes:
          jmp = ('S,D' if jmp else 'SRC')
        errFL('  inst: {:4} line:{:>4} {:3} {:26} {!r}',
          inst.offset,
          ('' if l is None else l),
          jmp,
          inst.opname,
          inst.argval)
  return code_inst_lines


def report_path(target, path, code_insts, dbg):
  from pithy.ansi import TXT_R, TXT_Y, RST
  from pithy.io import  outF, outFL, outL
  from pithy.seq import seq_int_intervals
  from pithy.fs import path_rel_to_current_or_abs

  rel_path = path_rel_to_current_or_abs(path)
  if not code_insts:
    outFL('\n{}: {}: NO COVERAGE.', target, rel_path)
    return

  traceable_count, untraced_lines = calculate_module_coverage(path, code_insts, dbg=dbg)
  if not untraced_lines:
    outFL('\n{}: {}: covered.', target, rel_path)
    return

  lines = [line.rstrip() for line in open(path).readlines()]
  ignored_lines = set(i for i, l in enumerate(lines, 1) if l.endswith('#no-cov!'))
  if untraced_lines == ignored_lines:
    outFL('\n{}: {}: {} lines; {} traceable; {} ignored.',
      target, rel_path, len(lines), traceable_count, len(ignored_lines))
    return

  outFL('\n{}: {}:', target, rel_path)

  ctx_lead = 4
  ctx_tail = 2
  intervals = seq_int_intervals(sorted(untraced_lines ^ ignored_lines))
  def next_interval():
    i = next(intervals)
    return (i[0] - ctx_lead, i[0], i[1], i[1] + ctx_tail)

  lead, start, last, tail_last = next_interval()
  uncovered_count = 0
  ignored_but_covered_count = 0
  for i, line in enumerate(lines, 1):
    if i < lead: continue
    assert i <= tail_last
    if i < start or i > last:
      color = ''
      symbol = ' '
    elif i in untraced_lines:
      color = TXT_R
      symbol = '!'
      uncovered_count += 1
    else:
      color = TXT_Y
      symbol = '?'
      ignored_but_covered_count += 1
    outFL('{:>4d}{}{} {}{}', i, color, symbol, line, RST)
    if i == tail_last:
      try: lead, start, last, tail_last = next_interval()
      except StopIteration: break

  outFL('{}: {}: {} lines; {} traceable; {} IGNORED but covered; {} UNCOVERED.',
    target, rel_path, len(lines), traceable_count, ignored_but_covered_count, uncovered_count)


def write_coverage(output_path, target_paths, trace_set):
  import marshal
  data = {
    'target_paths': target_paths,
    'trace_set': trace_set
  }
  with open(output_path, 'wb') as f:
    marshal.dump(data, f)


def coalesce(trace_paths, arg_targets, dbg):
  import marshal
  from collections import defaultdict
  target_paths = defaultdict(set)
  for arg_target in arg_targets:
    target_paths[arg_target]
  trace_sets = []
  for trace_path in trace_paths:
    with open(trace_path, 'rb') as f:
      data = marshal.load(f)
      for target, paths in data['target_paths'].items():
        if arg_targets and target not in arg_targets: continue
        target_paths[target].update(paths)
      trace_sets.append(data['trace_set'])
  report(target_paths, trace_sets, dbg=dbg)


if __name__ == '__main__': main()

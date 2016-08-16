#!/usr/bin/env python3

import argparse
import marshal
import sys

from collections import defaultdict, namedtuple
from dis import findlinestarts, findlabels, get_instructions
from inspect import getmodule
from runpy import run_module, run_path
from sys import stdout, settrace
from pprint import pprint
from types import CodeType

from pithy.fs import abs_path, path_dir, path_name, path_name_stem
from pithy.io import errFL, errL, errSL, fail, failF, outZ, outF, outFL, outL, outSL
from pithy.seq import seq_int_intervals
from pithy.task import run


def main():
  arg_parser = argparse.ArgumentParser(description='cove: code coverage harness.')
  arg_parser.add_argument('-targets', nargs='*', default=[])
  excl = arg_parser.add_mutually_exclusive_group()
  excl.add_argument('-coalesce', nargs='+')
  trace_group = excl.add_argument_group('trace')
  trace_group.add_argument('-output')
  trace_group.add_argument('cmd', nargs='*')
  args = arg_parser.parse_args()

  if args.coalesce:
    coalesce(args.coalesce, arg_targets=args.targets)
  else:
    if not args.cmd:
      arg_parser.error('please specify a command.')
    trace(cmd=args.cmd, arg_targets=args.targets, output_path=args.output)


def trace(cmd, arg_targets, output_path):
  cmd_head = abs_path(cmd[0])
  targets = set(arg_targets or ['__main__'])
  # although run_module and run_path alter sys.argv[0],
  # we need to replace all of argv to provide the correct arguments.
  orig_argv = sys.argv.copy()
  sys.argv = cmd.copy()
  # TODO: patch and restore sys.modules['__main__'] also?
  exit_code = 0
  trace_set = install_trace(targets)
  try:
    run_path(cmd_head, run_name='__main__')
  except FileNotFoundError as e:
    failF('cove error: could not find command to run: {!r}', cmd_head)
  except SystemExit as e:
    exit_code = e.code
  finally:
    settrace(None)
  sys.argv = orig_argv
  target_paths = gen_target_paths(targets, cmd_head)
  if output_path:
    write_coverage(output_path=output_path, target_paths=target_paths, trace_set=trace_set)
    exit(exit_code)
  else:
    report(target_paths=target_paths, trace_sets=[trace_set])


def install_trace(targets):
  traces = set()
  file_name_filter = {}

  def is_code_path_targeted(code):
    module = getmodule(code)
    if module is None: return False # probably a python builtin; not traceable.
    # note: the module filename may not equal the code filename.
    # example: .../python3.5/collections/abc.py != .../python3.5/_collections_abc.py
    # thus the following check sometimes fires, but it seems acceptable.
    #if module.__file__ != code.co_filename:
    #  errFL('MODULE FILE: {} != CODE FILE: {}', module.__file__, code.co_filename)
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
      #errFL("trace_callback: d:{} e:{} p:{} l:{} i:{}", stack_depth, event, code.co_filename, frame.f_lineno, frame.f_lasti)
      traces.add((frame.f_code, frame.f_lasti, event))
      return cove_tracer

  settrace(cove_tracer)
  return traces


def gen_target_paths(targets, cmd_head):
  '''
  Given a list of target module names and the path that serves as __main__,
  return a dictionary mapping targets to lists of path|None.
  The values are sequences to allow __main__ to map to multiple paths,
  which can occur during coalescing.
  '''
  target_paths = defaultdict(list)
  for target in targets:
    if target.endswith('.*'): # need to import all modules under wildcard in order to find them.
      target_base = path_name_stem(target)
      module = sys.modules.get(target_base)
      if module:
        target_paths[target_base].append(module.__file__)
      import_str = 'from {} import *'.format(target_base)
      try: exec(import_str)
      except (SyntaxError, ImportError): failF('invalid target: {!r}', target)
      prefix = target[:-1]
      for (name, module) in sys.modules.items():
        if name.startswith(prefix):
          target_paths[name].append(module.__file__)
    else: # regular target.
      module = sys.modules.get(target)
      if module:
        target_paths[target].append(module.__file__)
  if '__main__' in target_paths:
     # sys.modules['__main__'] points to cove; we want cmd_head.
    target_paths['__main__'] = [abs_path(cmd_head)]
  return dict(target_paths)


def report(target_paths, trace_sets):
  errL('\ncove report:')
  path_code_insts = gen_path_code_insts(trace_sets)
  for target, paths in sorted(target_paths.items()):
    if not paths:
      outFL('{}: never imported.', target)
      continue
    for path in sorted(paths):
      code_insts = path_code_insts[path]
      report_path(path, code_insts)


def gen_path_code_insts(trace_sets):
  path_code_insts = defaultdict(lambda: defaultdict(list)) # path -> code -> inst offsets.
  for trace_set in trace_sets:
    for code, inst, event in trace_set:
      path = code.co_filename
      path_code_insts[path][code].append((inst, event))
  return path_code_insts


def calculate_module_coverage(path, code_insts):
  dbg = False
  traceable_lines = set()
  traced_lines = set()
  code_inst_lines  = visit_codes(path, set(code_insts.keys()), traceable_lines, dbg=dbg)
  for code, values in code_insts.items():
    if dbg: errFL('code: {}: {}', path, code.co_name)
    inst_lines = code_inst_lines[code]
    for inst, event in sorted(values):
      if dbg: errFL('  trace inst: {:3}; event: {}', inst, event)
      try: line = inst_lines[inst]
      except KeyError: assert inst == -1 and event == 'call'
      else: traced_lines.add(line)
  return len(traceable_lines), sorted(traceable_lines - traced_lines)


def visit_codes(path, codes, traceable_lines, dbg):
  code_inst_lines = {}
  while codes:
    code = codes.pop()
    if dbg: errFL('code: {}: {}', path, code.co_name)
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
        errFL('  inst: {:3} line:{:>4} {:3} {:14} {}',
          inst.offset,
          ('' if l is None else l),
          ('DST' if inst.is_jump_target else ''),
          inst.opname,
          inst.argval)
  return code_inst_lines


def report_path(path, code_insts):
  outZ(path, ':')
  if not code_insts:
    outL(' NO COVERAGE.')
    return
  traceable_count, untraced_lines = calculate_module_coverage(path, code_insts)
  ctx_lead = 4
  ctx_tail = 2
  intervals = seq_int_intervals(untraced_lines)

  def next_interval():
    i = next(intervals)
    return (i[0] - ctx_lead, i[0], i[1], i[1] + ctx_tail)

  try: lead, start, last, tail_last = next_interval()
  except StopIteration:
    outL(' covered.')
    return
  outL()

  with open(path) as f:
    line_count = 0
    line = '\n' # satisfies the missing final newline check if file is empty.
    for i, line in enumerate(f, 1):
      if i < lead: continue
      assert i <= tail_last
      unexecuted = (start <= i <= last)
      outF('{:>4d}{} {}', i, ('!' if unexecuted else ' '), line)
      if i == tail_last:
        try: lead, start, last, tail_last = next_interval()
        except StopIteration:
          lead = 1<<63 # continue iterating to get line_count.
        else:
          if i < lead:
            outL()
    line_count = i
    if not line.endswith('\n'): outL() # handle missing final newline.

  outFL('{}: {} lines; {} traceable: {} untraced.\n',
    path, line_count, traceable_count, len(untraced_lines))


def write_coverage(output_path, target_paths, trace_set):
  data = {
    'target_paths': target_paths,
    'trace_set': trace_set
  }
  with open(output_path, 'wb') as f:
    marshal.dump(data, f)


def coalesce(trace_paths, arg_targets):
  target_paths = defaultdict(set)
  trace_sets = []
  for trace_path in trace_paths:
    with open(trace_path, 'rb') as f:
      data = marshal.load(f)
      for target, paths in data['target_paths'].items():
        if arg_targets and target not in arg_targets: continue
        target_paths[target].update(paths)
      trace_sets.append(data['trace_set'])
  report(target_paths, trace_sets)


if __name__ == '__main__': main()

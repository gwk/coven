#!/usr/bin/env python3

# Note: any modules imported prior to the calls to install_trace and run_path
# will not report coverage fully, because their <module> code objects will not be captured.
# Therefore, we import only stdlib modules that we need.
import marshal
import sys
from collections import defaultdict
from dis import findlinestarts
from argparse import ArgumentParser
from inspect import getmodule
from os.path import abspath
from runpy import run_path
from sys import exc_info, stderr, stdout, settrace
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
  cmd_head = abspath(cmd[0])
  targets = set(arg_targets or ['__main__'])
  # although run_path alters and restores sys.argv[0],
  # we need to replace all of argv to provide the correct arguments to the command getting traced.
  orig_argv = sys.argv.copy()
  sys.argv = cmd.copy()
  exit_code = 0
  trace_set = install_trace(targets, dbg=dbg)
  #if dbg: print('cove untraceable modules (imported prior to `install_trace`):', sorted(sys.modules.keys()), file=stderr)
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

  def cove_global_tracer(global_frame, global_event, _global_arg_is_none):
    #print('GTRACE:', global_event, global_frame.f_code.co_name)
    if global_event != 'call': return None
    code = global_frame.f_code
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
    def cove_local_tracer(frame, event, arg):
      #print('LTRACE:', event, frame.f_lineno, frame.f_code.co_name, file=stderr)
      traces.add((frame.f_lineno, frame.f_code))
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
      print('gen_target_paths: {} -> {}'.format(t, p), file=stderr)
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
    with open(trace_path, 'rb') as f:
      data = marshal.load(f)
      for target, paths in data['target_paths'].items():
        if arg_targets and target not in arg_targets: continue
        target_paths[target].update(paths)
      trace_sets.append(data['trace_set'])
  report(target_paths, trace_sets, dbg=dbg)


def report(target_paths, trace_sets, dbg):
  print('\ncove report:')
  path_traces = gen_path_traces(trace_sets)
  for target, paths in sorted(target_paths.items()):
    if not paths:
      print('\n{}: NEVER IMPORTED.'.format(target))
      continue
    for path in sorted(paths):
      coverage = calculate_coverage(path=path, traces=path_traces[path], dbg=dbg)
      report_path(target=target, path=path, coverage=coverage, dbg=dbg)


def gen_path_traces(trace_sets):
  'Group the traces by path: path -> trace pair (line, code).'
  path_traces = defaultdict(list)
  for trace_set in trace_sets:
    for trace in trace_set:
      code = trace[1]
      path_traces[code.co_filename].append(trace)
  return path_traces


def calculate_coverage(path, traces, dbg):
  '''
  Calculate and return the coverage data structure,
  Which maps line numbers to pairs of (traced, traceable) sets.
  Each set contains traces, which are pairs of (line, code).
  A line is fully covered if (traced == traceable).
  '''
  if dbg:
    print('\ntrace: {}:'.format(path), file=stderr)
    traces = sorted_traces(traces)
  traced_codes = (t[1] for t in traces)
  all_codes = visit_nodes(start_nodes=traced_codes, visitor=sub_codes)
  coverage = {} # do not use defaultdict here because reporting logic depends on KeyError.
  # generate all possible traces.
  for code in all_codes:
    for _offset, line in findlinestarts(code):
      try: tt = coverage[line] # traced_traceable pair.
      except KeyError:
        tt = (set(), set())
        coverage[line] = tt
      trace = (line, code)
      tt[1].add(trace)
      if dbg: err_trace('-', trace)
  # then fill in the traced sets.
  for trace in traces:
    if dbg: err_trace('+', trace)
    line = trace[0]
    coverage[line][0].add(trace)
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


def report_path(target, path, coverage, dbg):
  from pithy.ansi import TXT_C, TXT_D, TXT_L, TXT_M, TXT_R, TXT_Y, RST
  from pithy.io import errFL, outFL
  from pithy.iterable import closed_int_intervals
  from pithy.fs import path_rel_to_current_or_abs

  rel_path = path_rel_to_current_or_abs(path)

  no_cov = True
  uncovered_lines = set() # any line that is not perfectly covered.
  for line, (traceable, traced) in coverage.items():
    no_cov &= not traced
    if traceable != traced:
      uncovered_lines.add(line)

  if not uncovered_lines:
    outFL('\n{}: {}: {} lines covered.', target, rel_path, len(coverage))
    return

  if no_cov:
    outFL('\n{}: {}: {} LINES NOT COVERED.', target, rel_path, len(uncovered_lines))
    return

  line_texts = [text.rstrip() for text in open(path).readlines()]
  ignored_lines = set(line for line, text in enumerate(line_texts, 1) if text.endswith('#no-cov!'))
  if uncovered_lines == ignored_lines:
    outFL('\n{}: {}: {} lines; {} traceable; {} ignored.',
      target, rel_path, len(line_texts), len(coverage), len(ignored_lines))
    return

  outFL('\n{}: {}:', target, rel_path)
  ctx_lead = 4
  ctx_tail = 2
  intervals = closed_int_intervals(sorted(uncovered_lines ^ ignored_lines))
  def next_interval():
    i = next(intervals)
    return (i[0] - ctx_lead, i[0], i[1], i[1] + ctx_tail)

  lead, start, last, tail_last = next_interval()
  uncovered_count = 0
  ignored_but_covered_count = 0
  for line, text in enumerate(line_texts, 1):
    if line < lead: continue
    assert line <= tail_last
    sym = ' '
    color = RST
    try:
      traced, traceable = coverage[line]
      if traced == traceable: # fully covered.
        if line in ignored_lines:
          ignored_but_covered_count += 1
          sym = '?'
          color = TXT_Y
      elif traceable.issuperset(traced): # ignored, partially covered or uncovered.
        if line in ignored_lines:
          sym = '|'
          color = TXT_C
        else:
          uncovered_count += 1
          sym = '%' if traced else '!'
          color = TXT_R
      else: # confused. traceable is missing something.
        sym = '\\'
        color = TXT_M
        err_traces('{:4}: ERROR: traced:   '.format(line), traced)
        err_traces('{:4}: ERROR: traceable:'.format(line), traceable)
    except KeyError:
      cov = '   '
      sym = ' '
    outFL('{dark}{line:4} {color}{sym} {text}{rst}',
      dark=TXT_D, line=line, color=color, sym=sym, text=text, rst=RST)

    if line == tail_last:
      try: lead, start, last, tail_last = next_interval()
      except StopIteration: break

  outFL('{}: {}: {} lines; {} traceable; {} ignored; {} IGNORED but covered; {} NOT COVERED.',
    target, rel_path, len(line_texts), len(coverage), len(ignored_lines), ignored_but_covered_count, uncovered_count)


def sorted_traces(traces):
  return sorted(traces, key=lambda t: (t[0], t[1].co_name))

def err_trace(label, trace):
  print('{} {:4}: {}'.format(label, trace[0], trace[1].co_name), file=stderr)

def err_traces(label, traces):
  for t in sorted_traces(traces):
    err_trace(label, t)


if __name__ == '__main__': main()

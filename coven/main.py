# Dedicated to the public domain under CC0: https://creativecommons.org/publicdomain/zero/1.0/.

# Coven is a code coverage tool that analyzes CPython bytecode to opcode-accurate coverage reports.
# It depends on FrameType.f_trace_opcodes to generate opcode-level trace data,
# and CodeType.co_lines (PEP 626, Python 3.10) to get accurate line information.
# Note: any modules imported prior to the calls to install_trace and run_path
# will not report coverage fully, because their <module> code objects will not be captured.
# Therefore, we only use stdlib modules.

import sys
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from marshal import dump as write_marshal, load as load_marshal
from sys import stdout
from types import CodeType

from .analysis import calculate_coverage
from .disassemble import Src
from .report import Stats, report_path
from .trace import TraceEdge, trace_cmd


assert sys.version_info >= (3, 10, 0)


def main() -> None:

  parser = ArgumentParser(description='coven: code coverage harness.')
  parser.add_argument('-targets', nargs='*', default=[])
  parser.add_argument('-block', help='Name of code block to display and debug.')
  parser.add_argument('-list-blocks', action='store_true', help='List all code block names.')
  parser.add_argument('-show-all', action='store_true')
  parser.add_argument('-color-on', dest='color', action='store_true', default=stdout.isatty())
  parser.add_argument('-color-off', dest='color', action='store_false')

  #excl = parser.add_mutually_exclusive_group() # `coalesce` and `trace` are mutually exclusive.
  parser.add_argument('-coalesce', nargs='+')

  trace_group = parser.add_argument_group('trace')
  trace_group.add_argument('-output')
  trace_group.add_argument('cmd', nargs='*')

  args = parser.parse_args()
  arg_targets = expand_targets(args.targets)

  if args.coalesce:
    coalesce_and_report(trace_paths=args.coalesce, arg_targets=arg_targets, args=args)
    return

  if not args.cmd: parser.error('please specify a command.')

  trace = trace_cmd(cmd=args.cmd, arg_targets=arg_targets, args=args)
  if args.output:
    write_trace_data(output_path=args.output, target_paths=trace.target_paths, path_code_edges=trace.path_code_edges)
  else:
    target_path_lists: dict[str,list[str]] = { t : [p] for t, p in trace.target_paths.items() if p is not None}
    report(target_path_lists=target_path_lists, path_code_edges=trace.path_code_edges, args=args)
  exit(trace.exit_code)


def expand_targets(arg_targets: list[str]) -> set[str]:
  'Convert the list of targets to a set of dot-separated module paths.'
  targets = set()
  for arg in arg_targets:
    targets.add(expand_module_name_or_path(arg))
  return targets


def expand_module_name_or_path(word: str) -> str:
  'if `word` appears to be a file path, convert it to a dot-separated module path.'
  if word.endswith('.py') or '/' in word:
    return expand_module_path(word)
  else:
    return word


def expand_module_path(path: str) -> str:
  'Convert a file path to a dot-separated module path.'
  slash_pos = path.find('/')
  if slash_pos == -1: slash_pos = 0
  dot_pos = path.find('.', slash_pos)
  if dot_pos == -1: dot_pos = len(path)
  stem = path[:dot_pos]
  return stem.replace('/', '.')


def write_trace_data(output_path:str, target_paths:dict[str, str|None], path_code_edges:dict[str,dict[CodeType,set[TraceEdge]]]) -> None:
  'Write the raw trace data to `output_path`, so that it can be coalesced with other runs later.'
  data = {
    'target_paths': target_paths,
    'path_code_edges': path_code_edges,
  }
  with open(output_path, 'wb') as f:
    write_marshal(data, f)


def coalesce_and_report(trace_paths: list[str], arg_targets: set[str], args: Namespace) -> None:
  'Coalesce multiple saved coverage file and report.'
  target_path_sets: defaultdict[str,set[str]] = defaultdict(set)
  for t in arg_targets:
    target_path_sets[t] = set()

  path_code_edges_dd: defaultdict[str, defaultdict[CodeType, set[TraceEdge]]] = defaultdict(lambda: defaultdict(set))
  for trace_path in trace_paths:
    try: f = open(trace_path, 'rb')
    except FileNotFoundError:
      exit(f'coven error: trace file not found: {trace_path}')
    with f: data = load_marshal(f)
    for target, path in data['target_paths'].items():
      if arg_targets and target not in arg_targets: continue
      s = target_path_sets[target] # materialize the set; leave empty for None case.
      if path is not None: s.add(path)
    target_path_lists: dict[str, list[str]] = { t : sorted(paths) for t, paths in target_path_sets.items() }
    for path, code_edges in data['path_code_edges'].items():
      for code, edges in code_edges.items():
        path_code_edges_dd[path][code].update(edges)

  path_code_edges = { p : dict(ce) for p, ce in path_code_edges_dd.items() }
  report(target_path_lists=target_path_lists, path_code_edges=path_code_edges, args=args)


def report(target_path_lists: dict[str,list[str]], path_code_edges: dict[str,dict[CodeType,set[TraceEdge]]], args: Namespace) -> None:
  print('----------------')
  print('Coverage Report:')
  totals = Stats()
  for target, paths in sorted(target_path_lists.items()):
    if not paths:
      print(f'\n{target}: NEVER IMPORTED.')
      continue
    for path in paths:
      code_src = Src.from_path(path=path)
      coverage = calculate_coverage(code_src=code_src, code_traced_edges=path_code_edges[path], dbg_name=args.block)
      report_path(target=target, path=path, coverage=coverage, totals=totals, args=args)
  if sum(len(paths) for paths in target_path_lists.values()) > 1:
    totals.describe('\nTOTAL', args.color)


if __name__ == '__main__': main()

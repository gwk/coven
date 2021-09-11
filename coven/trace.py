# Dedicated to the public domain under CC0: https://creativecommons.org/publicdomain/zero/1.0/.

import sys

from os.path import dirname, abspath as abs_path
from collections import defaultdict
from sys import settrace, stdout, stderr, exc_info
from traceback import TracebackException
from typing import Callable, Any, Optional
from types import FrameType, CodeType
from inspect import getmodule
from argparse import Namespace
from runpy import run_path
from dataclasses import dataclass

from .util import errSL
from .disassemble import OFF_BEGIN


TraceEdge = tuple[int,int] # (prev_off, off)
#^ The raw trace data that we collect during execution.

@dataclass
class Trace:
  target_paths: dict[str, str|None]
  path_code_edges: dict[str,dict[CodeType,set[TraceEdge]]]
  exit_code: int


TraceFn = Callable[[FrameType,str,Any], Optional[Callable]]
#^ The type of `sys.settrace`. The last parameter type is dependent on the second `event` string.
#^ This is the best type signature we can write at this time;
#^ return value should be 'TraceFn' but mypy cannot handle the cyclic definition.


def trace_cmd(cmd: list[str], arg_targets: set[str], args: Namespace) -> Trace:
  '''
  Invoke and trace `cmd`, which is an argv-like list of strings.
  `cmd` is invoked using the `runpy` module, after manipulating sys.argv and sys.path.

  NOTE: this must be called before importing any module that we might wish to trace with coven.
  '''
  cmd_path = cmd[0]
  targets = set(arg_targets) or {'__main__'}
  # Although run_path alters and restores sys.argv[0],
  # we need to replace all of argv to provide the correct arguments to the command getting traced.
  orig_argv = sys.argv.copy()
  sys.argv = cmd.copy()
  # Also need to fix the search path to imitate the regular interpreter.
  orig_path = sys.path
  sys.path = orig_path.copy()
  sys.path[0] = dirname(cmd[0]) # Not sure if this is right in all cases.
  exit_code = 0
  code_edges = install_trace(targets, dbg_name=args.block)
  #if args.block: errSL('coven untraceable modules (imported prior to `install_trace`):', sorted(sys.modules.keys()))
  try:
    run_path(cmd_path, run_name='__main__')
    #^ Use cmd_path as is (instead of the absolute path), so that it appears as it would naturally in a stack trace.
    #^ NOTE: this changes the appearance of stack traces; see fixup_traceback below.
    #^ It might also cause other subtle behavioral changes.
    # TODO: should compile code first, outside of try block, to distinguish missing code from a FileNotFoundError in the script.
  except FileNotFoundError as e:
    exit(f'coven error: could not find command to run: {cmd_path!r}')
  except SystemExit as e:
    exit_code = e.code
  except BaseException:
    from traceback import TracebackException
    exit_code = 1 # exit code that Python returns when an exception raises to toplevel.
    # Format the traceback exactly as it would appear when run without coven.
    traceback = TracebackException(*exc_info())
    fixup_traceback(traceback)
    print(*traceback.format(), sep='', end='', file=stderr)
  finally:
    settrace(None)
    stdout.flush()
    stderr.flush()
  sys.argv = orig_argv

  # Generate the target paths dictionary.
  # Path values may be None, indicating that the target was never imported / has no coverage.
  # Note: __main__ is handled specially:
  # sys.modules['__main__'] points to coven, while we want the absolute guest command path.
  target_paths: dict[str, str|None] = {}
  for target in sorted(targets):
    path: str|None
    if target == '__main__':
      path = abs_path(cmd_path)
    else:
      try: path = sys.modules[target].__file__
      except KeyError: path = None
    target_paths[target] = path
    if args.block: errSL(f'target_paths: {target} -> {path}')

  # Group code by path; this is necessary for per-file display,
  # and also lets us store code belonging to __main__ by absolute path,
  # which disambiguates multiple different mains for coalesced test scripts.
  # Without the call to `abs_path`, co_filename might be relative in the __main__ case.
  path_code_edges_dd:defaultdict[str,dict[CodeType,set[TraceEdge]]] = defaultdict(dict)
  for code, edges in code_edges.items():
    path_code_edges_dd[abs_path(code.co_filename)][code] = edges
  path_code_edges = dict(path_code_edges_dd) # convert to plain dict for marshal and safety.

  return Trace(target_paths=target_paths, path_code_edges=path_code_edges, exit_code=exit_code)


def install_trace(targets: set[str], dbg_name: bool) -> defaultdict[CodeType,set[TraceEdge]]:
  '''
  Set up the tracing mechanism in the current python interpreter.
  NOTE: this must be called before importing any module that we might wish to trace with coven.
  '''
  if dbg_name: errSL("coven targets:", targets)

  code_edges: defaultdict[CodeType,set[TraceEdge]] = defaultdict(set)
  file_name_filter: dict[str,bool] = {}

  def is_code_targeted(code: CodeType) -> bool:
    'Determine whether a code object belongs to the target set.'
    module = getmodule(code)
    if module is None: return False # probably a python builtin; not traceable.
    is_target = (module.__name__ in targets)
    # Note: the module filename may not equal the code filename.
    # Example: .../python3.5/collections/abc.py != .../python3.5/_collections_abc.py
    # Thus the following check sometimes fires, but it seems acceptable.
    #if dbg_name and module.__file__ != code.co_filename:
    #  errSL(f'  note: module file: {module.__file__} != code file: {code.co_filename}')
    if dbg_name:
      stderr.flush()
      errSL(f'coven.is_code_targeted: {code.co_filename}:{code.co_name} -> {module.__name__} : {is_target}')
    return is_target

  def coven_global_tracer(g_frame: FrameType, g_event: str, _g_arg_is_none: None) -> TraceFn|None:
    'The global trace function installed with `sys.settrace.`'
    code = g_frame.f_code
    #if dbg_name == code.co_name: errSL(f'GTRACE: {code.co_name}:{g_frame.f_lineno or 0} {g_event}')
    if g_event != 'call': return None
    path = code.co_filename
    try:
      is_target = file_name_filter[path]
    except KeyError:
      is_target = is_code_targeted(code)
      file_name_filter[path] = is_target

    if not is_target: return None # do not trace this scope.

    # set tracing mode.
    g_frame.f_trace_lines = False
    g_frame.f_trace_opcodes = True

    # The local tracer lives only as long as execution continues within the code block.
    # For a generator, this can be less than the lifetime of the frame,
    # which is saved and restored when resuming from a `yield`.
    edges = code_edges[code]
    prev_off  = OFF_BEGIN
    def coven_local_tracer(frame: FrameType, event: str, arg: Any) -> TraceFn|None:
      nonlocal prev_off
      off = frame.f_lasti
      #errSL(f'LTRACE: {code.co_name}:{frame.f_lineno} {event[:6]} {prev_off:2} -> {off:2}  arg: {arg}')
      if event == 'opcode':
        edges.add((prev_off, off))
        prev_off = off
      return coven_local_tracer # Local tracer keeps itself in place during its local scope.

    return coven_local_tracer # Global tracer installs a new local tracer for every call.

  settrace(coven_global_tracer)
  return code_edges


def fixup_traceback(traceback: TracebackException) -> None:
  'Remove frames from TracebackException object that refer to coven, rather than the child process under examination.'
  stack = traceback.stack # StackSummary is a subclass of list.
  if not stack or 'coven' not in stack[0].filename: return # not the root exception.
  #^ TODO: verify that the above is sufficiently strict,
  #^ while also covering both the installed entry_point and the local dev cases.
  del stack[0]
  while stack and stack[0].filename.endswith('runpy.py'): del stack[0] # remove coven runpy.run_path frames.

# Dedicated to the public domain under CC0: https://creativecommons.org/publicdomain/zero/1.0/.

from collections import defaultdict
from dataclasses import dataclass
from types import CodeType
from typing import Any, Iterable, NamedTuple, Union

from .disassemble import OFF_RAISED, Code, Disassembly, InstOptReason, Src, disassemble
from .trace import TraceEdge
from .util import errL, errLL, errSL, visit_nodes


CovEdge = tuple[int,int,CodeType] # (src_off:int, dst_off:int, code:CodeType).


class ReqMatchedSetPairs(NamedTuple):
  req: set[CovEdge]
  opt: set[CovEdge]

@dataclass
class Coverage:
  '''
  `line_sets` maps line numbers to (required, matched) pairs of sets of CovEdge (src_off:int, dst_off:int, code:CodeType).
  Each set contains Edge tuples.
  An Edge is (prev_offset, offset, code).
  A line is fully covered if (required <= traced).
  '''
  all_codes: list[Code]
  line_sets: dict[int,ReqMatchedSetPairs]


def sub_codes_raw(code: CodeType) -> list[CodeType]:
  'Return a list of all the code objects referenced by this one.'
  return [c for c in code.co_consts if isinstance(c, CodeType)]


def calculate_coverage(code_src: Src, code_traced_edges: dict[CodeType,set[TraceEdge]], dbg_name: str) -> Coverage:
  '''
  Calculate and return the coverage data structure.
  '''
  if dbg_name: errSL(f'\ncalculate_coverage: {code_src.path}:')

  all_raw_codes: list[CodeType] = list(visit_nodes(start_nodes=code_traced_edges, visitor=sub_codes_raw)) # Collect all of the reachable code objects.
  all_codes = [Code(src=code_src, raw=r) for r in all_raw_codes]

  if dbg_name: all_codes.sort() # In debug mode, sort for reproduceability.

  line_sets_dd: defaultdict[int,ReqMatchedSetPairs] = defaultdict(lambda: ReqMatchedSetPairs(set(), set()))
  #^ Keyed by line number.

  for code in all_codes:
    dbg = (code.raw.co_name == dbg_name)
    disassembly: Disassembly = disassemble(code, dbg=dbg)

    if dbg: errLL(*disassembly.render(show_src=True)) # TODO: support -no-src option?

    try: traced:set[TraceEdge] = code_traced_edges[code.raw]
    except KeyError: traced = set()
    if dbg: err_edges('traced:', traced, suffix=code.raw.co_name)

    req, opt = crawl_code_for_edges(disassembly=disassembly, dbg=dbg)

    # match traced to inferred edges.
    # "Exception edges" are those with src == OFF_RAISED.
    # They represent the start of an exception handling trace as inferred by `Disassembly` and `crawl_code_for_edges`.
    # The tracer does not generate these edges exactly,
    # as the src offset will be the offset of the real instruction that raised the exception.
    # We have to convert them to the simplified exception edges, or else they would appear to be unexpected,
    # and exception flows would appear uncovered.

    raise_reqs = { edge[1] for edge in req if edge[0] == OFF_RAISED }
    raise_opts = { edge[1] for edge in opt if edge[0] == OFF_RAISED }

    off_lines = disassembly.off_lines
    matched = set[TraceEdge]() # Expected exception edges (src == OFF_RAISED) that replace an actual traced edge.
    for edge in traced:
      assert len(edge) == 2 # TraceEdge.
      dst_off = edge[1]
      line = off_lines[dst_off]
      if edge in req:
        matched.add(edge)
      elif dst_off in raise_reqs:
        #err_edge('RAISE EDGE:', edge[:2], code.name)
        raise_edge = (OFF_RAISED, dst_off)
        matched.add(raise_edge)
      elif not (edge in opt or dst_off in raise_opts):
        err_edge('UNEXPECTED:', edge, code.name)
        errSL(*raise_reqs)

    # assemble final coverage data by line.

    for src, dst in req:
      line = off_lines[dst]
      assert line >= 0, (src, dst, line)
      line_sets_dd[line][COV_REQ].add((src, dst, code.raw))

    for src, dst in matched:
      line = off_lines[dst]
      assert line >= 0, (src, dst, line)
      line_sets_dd[line][COV_MATCHED].add((src, dst, code.raw))

  line_sets = dict(line_sets_dd)
  return Coverage(all_codes=all_codes, line_sets=line_sets)


COV_REQ, COV_MATCHED = range(2)

Edge = tuple[int,int]

def crawl_code_for_edges(disassembly:Disassembly, dbg:bool) -> tuple[set[Edge],set[Edge]]:
  '''
  Given a code block, construct a pair of (req, opt) mappings.
  Each maps ? to ?.
  '''
  if dbg: errSL(f'\ncrawl_code_for_edges: {disassembly.code.name}')

  insts = disassembly.insts
  inst_srcs = disassembly.inst_srcs
  inst_opt_reasons = disassembly.inst_opt_reasons

  req = set[Edge]() # Required edges.
  opt = set[Edge]() # Optional edges.

  for inst in insts.values():
    if inst.off < 0: continue # Skip the raised_inst and begin_inst pseudo-instructions.
    is_req = (inst_opt_reasons[inst] == InstOptReason.REQ)
    for src in inst_srcs[inst]:
      (req if is_req else opt).add((src.off, inst.off))

  return req, opt


AnyEdge = Union[TraceEdge,CovEdge]

def any_edge_to_pair(edge: AnyEdge) -> tuple[int,int]:
  return edge[:2]


def err_edge(label:str, edge:AnyEdge, name:str) -> None:
  src, dst = any_edge_to_pair(edge)
  errL(f'{label} edge: {src:4} -> {dst:4}  {name}')


def err_edges(label: str, edges: Iterable[AnyEdge], suffix:str='') -> None:
  if suffix: suffix = '  ' + suffix
  sorted_edges = sorted(any_edge_to_pair(e) for e in edges)
  #errSL("ERR_EDGES", len(sorted_edges))
  jump_offs = set()
  for edge in sorted_edges:
    src = edge[0]
    dst = edge[1]
    if src + 2 < dst: # Jump or extended instruction.
      jump_offs.add(src)
      jump_offs.add(dst)

  start: tuple[int,int]|None = None
  prev: tuple[int,int] = (0, 0) # Dummy value.

  def flush() -> None:
    nonlocal start, prev
    assert start
    arrow = '->' if start == prev else '=>'
    errL(f'{label} {start[0]:4} {arrow} {prev[1]:4}  {suffix}')
    start = None
    prev = (0, 0) # Dummy value.

  for edge in sorted_edges:
    if edge[0] + 2 < edge[1]: # Jump or extended instruction (not "straight").
      if start: flush()
      errL(f'{label} {edge[0]:4} -> {edge[1]:4}  {suffix}')
      continue
    if start is None: # Edge is first "straight" (not jump or extended) edge of the run.
      start = prev = edge
    elif prev[1] == edge[0] and edge[0] not in jump_offs: # Subsequent straight edge; continue the run.
      #errL(f'  CONTINUE: {start} {prev} {edge}')
      prev = edge
    else: # The run is broken.
      flush()
      start = prev = edge
  if start:
    #errL(f'  FINAL: {start} {prev} {edge}')
    flush()

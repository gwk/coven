# Dedicated to the public domain under CC0: https://creativecommons.org/publicdomain/zero/1.0/.

from collections import defaultdict
from dataclasses import dataclass
from types import CodeType
from typing import Any, Iterable, NamedTuple

from .disassemble import OFF_RAISED, Code, Disassembly, InstOptReason, Src, disassemble
from .trace import TraceEdge
from .util import errL, errLL, errSL, visit_nodes


CovTriple = tuple[int,int,CodeType] # (src_off:int, dst_off:int, code:CodeType).


class ReqMatchedSetPairs(NamedTuple):
  req: set[CovTriple]
  opt: set[CovTriple]

@dataclass
class Coverage:
  '''
  `line_sets` maps line numbers to (required, matched) pairs of sets of CovTriple (src_off:int, dst_off:int, code:CodeType).
  Each set contains Edge tuples.
  An Edge is (prev_offset, offset, code).
  A line is fully covered if (required <= traced).
  '''
  all_codes: list[Code]
  line_sets: dict[int,ReqMatchedSetPairs]


def sub_codes_raw(code: CodeType) -> list[CodeType]:
  'Return a list of all the code objects referenced by this one.'
  return [c for c in code.co_consts if isinstance(c, CodeType)]


def calculate_coverage(path: str, code_traced_edges: dict[CodeType,set[TraceEdge]], dbg_name: str) -> Coverage:
  '''
  Calculate and return the coverage data structure.
  '''
  if dbg_name: errSL(f'\ncalculate_coverage: {path}:')

  all_raw_codes: list[CodeType] = list(visit_nodes(start_nodes=code_traced_edges, visitor=sub_codes_raw)) # Collect all of the reachable code objects.
  src = Src.from_path(path=path)
  all_codes = [Code(src=src, raw=r) for r in all_raw_codes]
  if dbg_name: all_codes.sort() # In debug mode, sort them for reproduceability.

  line_sets_dd: defaultdict[int,ReqMatchedSetPairs] = defaultdict(lambda: ReqMatchedSetPairs(set(), set()))
  #^ Keyed by line number.

  for code in all_codes:
    dbg = (code.raw.co_name == dbg_name)
    disassembly: Disassembly = disassemble(code, dbg=dbg)
    if dbg:
      errLL(*disassembly.render(show_src=True)) # TODO: support -no-src option?

    try: traced:set[TraceEdge] = code_traced_edges[code.raw]
    except KeyError: traced = set()

    req, opt = crawl_code_for_edges(disassembly=disassembly, dbg=dbg)
    if dbg:
      err_edges('traced:', traced, suffix=code.raw.co_name)
    # match traced to inferred edges.
    raise_reqs = { edge[1] : (edge, lines) for edge, lines in req.items() if edge[0] == OFF_RAISED }
    raise_opts = { edge[1] for edge in opt if edge[0] == OFF_RAISED }
    matched = defaultdict(set) # expected exception edges that matched an actual traced edge.
    for trace_edge in traced:
      edge = trace_edge[:2]
      dst_off = trace_edge[1]
      line = trace_edge[2]
      if edge in req:
        matched[edge].add(line)
      elif dst_off in raise_reqs:
        e, l = raise_reqs[dst_off]
        matched[e].update(l) # add all the lines (really just one line?) implied by the exception edge.
      elif not (edge in opt or dst_off in raise_opts):
        err_edge('UNEXPECTED:', edge, code.name)
        errSL(*raise_reqs)

    # assemble final coverage data by line.

    for edge, lines in req.items():
      for line in lines:
        assert line >= 0, (edge, line)
        line_sets_dd[line][COV_REQ].add((edge[0], edge[1], code.raw))

    for edge, lines in matched.items():
      for line in lines:
        assert line >= 0, (edge, line)
        line_sets_dd[line][COV_MATCHED].add((edge[0], edge[1], code.raw))

  line_sets = dict(line_sets_dd)
  return Coverage(all_codes=all_codes, line_sets=line_sets)


COV_REQ, COV_MATCHED = range(2)

EdgesToLines = defaultdict[tuple[int,int], set[int]]

def crawl_code_for_edges(disassembly:Disassembly, dbg:bool) -> tuple[EdgesToLines, EdgesToLines]:
  '''
  Given a code block, construct a pair of (req, opt) mappings.
  Each maps ? to ?.
  '''
  if dbg: errSL(f'\ncrawl_code_for_edges: {disassembly.code.name}')

  insts = disassembly.insts
  inst_srcs = disassembly.inst_srcs
  inst_opt_reasons = disassembly.inst_opt_reasons

  # Emit edges for each basic block, taking care to represent lines as they will be traced.
  req: EdgesToLines = defaultdict(set) # Maps edges to sets of lines.
  opt: EdgesToLines = defaultdict(set) # Ditto.

  def add_edge(edge: tuple[int,int], line: int, is_req: bool) -> None:
    #if dbg: err_edge('   ' + ("req" if is_req else "opt"), edge, disassembly.short_name)
    (req if is_req else opt)[edge].add(line)

  for inst in insts.values():
    if inst.off < 0: continue # Skip the raised_inst and begin_inst pseudo-instructions.
    srcs = inst_srcs[inst]
    is_req = (inst_opt_reasons[inst] == InstOptReason.REQ)
    #if dbg: errL(f'BB: {bb}; srcs: {sorted(i.off for i in srcs)}')
    #assert srcs, (inst, srcs)
    for src in srcs:
      add_edge((src.off, inst.off), line=inst.line, is_req=is_req)

  return req, opt


def err_edge(label: str, edge: tuple[int,int], name: str) -> None:
  src, dst = edge
  errL(f'{label} edge: {src:4} -> {dst:4}  {name}')


def err_edges(label: str, edges: Iterable[tuple[int,int,Any]], suffix:str='') -> None:
  if suffix: suffix = '  ' + suffix
  sorted_edges = sorted(edges)
  #errSL("ERR_EDGES", len(sorted_edges))
  jump_offs = set()
  for src, dst, _ in sorted_edges:
    if src + 2 < dst: # Jump or extended instruction.
      jump_offs.add(src)
      jump_offs.add(dst)

  start: tuple[int,int,Any]|None = None
  prev: tuple[int,int,Any] = (0, 0, None) # Dummy value.

  def flush() -> None:
    nonlocal start, prev
    assert start
    arrow = '->' if start == prev else '=>'
    errL(f'{label} {start[0]:4} {arrow} {prev[1]:4}  {suffix}')
    start = None
    prev = (0, 0, None) # Dummy value.

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

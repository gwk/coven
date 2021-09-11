#!/usr/bin/env python3
# Dedicated to the public domain under CC0: https://creativecommons.org/publicdomain/zero/1.0/.

import sys
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass, replace
from dis import HAVE_ARGUMENT, get_instructions, hasjabs, hasjrel, opname as opcodes_to_names
from enum import IntEnum
from types import CodeType
from typing import Any, Iterable, Iterator, NamedTuple, Optional, Union

from .opcodes import (CALL_FUNCTION, END_ASYNC_FOR, EXTENDED_ARG, JUMP_ABSOLUTE, JUMP_FORWARD, JUMP_IF_NOT_EXC_MATCH,
  LOAD_CONST, LOAD_FAST, LOAD_GLOBAL, POP_BLOCK, POP_EXCEPT, RAISE_VARARGS, RERAISE, RETURN_VALUE, SETUP_ASYNC_WITH,
  SETUP_FINALLY, SETUP_WITH, WITH_EXCEPT_START, YIELD_FROM, YIELD_VALUE)
from .util import BG_K2, FILL, RST_TXT, TXT_C, TXT_DN, TXT_G, TXT_M, TXT_N, TXT_R, TXT_Y, errL, errSL, visit_nodes


assert sys.version_info >= (3, 10, 0)


def main() -> None:

  parser = ArgumentParser(description='Python bytecode dissassembler and visualizer.')
  parser.add_argument('paths', nargs='*', default=[])
  parser.add_argument('-block', help='Name of code block to display.')
  parser.add_argument('-list-blocks', action='store_true', help='List all code block names.')
  parser.add_argument('-no-src', dest='show_src', action='store_false', help='Do not show interleaved source lines.')
  args = parser.parse_args()

  module_codes = [compile_module(path) for path in args.paths]
  all_codes = get_all_codes(module_codes)

  for code in all_codes:
    if not args.block or args.block == code.short_name:
      disassembly = disassemble(code, dbg=True)
      print(*disassembly.render(show_src=args.show_src), sep='\n')

  if args.list_blocks:
    block_names = [c.short_name for c in all_codes]
    print('block names:', *block_names, sep='\n  ', end='\n\n')
  elif not args.paths:
    errL('No paths provided.')


@dataclass(frozen=True)
class Src:
  path: str
  text: str
  lines: tuple[str,...]

  def __lt__(self, r: 'Src') -> bool: return self.path < r.path

  def __repr__(self) -> str: return f'Src(path={self.path!r})'

  @staticmethod
  def from_path(path: str) -> 'Src':
    with open(path) as f:
      text = f.read()
      lines = text.split('\n')
      if lines and lines[-1] == '':
        lines.pop()
      return Src(path=path, text=text, lines=tuple(lines))


@dataclass(frozen=True)
class Code:
  src: Src
  raw: CodeType

  @property
  def name(self) -> str: return f'{self.src.path.removesuffix(".py")}:{self.raw.co_name}'

  @property
  def short_name(self) -> str: return self.raw.co_name

  def __repr__(self) -> str: return f'Code(name={self.name!r})'

  def __lt__(self, r: 'Code') -> bool:
    if self.src == r.src:
      return (self.raw.co_name, self.raw.co_code) < (r.raw.co_name, r.raw.co_code)
    return self.src < r.src

  def sort_key(self) -> tuple[str, int]:
    return (self.src.path, self.raw.co_firstlineno)

  def sub_codes(self) -> list['Code']:
    'Return a list of all the Code objects referenced by this one.'
    return [Code(src=self.src, raw=c) for c in self.raw.co_consts if isinstance(c, CodeType)]


def compile_module(path: str) -> Code:
  'Get source and compiled code from a path.'
  src = Src.from_path(path)
  raw = compile(source=src.text, filename=path, mode='exec')
  return Code(src=src, raw=raw)


def get_all_codes(codes: list[Code]) -> list[Code]:
  'Extract all reachable code objects from root `code`.'
  return sorted(visit_nodes(start_nodes=codes, visitor=Code.sub_codes), key=Code.sort_key)


@dataclass(frozen=True, order=True)
class Inst:
  off: int
  off_ext: int|None
  line: int
  op: int
  arg: Any

  @property
  def opname(self) -> str: return opcodes_to_names[self.op]

  @property
  def dst_off(self) -> int|None:
    return self.arg if self.op in has_dst_opcodes else None

  @property
  def exec_jump_dst_off(self) -> int|None:
    '''
    An "executable jump" is our term for a jump transition that the interpreter may actually take.
    This is in contrast to the SETUP instructions which are described as having a jump
    but never do so upon execution (they instead push a frameblock with that jump destination).
    '''
    return self.arg if self.op in exec_jump_opcodes else None

  @property
  def has_arg(self) -> bool: return self.op >= ops_start_having_args

  @property
  def arg_repr(self) -> str:
    return repr(self.arg) if self.has_arg else ''

  def arg_suffix(self, prefix:str) -> str:
    return prefix + repr(self.arg) if self.has_arg else ''

  def __repr__(self) -> str:
    arg_suffix = self.arg_suffix(prefix=', arg=')
    off_ext_str = '' if self.off_ext is None else f', ext={self.off_ext}'
    return f'Inst({self.off}{off_ext_str}, line={self.line}, op={self.opname}{arg_suffix})'


  def render_rough(self) -> str:
    'For debugging when Disassembly is not complete.'
    off = self.off
    op = self.op
    sym = ' '
    if op == JUMP_IF_NOT_EXC_MATCH: sym = '^'
    if op in exec_jump_opcodes:
      target = f'jump {self.arg:4}'
    elif op in frameblock_setup_opcodes:
      target = f'push {self.arg:4}'
    else: target = ''
    return f'line:{self.line:>4}  off:{off:>4} {sym} {target:9}  {self.opname:{opname_len}} {self.arg}'


LINE_BEGIN  = OFF_BEGIN  = OP_BEGIN  = -2
LINE_RAISED = OFF_RAISED = OP_RAISED = -4

begin_inst = Inst(off=OFF_BEGIN, off_ext=None, line=LINE_BEGIN, op=OP_BEGIN, arg=None)
raised_inst = Inst(off=OFF_RAISED, off_ext=None, line=LINE_RAISED, op=OP_RAISED, arg=None)


class InstOptReason(IntEnum):
  '''
  Enumeration of reasons why an instruction can be considered optional for coverage purposes.
  '''
  REQ = 0
  RAISED_RERAISE = 1 # sal_src is OP_RAISED, terminal is RERAISE. This is a catch-all pathway that may not be reasonably covered.
  NO_MATCH_RERAISE = 2 #  sal_src is JUMP_IF_NOT_EXC_MATCH, terminal is RERAISE. this is a catch-all pathway that may not be reasonably covered.
  WITH_EXCEPT_START = 3, # sal_src is WITH_EXCEPT_START. Many `with` exception pathways cannot be reasonably covered, e.g. simple calls to `open` that do not fail for a reasonable testing regime.
  POST_EXIT = 4 # sal_src is a call to exit.

class StackVal(NamedTuple):
  '''
  Static approximation of a value on the stack. Each Inst has a tuple of these approximating the value stack.
  '''
  off: int
  op: int
  val: Any


ValStack = tuple[StackVal,]


@dataclass(frozen=True)
class BasicBlock:

  insts: tuple[Inst,...]

  @property
  def start(self) -> Inst: return self.insts[0]

  @property
  def last(self) -> Inst: return self.insts[-1]

  def __repr__(self) -> str: return f'BasicBlock({self.start.off} … {self.last.off})'

  def contains_off(self, off:int) -> bool:
    return self.start.off <= off <= self.last.off

  def chart_char(self, inst:Inst, is_exit:bool) -> str:
    # TODO: indicate entry blocks?
    off = inst.off
    op = inst.op
    if off == self.start.off:
      if off == self.last.off: # Single-instruction block.
        if op in nonadvancing_opcodes: return '╺'
        if op in always_jump_opcodes: return '╼'
        if op in exec_jump_opcodes: return '┮'
        return '┍'
      return '┌' # Start of multi-instruction block.
    if off == self.last.off: # End of multi-instruction block.
      if op in nonadvancing_opcodes: return '└'
      if op in always_jump_opcodes: return '┴'
      if op in exec_jump_opcodes: return '┼'
      return '├'
    if is_exit: return '╵'
    if self.contains_off(off): return '│'
    return ' '


@dataclass(frozen=True)
class Jump:
  src_off: int
  dst_off: int
  is_exc: bool

  @property
  def is_fwd(self) -> bool: return self.src_off <= self.dst_off

  def __contains__(self, off: int) -> bool:
    if self.is_fwd:
      return self.src_off <= off <= self.dst_off
    else:
      return self.dst_off <= off <= self.src_off

  def chart_char(self, off: int) -> str:
    'Visualize the jump using unicode box drawing.'
    if off not in self: return ' '
    if off == self.src_off:
      glyph = '┍' if self.is_fwd else '┕'
    elif off == self.dst_off:
      glyph = '╰' if self.is_fwd else '╭'
    else:
      glyph = '│'
    color = TXT_Y if self.is_exc else TXT_G
    return f'{color}{glyph}'


@dataclass(frozen=True, order=True)
class FrameBlock(Iterable):
  idx: int # Frameblocks are enumerated in instruction order within a code object, counting from 0.
  #^ Note: When a frameblock is replaced with its `is_except_handler` counterpart,
  #^ the two share the same index number.
  op: int
  setup_off: int
  raise_dst_off: int
  is_except_handler: bool
  parent: Optional['FrameBlock']

  def __iter__(self) -> Iterator['FrameBlock']:
    b: FrameBlock|None = self
    while b is not None:
      yield b
      b = b.parent

  def chart_char(self, off: int) -> str:
    'Visualize the frameblock raise jump using unicode box drawing.'
    if not (self.setup_off <= off <= self.raise_dst_off): return ' '
    assert self.setup_off < self.raise_dst_off
    if off == self.setup_off: return '╕'
    if off == self.raise_dst_off: return '╯'
    return '│'


@dataclass
class Disassembly:
  code: Code
  off_lines: dict[int,int] # Instruction offsets to line numbers.
  insts: dict[int,Inst]
  basic_blocks: list[BasicBlock]
  frameblocks: list[FrameBlock]
  jumps: list[Jump]
  inst_dsts: dict[Inst,frozenset[Inst]]
  inst_srcs: dict[Inst,frozenset[Inst]]
  inst_terminals: dict[Inst,frozenset[Inst]] # Terminals are instructions with no forward destinations. This dict maps each intsruction to its ultimate destinations.
  inst_sal_srcs: dict[Inst,frozenset[Inst]] # Salient sources are the upstream instructions that determine whether an instruction sequence is required or optional.
  inst_opt_reasons: dict[Inst,InstOptReason]
  inst_bbs: dict[Inst,BasicBlock]
  inst_frameblocks: dict[Inst,FrameBlock]
  frameblock_dsts_to_setups: dict[Inst,Inst]
  exit_offs: frozenset[int]

  @property
  def name(self) -> str: return self.code.name

  @property
  def short_name(self) -> str: return self.code.short_name


  def fwd_dsts(self, inst:Inst) -> frozenset[Inst]:
    return frozenset(d for d in self.inst_dsts[inst] if d.off > inst.off)


  def fwd_srcs(self, inst:Inst) -> frozenset[Inst]:
    'Note: forward means srcs that precede inst (most sources, excluding backwards jumps).'
    return frozenset(s for s in self.inst_srcs[inst] if s.off < inst.off)


  def render(self, show_src: bool) -> Iterator[str]:
    'Yield lines that make a ANSI-colored text representation of the code block.'

    def frameblock_cols(inst:Inst) -> list[str]:
      block = self.inst_frameblocks.get(inst)
      cols = [f'{TXT_R if b.is_except_handler else TXT_M}{b.idx}' for b in block] if block else []
      cols.reverse()
      if inst.op == POP_EXCEPT:
        cols.append(f'{TXT_R}^')
      elif inst.op == POP_BLOCK:
        cols.append(f'{TXT_M}^')
      return cols

    inst_stack_cols = [frameblock_cols(inst) for inst in self.insts.values()]
    stack_width = max(len(cols) for cols in inst_stack_cols)
    jump_width = len(self.jumps)
    opt_reason_width = max(len(r.name) for r in InstOptReason)

    code_prev = begin_inst
    for inst, stack_cols in zip(self.insts.values(), inst_stack_cols):
      if inst.off < 0: continue # Skip the raised_inst and begin_inst pseudo-instructions.

      if show_src and code_prev.line != inst.line: # Print source line.
        line_num = 'None' if inst.line is None else f'{inst.line:4}'
        line_text = '' if inst.line is None else self.code.src.lines[inst.line-1]
        yield f'{BG_K2}{TXT_N}{line_num}    {TXT_DN}{line_text}{FILL}'

      stack_str = ''.join(stack_cols) + ' ' * (stack_width - len(stack_cols))
      fb_jump_str = ''.join(fb.chart_char(inst.off) for fb in self.frameblocks)
      jump_str = ''.join(j.chart_char(inst.off) for j in self.jumps)

      is_exit = (inst.off in self.exit_offs)
      bb = self.inst_bbs[inst]
      bb_col = bb.chart_char(inst=inst, is_exit=is_exit)
      opt_reason = self.inst_opt_reasons[inst]
      opt_r = '' if opt_reason == InstOptReason.REQ else opt_reason.name
      note = ''
      if is_exit: note = f'{TXT_R} (NoReturn){RST_TXT}'
      #sal_srcs = ','.join(sorted(str(i.off) for i in self.inst_sal_srcs[inst]))
      #terminals = ','.join(sorted(str(i.off) for i in self.inst_terminals[inst]))
      yield (
        #f'    ss:{sal_srcs:12} t:{terminals:12}'
        f'    {inst.off:4} {stack_str} {TXT_R}{fb_jump_str} {jump_str:{jump_width}}'
        f' {TXT_C}{bb_col}{RST_TXT} {inst.opname:{opname_len}} {TXT_N}{opt_r:{opt_reason_width}}{RST_TXT} {inst.arg_repr}{note}')

      code_prev = inst


def disassemble(code:Code, dbg:bool=False) -> Disassembly:
  '''
  '''
  if dbg: errSL(f'\ndisassemble: {code.name}')

  off_lines = build_off_lines(code)
  insts = build_insts(code, off_lines) # Scan raw instructions and convert to our Inst type.

  inst_dsts, inst_srcs, frameblock_dsts_to_setups, jumps, exit_offs = build_inst_mappings(insts) # Step 2: instruction mappings.

  inst_terminals = build_inst_terminals(insts, inst_dsts)
  inst_sal_srcs = build_inst_sal_srcs(insts, inst_srcs, exit_offs=exit_offs)
  inst_opt_reasons = build_inst_opt_reasons(insts=insts, inst_terminals=inst_terminals, inst_sal_srcs=inst_sal_srcs, exit_offs=exit_offs)

  basic_blocks, inst_bbs = build_basic_blocks(insts, inst_dsts, inst_srcs) # Step 3: compute basic blocks.

  frameblocks, inst_frameblocks = calc_frameblocks(insts, inst_srcs, frameblock_dsts_to_setups)
  #^ Step 4: compute frameblock stack for each instruction.

  return Disassembly(code=code, off_lines=off_lines, insts=insts, basic_blocks=basic_blocks, frameblocks=frameblocks, jumps=jumps,
    inst_dsts=inst_dsts, inst_srcs=inst_srcs, inst_sal_srcs=inst_sal_srcs, inst_terminals=inst_terminals,
    inst_opt_reasons=inst_opt_reasons, inst_bbs=inst_bbs, inst_frameblocks=inst_frameblocks,
    frameblock_dsts_to_setups=frameblock_dsts_to_setups, exit_offs=exit_offs)


def build_off_lines(code:Code) -> dict[int,int]:
  off_lines: dict[int,int] = {}
  for s, e, line in code.raw.co_lines(): # type: ignore
    if line is None: line = 0 # As of PEP 626, some lines get no line number. We convert this to zero for simpler type signatures.
    else: assert line > 0
    for off in range(s, e):
      off_lines[off] = line
  return off_lines


def build_insts(code:Code, off_lines:dict[int,int]) -> dict[int,Inst]:
  'Scan raw instructions and convert to our Inst type.'

  # We have two pseudo-instructions that normal and exception entry.
  insts: dict[int,Inst] = { i.off : i for i in (raised_inst, begin_inst) }

  first_extended_arg_off: Optional[int] = None # first preceding EXTENDED_ARG inst.
  #^ EXTENDED_ARG (up to three) can precede an regular instruction.
  #^ We synthesize Inst values using the first offset (the destination address of jumps),
  #^ and omit EXTENDED_ARG entirely from our disassembly.
  #^ In other words, Inst represents logical instructions, and EXTENDED_ARG is a low level detail that is omitted.

  for i in get_instructions(code.raw):
    op = i.opcode

    if op == EXTENDED_ARG:
      if first_extended_arg_off is None:
        first_extended_arg_off = i.offset
      continue # Do not emit EXTENDED_ARG instructions.

    # If there were EXTENDED_ARGs, then use the first offset, since that is what opcode tracing emits.
    off = i.offset if first_extended_arg_off is None else first_extended_arg_off
    line = off_lines[off]

    inst = Inst(off=off, off_ext=first_extended_arg_off, line=line, op=i.opcode, arg=i.argval)
    insts[off] = inst

    first_extended_arg_off = None

  return insts


def build_inst_mappings(insts: dict[int,Inst]) -> tuple[
 dict[Inst, frozenset[Inst]], dict[Inst,frozenset[Inst]], dict[Inst,Inst], list[Jump], frozenset[int]]:

  'Step 2: instruction mappings.'

  dsts_dd: defaultdict[Inst,set[Inst]] = defaultdict(set) # Insts to all destination insts.
  frameblock_dsts_to_setups: dict[Inst,Inst] = {}
  jumps: list[Jump] = []
  #inst_stacks: dict[Inst,ValStack] = {}
  exits: set[int] = set()

  prev: Inst = begin_inst
  #val_stack = []

  # Hacky state machine to detect simple calls to `exit`. TODO: need to use the type information to find called NoReturn functions.
  # Note that this cannot yet handle a conditional case like `exit(1 if pred else 0)`.
  exit_load_global_fn = False
  exit_load_arg = False

  for inst in insts.values():
    if inst.off < 0: continue # Skip the raised_inst and begin_inst pseudo-instructions.

    if prev.op not in nonadvancing_opcodes:
      dsts_dd[prev].add(inst)
    elif prev.op in (YIELD_FROM, YIELD_VALUE):
      dsts_dd[begin_inst].add(inst) # Preceding yield causes coroutine frame to resume at the following inst when done.

    if inst.op == YIELD_FROM:
      dsts_dd[begin_inst].add(inst) # YIELD_FROM alse causes coroutine frame to resume at itself during iteration.

    # Add jumps to dsts. We do not include the SETUP_* destinations as jumps because those edges never actually occur.
    # Instead, some other instruction within the frameblock scope is the source of a "raise" edge.
    if dst_off := inst.exec_jump_dst_off:
      dst = insts[dst_off]
      dsts_dd[inst].add(dst)
      jumps.append(Jump(inst.off, dst.off, is_exc=(inst.op == JUMP_IF_NOT_EXC_MATCH)))

    # Calculate frameblock jump info.
    if inst.op in frameblock_setup_opcodes:
      dst = insts[inst.arg]
      assert dst not in frameblock_dsts_to_setups # Frameblocks destinations appear to be unique.
      frameblock_dsts_to_setups[dst] = inst
      dsts_dd[raised_inst].add(dst) # We choose to represent raise transitions as from the special `raised_inst`.
      #^ This is because the actual source instruction can be from anywhere within the frameblock.

    # Detect simple calls to `exit`. Note that this code reads backwards in terms of state machine dependency steps.
    # This is not foolproof, but detects code like `exit()`, `exit(0)`, and `exit(v)` (globals and local vars).
    if (inst.op == CALL_FUNCTION and ((inst.arg == 0 and exit_load_global_fn) or (inst.arg == 1 and exit_load_arg))):
      exits.add(inst.off)
    exit_load_arg = (exit_load_global_fn and inst.op in (LOAD_CONST, LOAD_FAST, LOAD_GLOBAL)) # Prev was load `exit`, current is load arg.
    exit_load_global_fn = (inst.op == LOAD_GLOBAL and inst.arg == 'exit')

    prev = inst

  srcs_dd: defaultdict[Inst,set[Inst]] = defaultdict(set) # Temp instructions to source insts.
  for src, dst_set in dsts_dd.items():
    for dst in dst_set:
      srcs_dd[dst].add(src)

  inst_dsts: dict[Inst,frozenset[Inst]] = { i : frozenset(dsts_dd.get(i, ())) for i in insts.values() }
  inst_srcs: dict[Inst,frozenset[Inst]] = { i : frozenset(srcs_dd.get(i, ())) for i in insts.values() }

  exit_offs = frozenset(exits)

  return (inst_dsts, inst_srcs, frameblock_dsts_to_setups, jumps, exit_offs)




def build_inst_terminals(insts:dict[int,Inst], inst_dsts:dict[Inst,frozenset[Inst]]) -> dict[Inst,frozenset[Inst]]:
  '''
  Terminals are instructions with no forward destinations.
  '''
  inst_terminals: dict[Inst,frozenset[Inst]] = {}
  for inst in reversed(insts.values()):
    fwd_dsts = sorted(d for d in inst_dsts[inst] if d.off > inst.off)
    if not fwd_dsts:
      inst_terminals[inst] = frozenset((inst,))
    else:
      inst_terminals[inst] = frozenset.union(*(inst_terminals[d] for d in fwd_dsts))
  return inst_terminals


salient_src_ops = frozenset((OP_RAISED, OP_BEGIN, JUMP_IF_NOT_EXC_MATCH, WITH_EXCEPT_START))

def build_inst_sal_srcs(insts:dict[int,Inst], inst_srcs:dict[Inst,frozenset[Inst]], exit_offs: frozenset[int]
 ) -> dict[Inst,frozenset[Inst]]:
  '''
  Salient sources are the upstream instructions that determine whether an instruction sequence is required or optional.
  '''
  def is_salient(i:Inst) -> bool: return (i.op in salient_src_ops or i.off in exit_offs)

  inst_sal_srcs: dict[Inst,frozenset[Inst]] = {}
  for inst in insts.values():
    if inst.off < 0: # The raised_inst and begin_inst pseudo-instructions. No sources.
      sal_srcs = frozenset[Inst]()
    else:
      ss = set[Inst]()
      for src in inst_srcs[inst]:
        if src.off > inst.off: continue # Ignore reverse (loop) edges.
        if is_salient(src): ss.add(src)
        else: ss.update(inst_sal_srcs[src])
      sal_srcs = frozenset(ss)
    inst_sal_srcs[inst] = sal_srcs
  return inst_sal_srcs


def build_inst_opt_reasons(insts:dict[int,Inst], inst_sal_srcs:dict[Inst,frozenset[Inst]], inst_terminals:dict[Inst,frozenset[Inst]],
 exit_offs:frozenset[int]) -> dict[Inst,InstOptReason]:

  'Calculate whether each instruction is required or optional and why.'

  return { inst : calc_inst_opt_reason(inst, inst_terminals, inst_sal_srcs, exit_offs) for inst in insts.values() if inst.off >= 0 }


def calc_inst_opt_reason(inst:Inst, inst_terminals:dict[Inst,frozenset[Inst]], inst_sal_srcs:dict[Inst,frozenset[Inst]],
 exit_offs:frozenset[int]) -> InstOptReason:

  terminals = inst_terminals[inst]
  sal_srcs = inst_sal_srcs[inst]
  if inst.op == WITH_EXCEPT_START or all(s.op == WITH_EXCEPT_START for s in sal_srcs):
    return InstOptReason.WITH_EXCEPT_START
  if all(d.op == RERAISE for d in terminals):
    if all(s.op == OP_RAISED for s in sal_srcs):
      return InstOptReason.RAISED_RERAISE
    if all(s.op == JUMP_IF_NOT_EXC_MATCH for s in sal_srcs):
      return InstOptReason.NO_MATCH_RERAISE # TODO: sal_srcs needs to handle this.
  if all(s.off in exit_offs for s in sal_srcs):
    return InstOptReason.POST_EXIT
  return InstOptReason.REQ


def build_basic_blocks(insts:dict[int,Inst], inst_dsts:dict[Inst,frozenset[Inst]], inst_srcs:dict[Inst,frozenset[Inst]]
 ) -> tuple[list[BasicBlock],dict[Inst,BasicBlock]]:

  'Step 3: Compute basic blocks.'
  inst_bbs: dict[Inst,BasicBlock] = {}
  basic_blocks: list[BasicBlock] = []
  curr_bb_insts: list[Inst] = [] # Temporary buffer of each instruction in the current basic block.

  def flush_bb() -> None:
    if not curr_bb_insts: return
    bb_insts = tuple(curr_bb_insts)
    bb = BasicBlock(bb_insts)
    basic_blocks.append(bb)
    inst_bbs.update((i, bb) for i in curr_bb_insts)
    curr_bb_insts.clear()

  for inst in insts.values():
    if inst.off < 0: continue # Skip the raised_inst and begin_inst pseudo-instructions.
    dsts = inst_dsts[inst]
    srcs = inst_srcs[inst]
    if curr_bb_insts and srcs != frozenset(curr_bb_insts[-1:]): # This instruction is a merge point.
      flush_bb()
    curr_bb_insts.append(inst)
    if inst.op in bb_last_opcodes: # Terminate the current basic block.
      flush_bb()
    else:
      assert len(dsts) == 1
  assert not curr_bb_insts

  return (basic_blocks, inst_bbs)


def calc_frameblocks(insts: dict[int,Inst], inst_srcs: dict[Inst,frozenset[Inst]], frameblock_dsts_to_setups: dict[Inst,Inst]
 ) -> tuple[list[FrameBlock], dict[Inst,FrameBlock]]:
  '''
  Step 4: compute frameblock stack for each instruction.
  Frame blocks are pushed by SETUP_ASYNC_WITH, SETUP_FINALLY, and SETUP_WITH.
  Frame blocks are popped by POP_BLOCK, POP_EXCEPT, and END_ASYNC_FOR (frameobject:PyFrame_BlockPop).
  (See also compile.c:compiler_try_finally, compiler_try_except).
  '''
  frameblocks: list[FrameBlock] = []
  inst_frameblocks: dict[Inst,FrameBlock] = {}

  for inst in insts.values():
    srcs = inst_srcs[inst]
    if setup := frameblock_dsts_to_setups.get(inst):
      assert len(srcs) == 1 and raised_inst in srcs # Frameblock destinations should only be reachable by raising from the frameblock.
      block = emulate_exception_unwind(inst_frameblocks[setup], inst)
    else:
      src_blocks = sorted(filter(None, (inst_frameblocks.get(src) for src in srcs)))
      if src_blocks:
        block = src_blocks[0]
        for b in src_blocks:
          if b != block:
            errSL('ERROR: multiple distinct source blocks:', inst, src_blocks)
      else:
        block = None

    if inst.op in frameblock_setup_opcodes: # Push a new frame.
      block = FrameBlock(idx=len(frameblocks), op=inst.op, setup_off=inst.off, raise_dst_off=inst.arg, is_except_handler=False, parent=block)
      frameblocks.append(block)
    elif inst.op in frameblock_pop_opcodes: # Pop the top frame.
      if block is None:
        errSL('ERROR: frameblock pop on empty stack:', inst)
      else:
        if inst.op == POP_EXCEPT and not block.is_except_handler:
          errSL('ERROR: POP_EXCEPT did not receive an EXCEPT_HANDLER block:', inst)

        block = block.parent
    if block is not None:
      inst_frameblocks[inst] = block

  return frameblocks, inst_frameblocks


def emulate_exception_unwind(block: FrameBlock|None, inst:Inst) -> FrameBlock|None:
  while block is not None:
    if block.is_except_handler: # Pop the EXCEPT_HANDLER block and continue.
      block = block.parent
    else: # Replace the SETUP_FINALLY block with an EXCEPT_HANDLER block and return.
      return replace(block, is_except_handler=True)
  # Frame stack is empty. This is an error.
  errSL('ERROR: exception_unwind emptied the block stack:', inst)
  return None


# Opcode information.

opname_len = max(len(name) for name in opcodes_to_names)

ops_start_having_args = HAVE_ARGUMENT # Opcodes less than this number ignore their arguments.

frameblock_setup_opcodes = {
#^ Opcodes that push a frame block specifying a block destination.
#^ A raised exception will unwind and then land at the block destination.
  SETUP_ASYNC_WITH,
  SETUP_FINALLY,
  SETUP_WITH,
}

frameblock_pop_opcodes = {
#^ Block frames are popped by frameobject:PyFrame_BlockPop, which is called only by these cases in ceval.c.
  END_ASYNC_FOR, # Conditioned upon `PyErr_GivenExceptionMatches(exc, PyExc_StopAsyncIteration)`.
  POP_BLOCK, # Unconditional. ceval.c:2669.
  POP_EXCEPT, # Unconditional. ceval.c:2691.
}

frameblock_setup_abbrs = {
  SETUP_ASYNC_WITH: 'A',
  SETUP_FINALLY:    'F',
  SETUP_WITH:       'W',
}

has_dst_opcodes = { *hasjabs, *hasjrel }
#^ The jumping opcodes, as well as frameblock setup opcodes that do not immediately branch.

exec_jump_opcodes = has_dst_opcodes - frameblock_setup_opcodes
#^ Opcodes that may jump during execution.
#^ The SETUP_* ops do not actually jump on execution, but instead store the jump destination in a new frameblock.

always_jump_opcodes = { # Opcodes that always jump on execution.
  JUMP_ABSOLUTE,
  JUMP_FORWARD,
}

branching_opcodes = exec_jump_opcodes - always_jump_opcodes

raise_opcodes = { # Opcodes that raise exceptions.
  RAISE_VARARGS,
  RERAISE,
}

nonadvancing_opcodes = { # Opcodes never advance to the next instruction.
  *always_jump_opcodes,
  *raise_opcodes,
  RETURN_VALUE,
  YIELD_VALUE,
}

bb_last_opcodes = exec_jump_opcodes | nonadvancing_opcodes # Opcodes that end a basic block.


if __name__ == '__main__': main()

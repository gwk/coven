# Dedicated to the public domain under CC0: https://creativecommons.org/publicdomain/zero/1.0/.

from typing import TypeVar, Hashable, Iterable, Callable, Any
from sys import stderr
from pprint import pprint


_H = TypeVar('_H', bound=Hashable)

def visit_nodes(start_nodes:Iterable[_H], visitor:Callable[[_H], Iterable[_H]]) -> set[_H]:
  'Walk a graph from `start_nodes` using `visitor`.'
  remaining = set(start_nodes)
  visited = set()
  while remaining:
    node = remaining.pop()
    assert node not in visited
    visited.add(node)
    discovered = visitor(node)
    remaining.update(n for n in discovered if n not in visited)
  return visited


El = TypeVar('El')

def first(iterable: Iterable[El]) -> El: return next(iter(iterable))


def errL(*items: Any) -> None: print(*items, sep='', file=stderr)

def errLL(*items: Any) -> None: print(*items, sep='\n', file=stderr)

def errP(item: Any) -> None: pprint(item, stream=stderr)

def errSL(*items: Any) -> None: print(*items, file=stderr)

def errLSSL(*items: Any) -> None: print(*items, sep='\n  ', file=stderr)


RST = '\x1b[0m'
TXT_B = '\x1b[34m'
TXT_C = '\x1b[36m'
TXT_G = '\x1b[32m'
TXT_M = '\x1b[35m'
TXT_R = '\x1b[31m'
TXT_Y = '\x1b[33m'

TXT_KD= '\x1b[38;5;235m'
TXT_D = '\x1b[38;5;238m'
TXT_DN= '\x1b[38;5;241m'
TXT_N = '\x1b[38;5;244m'
TXT_NL= '\x1b[38;5;247m'
TXT_L = '\x1b[38;5;250m'
TXT_LW= '\x1b[38;5;253m'

RST = '\x1b[0m'
FILL = '\x1b[0K\x1b[m'
RST_TXT = '\x1b[39m'

BG_K2 = '\x1b[48;5;233m'
BG_KD = '\x1b[48;5;235m'
BG_D  = '\x1b[48;5;238m'
BG_DN = '\x1b[48;5;241m'
BG_N  = '\x1b[48;5;244m'
BG_NL = '\x1b[48;5;247m'
BG_L  = '\x1b[48;5;250m'
BG_LW = '\x1b[48;5;253m'

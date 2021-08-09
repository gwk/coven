# Dedicated to the public domain under CC0: https://creativecommons.org/publicdomain/zero/1.0/.

import re
import os

from os.path import abspath as abs_path, join as path_join, normpath as normalize_path
from typing import Iterable, Iterator, Sequence
from argparse import Namespace

from .analysis import Coverage, err_edges
from .util import errLSSL, TXT_L, TXT_C, TXT_Y, TXT_R, RST, TXT_D, TXT_M, TXT_G, TXT_B


class Stats:

  def __init__(self) -> None:
    self.lines = 0
    self.trivial = 0
    self.traceable = 0
    self.covered = 0
    self.ignored = 0
    self.ignored_but_covered = 0
    self.not_covered = 0

  def add(self, stats: 'Stats') -> None:
    self.lines += stats.lines
    self.trivial += stats.trivial
    self.traceable += stats.traceable
    self.covered += stats.covered
    self.ignored += stats.ignored
    self.ignored_but_covered += stats.ignored_but_covered
    self.not_covered += stats.not_covered

  def fmt_stat(self, name: str, val: int, colorize: bool) -> str:
    colors = {
      'trivial' : TXT_L if colorize else '',
      'ignored' : TXT_C if colorize else '',
      'ignored_but_covered' : TXT_Y if colorize else '',
      'not_covered' : TXT_R if colorize else '',
    }
    color = colors.get(name, '') if val > 0 else ''
    rst = RST if color else ''
    display_name = name.replace('_', ' ')
    return f'{color}{val} {display_name}{rst}'

  def describe(self, label: str, colorize: bool) -> None:
    s = self
    described_stats = [(name, getattr(self, name)) for name in ['traceable', 'covered', 'ignored', 'ignored_but_covered', 'not_covered']]
    print(label, ': ', '; '.join(self.fmt_stat(name, val, colorize) for name, val in described_stats), '.', sep='')


def report_path(target: str, path: str, coverage: Coverage, totals: Stats, args: Namespace) -> None:

  dbg_name = args.block
  line_sets = coverage.line_sets

  line_texts = [text.rstrip() for text in open(path).readlines()]
  ignored_lines, explicitly_ignored_lines = calc_ignored_lines(line_texts)

  covered_lines = set() # line indices that are perfectly covered.
  ign_cov_lines = set()
  not_cov_lines = set()

  for line, (required, matched) in line_sets.items():
    if line == 0: continue # Just skip the edges that were not attributed to any line (PEP 626).
    if matched >= required:
      if line in explicitly_ignored_lines:
        ign_cov_lines.add(line)
      else:
        covered_lines.add(line)
    elif line not in ignored_lines:
      not_cov_lines.add(line)

  problem_lines = ign_cov_lines | not_cov_lines

  length = len(line_texts)
  stats = Stats()
  stats.lines = length
  stats.trivial = max(0, length - len(line_sets))
  stats.traceable = len(line_sets)
  stats.covered = len(covered_lines)
  stats.ignored_but_covered = len(ign_cov_lines)
  stats.not_covered = len(not_cov_lines)
  stats.ignored = len(ignored_lines - covered_lines - ign_cov_lines - not_cov_lines)
  totals.add(stats)

  rel_path = path_rel_to_current_or_abs(path)
  label = f'\n{target}: {rel_path}'
  if not problem_lines:
    stats.describe(label, args.color)
    return

  RST1 = RST if args.color else ''
  TXT_B1 = TXT_B if args.color else ''
  TXT_C1 = TXT_C if args.color else ''
  TXT_D1 = TXT_D if args.color else ''
  TXT_G1 = TXT_G if args.color else ''
  TXT_L1 = TXT_L if args.color else ''
  TXT_M1 = TXT_M if args.color else ''
  TXT_R1 = TXT_R if args.color else ''
  TXT_Y1 = TXT_Y if args.color else ''
  print(label, ':', sep='')
  if args.show_all:
    reported_lines:Sequence[int] = range(1, length + 1) # entire document, 1-indexed.
  else:
    reported_lines = sorted(problem_lines)
  ranges = line_ranges(reported_lines, before=4, after=1, terminal=length+1)
  for r in ranges:
    if r is None:
      print(f'{TXT_D1} ...{RST1}')
      continue
    for line in r:
      text = line_texts[line - 1] # line is a 1-index.
      color = RST1
      sym = ' '
      needs_dbg = False
      if line not in line_sets: # trivial.
        color = TXT_L1
      else:
        required, matched = line_sets[line]
        if line in ign_cov_lines:
          color = TXT_Y1
          sym = '?'
        elif line in ignored_lines:
          color = TXT_C1
          sym = '|'
        elif line in not_cov_lines:
          color = TXT_R1
          if matched:
            sym = '%'
            needs_dbg = True
          else: # no coverage.
            sym = '!'
        else: assert line in covered_lines
      print(f'{TXT_D1}{line:4} {color}{sym} {text}{RST1}'.rstrip())
      if dbg_name and needs_dbg:
        #print(f'     {TXT_B1}^ required:{len(required)} traced:{len(traced)}.{RST1}')
        err_edges(f'{TXT_D1}{line:4} {TXT_B1}-', required - matched)
        err_edges(f'{TXT_D1}{line:4} {TXT_B1}=', matched)
  stats.describe(label, args.color)
  if args.list_blocks:
    errLSSL('\nBlock names:', *[c.raw.co_name for c in coverage.all_codes])



def path_rel_to_current_or_abs(path: str) -> str:
  ap = abs_path(path)
  ac = abs_path('.')
  comps = path_comps(ap)
  prefix = path_comps(ac)
  if comps == prefix:
    return '.'
  if prefix == comps[:len(prefix)]:
    return path_join(*comps[len(prefix):])
  return ap


def path_comps(path: str) -> list[str]:
  np = normalize_path(path)
  if np == '/': return ['/']
  assert not np.endswith('/')
  return [comp or '/' for comp in np.split(os.sep)]


indent_and_ignored_re = re.compile(r'''(?x:
(\s*) # capture leading space.
( .* (?P<directive> \#!cov-ignore )
| assert\b
| if \s+ __name__ \s* == \s* ['"]__main__['"] \s* :
)?
)''')

def calc_ignored_lines(line_texts: list[str]) -> tuple[set[int], set[int]]:
  explicit:set[int] = set()
  implicit:set[int] = set()
  indent = -1
  is_directive = False
  for line, text in enumerate(line_texts, 1):
    m = indent_and_ignored_re.match(text)
    assert m
    ind = m.end(1) - m.start(1)
    if m.lastindex == 2: # matched one of the ignore triggers.
      is_directive = bool(m.group('directive')) # explicit ignore.
      (explicit if is_directive else implicit).add(line)
      indent = ind
    elif -1 < indent < ind:
      (explicit if is_directive else implicit).add(line)
    else:
      indent = -1
  return (explicit | implicit), explicit


def line_ranges(iterable: Iterable[int], before: int, after: int, terminal: int) -> Iterator[range|None]:
  'Group individual line numbers (1-indexed) into chunks, with interstitial `None` values representing breaks.'
  assert terminal > 0
  it = iter(iterable)
  try:
    i = next(it)
    assert i > 0
  except StopIteration: return
  start = i - before
  end = i + after + 1
  for i in it:
    assert i > 0
    # +1 bridges chunks that would otherwise elide a single line, appearing replaced by '...'.
    if end + 1 < i - before:
      yield range(max(1, start), min(end, terminal))
      yield None # interstitial None causes '...' to be printed.
      start = i - before
    end = i + after + 1
  yield range(max(1, start), min(end, terminal))

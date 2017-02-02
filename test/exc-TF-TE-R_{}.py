
from exc import *

def top(arg):
  try:
    try:
      try_(arg, raise_start=2)
    except:
      exc()
    if arg == 1:
      raise TestException # this tests the is_TEF heuristic.
  finally:
    fin()

handle_args(top)


from exc import *

def top(arg):
  try:
    try_(arg)
  except E1:
    exc()
  finally:
    fin()

handle_args(top)

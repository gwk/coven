
from exc import *

def top(arg):
  try:
    try_(arg)
  except E1:
    exc()
  else:
    else_()

handle_args(top)

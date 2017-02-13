from exc import *

def top(arg):
  with CM(False):
    try:
      return try_(arg)
    finally:
      return None

handle_args(top)

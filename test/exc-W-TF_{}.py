from fixtures import *

def top(arg):
  with CM(False):
    try:
      return try_(arg)
    finally:
      return -1

handle_args(top)

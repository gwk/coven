from fixtures import *

def top(arg):
  with CM(False):
    try:
      return try_(arg)
    except:
      return None

handle_args(top)

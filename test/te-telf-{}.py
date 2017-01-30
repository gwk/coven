
from sys import argv


def top(arg):
  try:
    try:
      try_(arg)
    except ValueError:
      except_ValueError()
    else:
      else_()
    finally:
      finally_()
  except IndexError:
    pass

def try_(arg):
  if arg == 1:
    raise ValueError
  if arg == 2:
    raise IndexError


def except_ValueError(): pass
def else_(): pass
def finally_(): pass


for a in argv[1]: top(int(a))

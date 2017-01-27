
from sys import argv


def top(arg):
  try:
    try_(arg)
  except ValueError:
    except_ValueError()
  except Exception:
    except_Exception()
  else:
    else_()
  finally:
    finally_()


def try_(arg):
  if arg == 1:
    raise ValueError(arg)
  if arg == 2:
    raise Exception(arg)


def except_ValueError(): pass
def except_Exception(): pass
def else_(): pass
def finally_(): pass

for a in argv[1]: top(int(a))

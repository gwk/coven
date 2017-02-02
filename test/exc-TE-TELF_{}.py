
from sys import argv


def top(arg):
  try:
    try:
    if arg:
      raise Exception if arg == 1 else BaseException
    except Exception:
      except_Exception()
    else:
      else_()
    finally:
      finally_()
  except BaseException:
    except_BaseException()

def except_Exception(): pass
def else_(): pass
def finally_(): pass
def except_BaseException(): pass

for a in argv[1]: top(int(a))

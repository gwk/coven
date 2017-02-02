# exc-* test helper module.

from sys import argv

class TestException(Exception): pass
class E1(TestException): pass
class E2(TestException): pass
class E3(TestException): pass

def try_(arg, raise_start=1):
  if arg < raise_start: return
  if arg == 1: raise E1
  if arg == 2: raise E2
  if arg == 3: raise E3
  raise Exception(f"BAD ARG: {arg}")


def exc(*e): pass
def else_(): pass
def fin(): pass


def handle_args(fn):
  for char in argv[1]:
    i = int(char)
    try:
       fn(i)
    except TestException as e:
      print(f'handle_args: char:{char}; exception: {e!r}.')

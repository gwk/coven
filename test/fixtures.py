from sys import argv


'Test fixtures.'


class TestException(Exception): pass
class E1(TestException): pass
class E2(TestException): pass
class E3(TestException): pass


class CM:
  def __init__(self, silence: bool) -> None: self.silence = silence
  def __enter__(self) -> 'CM': return self
  def __exit__(self, *exc_info) -> bool: return self.silence


# TODO: CM that raises in __enter__, and one in __exit__.


def try_(arg):
  if arg == 0: return 0
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

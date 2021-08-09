from fixtures import TestException, exc, fin, handle_args, E2


def top(arg):
  try:
    try:
      try2(arg)
    except:
      exc()
    if arg == 1:
      raise TestException
  finally:
    fin()


def try2(arg):
  if arg == 0: return 0
  if arg == 1: return 1
  raise E2
  

handle_args(top)

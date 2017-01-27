
from sys import argv


def top(arg):

  r_if = 0
  if arg: r_if = (1 if arg == 1 else arg)
  assert r_if == arg

  r_while = 0
  i = arg
  while i: r_while = (1 if arg == 1 else arg); i -= 1
  assert r_while == arg

  r_for = 0
  for i in range(arg): r_for = (1 if arg == 1 else arg)
  assert r_for == arg

  try: raises_if(arg)
  except Exception: r_exc = (1 if arg == 1 else arg)
  else: r_exc = (1 if arg == 1 else arg)
  finally: r_fin = (1 if arg == 1 else arg)
  assert r_exc == arg
  assert r_fin == arg


def raises_if(arg):
  if arg: raise Exception(arg)


for a in argv[1]: top(int(a))

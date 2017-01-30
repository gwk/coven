
from sys import argv


def top(arg):

  r_if = 0
  if arg: r_if = (1 if arg == 1 else arg)
  else: r_if == arg # 0 case only.

  r_while = 0
  i = arg
  while i: r_while = (1 if arg == 1 else arg); i -= 1

  r_for = 0
  for i in range(arg): r_for = (1 if arg == 1 else arg)

  try: raises_if(arg)
  except Exception: r_exc = (1 if arg == 1 else arg)
  else: r_exc = arg # 0 case only.
  finally: r_fin = (1 if arg == 1 else arg)


def raises_if(arg):
  if arg: raise Exception(arg)


for a in argv[1]: top(int(a))

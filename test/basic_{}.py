# Dedicated to the public domain under CC0: https://creativecommons.org/publicdomain/zero/1.0/.

from sys import argv


# Test tracing for basic conditionals and function calls.

def top(arg):

  if arg:
    inlined()
    multi()

  if arg:
    f()
  else:
    g()


def inlined(): return None

def multi():
  return None

def f(): pass

def g():
  pass


for c in argv[1]: top(int(c))

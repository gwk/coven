
from sys import argv


def top(arg):
  if arg:
    inlined()
    multi()

def inlined(): return None

def multi():
  return None


for a in argv[1]: top(int(a))

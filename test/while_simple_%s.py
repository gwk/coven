from sys import argv


def while_simple(i):
  '''
  Python 3.10 duplicates the `while` test at the beginning and end of the loop.
  Complete coverage requires that both branches of both tests are exercised.
  Thus i=0 and i=1 are not enough for complete code covereage, but i=0 and i=2 are.
  '''
  while i:
   in_while_simple()
   i -= 1


def in_while_simple(): pass

for c in argv[1]:
  while_simple(int(c))

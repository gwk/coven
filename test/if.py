
from sys import argv


def top(arg):

  if arg:
    if_()

  if arg:
    if_()
  else:
    else_()


def if_(): pass
def else_(): pass

for a in argv[1]: top(int(a))

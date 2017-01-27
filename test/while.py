
from sys import argv


def top(arg):

  i = arg
  while i:
   while_()
   i -= 1

  i = arg
  while i:
    while_()
    if i == 2:
      break
    i -= 1
  else:
    else_()


def while_(): pass
def else_(): pass

for a in argv[1]: top(int(a))

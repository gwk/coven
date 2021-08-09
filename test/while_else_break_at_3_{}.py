from sys import argv


def while_else(i):
  while i:
    in_while()
    if i == 3:
      break
    i -= 1
  else:
    in_else()


def in_while(): print('While.')
def in_else(): print('Else.')


for c in argv[1]:
  while_else(int(c))

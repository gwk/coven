# The example from cpython lnotabs_notes.txt.
from sys import argv


def f(a):
  while a:
    print('W')
    break
  else:
    print('E')

f(int(argv[1]))

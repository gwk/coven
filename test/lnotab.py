# The example (in spirit) from cpython lnotabs_notes.txt.

from sys import argv

x = ''
def top(a):
  global x
  while a:
    x = a
    break
  else:
    x = a


for a in argv[1]:
  i = int(a)
  top(i)
  assert x == i

from sys import argv


'''
The example (approximately) from cpython lnotabs_notes.txt.
Note that as of Python 3.10 / PEP 626, lnotabs details are no longer relevant to coven.
'''

x = ''
def top(a):
  global x
  while a:
    x = a
    break
  else:
    x = a


for c in argv[1]:
  i = int(c)
  top(i)
  assert x == i

from sys import argv

def yield_(stop):
  for i in range(2):
    if i == stop: break
    yield i

def top(arg):
  print(*yield_(arg))

for a in argv[1]: top(int(a))

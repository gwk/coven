
from sys import argv
from exc import *

def top(arg):
  try: try_(arg)
  except E1: exc1()

for a in argv[1]:
  try: top(int(a))
  except: pass

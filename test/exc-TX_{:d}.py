
from fixtures import *

def top(arg):
  try:
    try_(arg)
  except:
    exc()

handle_args(top)

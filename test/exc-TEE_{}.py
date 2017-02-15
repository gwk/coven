
from fixtures import *

def top(arg):
  try:
    try_(arg)
  except E1:
    exc()
  except E2:
    exc()

handle_args(top)


from exc import *

def top(arg):
  try:
    try_(arg)
  finally:
    fin()

handle_args(top)

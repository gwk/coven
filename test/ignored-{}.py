
from sys import argv


def top(on):
  if on:
    inline(on)
    multi(on)
    partial(on)

def inline(on): return on if on else None #no-cov!

def multi(on):
  if on: #no-cov!
    return on
  return None #no-cov!

def partial(on):
  if on: #no-cov!
    return on
  return None # never covered; this test assures that previous annotation is terminated.


for a in argv[1]: top(int(a))

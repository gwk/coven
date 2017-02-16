
from sys import argv


def top(on):
  if on:
    inline(on)
    multi(on)
    partial(on)

def inline(on): return on if on else -1 #!cov-ignore.

def multi(on):
  if on: #!cov-ignore.
    return on
  return -1 #!cov-ignore.

def partial(on):
  if on: #!cov-ignore.
    return on
  return -1 # never covered; this test assures that previous annotation is terminated.


for a in argv[1]: top(int(a))

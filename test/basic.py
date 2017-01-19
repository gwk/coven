#!/usr/bin/env python3

from sys import argv


def main():
  arg = int(argv[1])
  if arg:
    called_one()
    called_multi()
    ignored_but_called_one()
    ignored_but_called_multi()
    # Padding lines to force hunk split.
    # THIS PADDING LINE SHOULD NOT APPEAR IN OUTPUT HUNK CONTEXT.
    # Padding lines to force hunk split.
    # ditto.


def called_one(): return None

def called_multi():
  return None


# the if statements below need a variable condition,
# because the python3.6.0 compiler is smart enough to elide the `if` when passed a constant.
x = 0


def ignored_but_called_one(): return None #no-cov!

def ignored_but_called_multi():
  if x: #no-cov!
    return None
  return None

def ignored_never_called_one(): return None #no-cov!

def ignored_never_called_multi():
  if x: #no-cov!
    return None
  return None #no-cov!


if __name__ == '__main__': main()

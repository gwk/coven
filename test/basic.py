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
# ^
# ^

def called_one(): return None

def called_multi():
  return None


def ignored_but_called_one(): return None #no-cov!

def ignored_but_called_multi():
  return None #no-cov!


def ignored_never_called_one(): return None #no-cov!

def ignored_never_called_multi():
  return None #no-cov!


if __name__ == '__main__': main()

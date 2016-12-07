#!/usr/bin/env python3

from sys import argv


def one_line(): return None #no-cov!

def multi_line():
  return None

arg = int(argv[1])

if arg:
  one_line()
  multi_line()

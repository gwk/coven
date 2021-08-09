from sys import argv


'''
These tests exercised FOR_ITER cases that were tricky when coven was first written for python 3.7.
Their efficacy has yet to be evaluated for 3.10.
'''

def dummy(): pass # add non-trivial loop body and return to make sure that StopIteration gets the right line.

def iter_multi():
  for i in range(1):
    dummy()
  dummy()

def iter_inline():
  for i in range(1): dummy()

def iter_trivial():
  for i in range(1): pass

def gen_multi():
  for i in (j for j in range(1)):
    dummy()
  dummy()

def gen_inline():
  for i in (j for j in range(1)): dummy()

def gen_trivial():
  for i in (j for j in range(1)): pass

iter_multi()
iter_inline()
iter_trivial()

gen_multi()
gen_inline()
gen_trivial()

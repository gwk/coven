from sys import argv

'''
This tests the FOR_ITER exception edge.
It is unusual because we are converting jump edges (rather than raises) to exception edges,
and because the static exception edge is given a real line number (that of FOR_ITER).
Without the real line number, non-raising iterators will get misreported.
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

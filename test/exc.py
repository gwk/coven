# exc-* test helper module.

class E1(Exception): pass
class E2(Exception): pass
class E3(Exception): pass

def try_(arg):
  if arg == 1: raise E1
  if arg == 2: raise E2
  if arg == 3: raise E3

def exc1(): pass
def exc2(): pass
def exc3(): pass

def fin1(): pass
def fin2(): pass

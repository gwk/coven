from fixtures import handle_args


def gen():
  for i in range(1):
    yield i

def yield_from_range():
  yield from range(1)

def yield_from_gen():
  yield from gen()

def top(arg):
  yfr = yield_from_range()
  yfg = yield_from_gen()
  for _ in range(arg): # Iterate a specified number of steps, to test early exit and exhaustion of the generators.
    assert next(yfr, None) == next(yfg, None)

handle_args(top)

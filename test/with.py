class CM:
  def __init__(self, silence): self.silence = silence
  def __enter__(self): pass
  def __exit__(self, *exc_info): return self.silence

# TODO: CM that raises in __enter__, and one in __exit__.

def top():
  with CM(False):
    raise Exception

top()

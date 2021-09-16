from fixtures import E1, exc, fin, handle_args, try_


def top(arg):
  try:
    try_(arg)
  except E1:
    exc()
  finally:
    fin()

handle_args(top)

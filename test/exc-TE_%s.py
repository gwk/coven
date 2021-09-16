from fixtures import E1, exc, handle_args, try_


def top(arg):
  try:
    try_(arg)
  except E1:
    exc()

handle_args(top)

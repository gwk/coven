from fixtures import try_, exc, handle_args, E1


def top(arg):
  try:
    try_(arg)
  except E1 as e1:
    exc(e1)

handle_args(top)

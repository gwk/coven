from fixtures import E1, else_, exc, handle_args, try_


def top(arg):
  try:
    try_(arg)
  except E1:
    exc()
  else:
    else_()

handle_args(top)

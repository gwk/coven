from fixtures import exc, handle_args, try_


def top(arg):
  try:
    try_(arg)
  except:
    exc()

handle_args(top)

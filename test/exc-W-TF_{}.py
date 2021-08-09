from fixtures import CM, handle_args, try_


def top(arg):
  with CM(silence=False):
    try:
      return try_(arg)
    finally:
      return -1
  return res


handle_args(top)

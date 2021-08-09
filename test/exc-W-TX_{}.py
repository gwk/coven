from fixtures import CM, handle_args, try_


def top(arg):
  with CM(silence=False):
    try:
      return try_(arg)
    except:
      return None

handle_args(top)

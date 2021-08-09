from fixtures import fin, handle_args, try_


def top(arg):
  try:
    try_(arg)
  finally:
    fin()

handle_args(top)

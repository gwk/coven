from sys import argv


def not_all(args):
  return all(int(a) for a in args)

print(not_all(argv[1]))

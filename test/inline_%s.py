from sys import argv


def test_if(i:int) -> None:
  r_if = 0
  if i: r_if = (1 if i == 1 else i)
  else: r_if == i # 0 case only.


def test_while(i:int) -> None:
  r_while = 0
  i = i
  while i: r_while = (1 if i == 1 else i); i -= 1


def test_for(i:int) -> None:
  r_for = 0
  for i in range(i): r_for = (1 if i == 1 else i)


def test_TEEF(i:int) -> None:
  try: raises_if(i)
  except Exception: r_exc = (1 if i == 1 else i)
  else: r_exc = i # 0 case only.
  finally: r_fin = (1 if i == 1 else i)


def raises_if(i:int) -> None:
  if i: raise Exception(i)


for c in argv[1]:
  i = int(c)
  test_if(i)
  test_while(i)
  test_for(i)
  test_TEEF(i)

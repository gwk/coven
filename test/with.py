from fixtures import CM

def top() -> None:
  with CM(silence=False) as cm:
    print(cm.silence)

top()

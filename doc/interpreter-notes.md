# CPython 3.10rc1 Interpreter Notes

The main difficulty that Coven faces when statically analyzing CPython bytecode is understanding the frame block stack. This is an internal manangement structure in the interpreter that keeps track of exception handler setup and teardown. Coven needs to understand try/except/finally structures because they imply which paths through the bytecode should be reasonably expected for coverage.

For example, if a function body wraps a dictionary access with `try` and `except KeyError` then Coven wants to ensure that the `except` clause is covered. This seems straightforward but internally the `except` generates a `JUMP_IF_NOT_EXC_MATCH` instruction, which matches the runtime exception to the expected `KeyError`, and jumps to a `RERAISE` if it does not match. In many cases, the `RERAISE` is not practically reachable, because `KeyError` is the only exception that can possibly be raised. Thus Coven needs to distinguish between `except` clauses and their implicit catch-all "no-match" branches.

# Exception Unwinding

There are currently only two kinds of frameblocks: SETUP_FINALLY and EXCEPT_HANDLER. The three SETUP_* opcodes push SETUP_FINALLY frameblocks. The `exception_unwind` section creatse EXCEPT_HANDLER frameblocks.

See the `exception_unwind:` label in `ceval.c`.
`exception_unwind` is a label in the interpreter main loop. It is jumped to from three opcodes: `RAISE_VARARGS`, `RERAISE`, and `END_ASYNC_FOR`. The label occurs at the end of interpreter main loop.

exception_unwind:
```
while stack is not empty:
  b = pop current block;
  if b.type is EXCEPT_HANDLER:
    rectify exception state
    continue
  else:
    unwind b # Manipulates value stack.
    if b.type is SETUP_FINALLY:
      b1 = push_block(FrameBlock(type=EXCEPT_HANDLER))
      rectify exception state
      jump to b.handler
      goto main_loop. # Done.
error
```

Roughly speaking, exception_unwind pops EXCEPT_HANDLER blocks and then the top SETUP_FINALLY block with a new EXCEPT_HANDLER block.

Crucially, this is the only place where EXCEPT_HANDLER frameblocks are pushed.


POP_EXCEPT:
```
b = pop_block()
assert b.type is EXCEPT_HANDLER
rectify exception state
```

Pops the top EXCEPT_HANDLER block.


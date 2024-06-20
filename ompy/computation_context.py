from collections import deque
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Generator


@dataclass
class ComputationContext:
    variables: list = field(default_factory=list)

    def push(self, value):
        self.variables.append(value)

    def __getitem__(self, key):
        return self.variables[key]

    def __contains__(self, item):
        return item in self.variables

    def __iter__(self):
        return iter(self.variables)


__ACTIVE_CONTEXTS: deque[ComputationContext] = deque()


def active_context() -> ComputationContext | None:
    if len(__ACTIVE_CONTEXTS) == 0:
        return None
    return __ACTIVE_CONTEXTS[-1]


def push_context(context) -> None:
    __ACTIVE_CONTEXTS.append(context)


def delete_context(context: ComputationContext) -> None:
    popped = __ACTIVE_CONTEXTS.pop()
    if context != popped:
        raise RuntimeError("Popped context does not match the provided context")
    for var in popped:
        var._disconnect_context(popped)


@contextmanager
def new_context() -> Generator[ComputationContext, None, None]:
    context = ComputationContext()
    push_context(context)
    try:
        yield context
    finally:
        delete_context(context)
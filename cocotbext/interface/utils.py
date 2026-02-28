from cocotb import start_soon
from cocotb.triggers import Event, ReadOnly
from cocotb.utils import get_sim_time
import re
from functools import lru_cache

@lru_cache(maxsize=128)
def _get_compiled_pattern(expr: str):
    if expr.startswith("/") and expr.endswith("/"):
        return re.compile(expr[1:-1])

    escape_pattern = re.escape(expr)
    escape_pattern = escape_pattern.replace(r"\*", ".*")
    escape_pattern = escape_pattern.replace(r"\+", ".+")
    escape_pattern = escape_pattern.replace(r"\?", ".")
    return re.compile(escape_pattern)


def is_match(expr: str, string: str) -> bool:
    pattern = _get_compiled_pattern(expr)
    return bool(pattern.fullmatch(string))


class ReadOnlyManager:
    _event = None
    _last_time = -1

    @classmethod
    async def wait(cls):
        current_time = get_sim_time()

        # If we are in a new timestep, we need a new ReadOnly trigger
        if current_time > cls._last_time:
            cls._last_time = current_time
            cls._event = Event()
            start_soon(cls._run())

        # If cls._event is None, it means we are calling this
        # for the first time or in a tricky edge case.
        if cls._event is None:
            return

        await cls._event.wait()

    @classmethod
    async def _run(cls):
        try:
            await ReadOnly()
        except RuntimeError:
            # If we are ALREADY in ReadOnly, awaiting it fails.
            # In this case, we just proceed because values are safe to read.
            pass

        if cls._event:
            cls._event.set()

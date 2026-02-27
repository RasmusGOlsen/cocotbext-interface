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

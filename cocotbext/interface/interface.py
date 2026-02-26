from __future__ import annotations

from cocotb.handle import SimHandleBase

import fnmatch
import re
from typing import TypeAlias, Union

from cocotb import top
from cocotb.handle import (
    ArrayObject,
    IntegerObject,
    LogicArrayObject,
    LogicObject,
    RealObject,
)

# Comprehensive Signal Alias for cocotb 2.0
Signal: TypeAlias = Union[
    LogicObject,      # For std_logic, logic, reg
    LogicArrayObject, # For std_logic_vector and packed arrays of logic and reg
    ArrayObject,      # For unpacked array types
    IntegerObject,    # For Integer/Enum types
    RealObject        # For Real/Float types
]

class Interface:
    signals: list[str] = []
    optional: list[str] = []

    def __init__(self, handle: SimHandleBase = top, pattern: str = "", **kwargs):
        self._handle = handle
        self._explicit_kwargs = kwargs
        self._pattern = pattern

        for sig in (self.signals + self.optional):
            # 1. Check explicit kwargs first (highest priority)
            hdl = self._explicit_kwargs.get(sig)

            # 2. If not explicit, try to derive from pattern
            if hdl is None and self._pattern:
                hdl = self._derive_from_pattern(sig)

            # 3. Assign or Raise
            if hdl is not None:
                setattr(self, sig, hdl)
            elif sig in self.signals:
                raise AttributeError(f"Required signal '{sig}' not found via pattern '{pattern}' or kwargs.")
            else:
                setattr(self, sig, None)

    def _derive_from_pattern(self, sig_name):
        # Case A: Regex (contains parentheses)
        if "(" in self._pattern and ")" in self._pattern:
            regex_compiled = re.compile(self._pattern)
            for hdl_obj in self._handle:
                match = regex_compiled.fullmatch(hdl_obj._name)
                if match and sig_name in match.groups():
                    return hdl_obj

        # Case B: Glob (contains asterisk)
        elif "*" in self._pattern:
            target_name = self._pattern.replace("*", sig_name)
            if hasattr(self._handle, target_name):
                return getattr(self._handle, target_name)

        # Case C: Simple Prefix (fallback)
        else:
            target_name = f"{self._pattern}{sig_name}"
            if hasattr(self._handle, target_name):
                return getattr(self._handle, target_name)

        return None

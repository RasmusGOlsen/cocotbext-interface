from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeAlias, Union, cast

try:
    from cocotb import top
except ImportError:
    top = None

from cocotb.handle import HierarchyObject

if TYPE_CHECKING:
    from cocotb.handle import (
    ArrayObject,
    IntegerObject,
    LogicArrayObject,
    LogicObject,
    RealObject,
    )
    Signal: TypeAlias = Union[
        LogicObject,
        LogicArrayObject,
        ArrayObject,
        IntegerObject,
        RealObject
    ]

from .utils import is_match


class ModportView:
    def __init__(self, interface: 'Interface', name: str, inputs: list[str], outputs: list[str]):
        self._name = name
        for sig in (inputs + outputs):
            setattr(self, sig, getattr(interface, sig, None))


class ClockingBlock(ModportView):
    def __init__(self, interface: 'Interface', name: str, clock: str, inputs: list[str], outputs: list[str]):
        super().__init__(interface, name, inputs, outputs)
        self._clock = getattr(interface, clock)


def modport(cls):
    cls._is_modport = True
    return cls


def clocking(cls):
    cls._is_clocking = True
    return cls


class Interface:
    signals: list[str] = []
    optional: list[str] = []

    def __init__(
        self,
        component: HierarchyObject = cast(HierarchyObject, top),
        pattern: str = "",
        idx: int | None = None,
        **kwargs
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        # Use cocotb.top if no component provided and top is available
        if component is None and top is not None:
            component = cast(HierarchyObject, top)
        self._component = component
        self._idx = idx
        self._discover_and_map_signals(pattern, kwargs)
        self._create_modports_and_clocking()

    def _create_modports_and_clocking(self):
        for name in dir(self.__class__):
            attr = getattr(self.__class__, name)
            if isinstance(attr, type):
                if getattr(attr, "_is_modport", False):
                    view = ModportView(self, name, getattr(attr, "inputs", []), getattr(attr, "outputs", []))
                    setattr(self, name, view)
                elif getattr(attr, "_is_clocking", False):
                    clk_name = getattr(attr, "clock", [None])[0]
                    view = ClockingBlock(self, name, clk_name, getattr(attr, "inputs", []), getattr(attr, "outputs", []))
                    setattr(self, name, view)

    def _apply_indexing(self, handle: Any) -> Any:
        """Applies the index if the handle is an array and self._idx is set."""
        if self._idx is not None and handle is not None:
            try:
                # In cocotb, ArrayObject (unpacked) supports integer indexing
                return handle[self._idx]
            except (IndexError, TypeError):
                # Fallback: if it's not indexable, return the original handle
                return handle
        return handle

    def _discover_and_map_signals(self, pattern: str, kwargs: dict[str, Any]):
        self._explicit_kwargs = kwargs
        self._pattern = pattern

        for name in (self.signals + self.optional):
            # 1. Check if signal is explicit connected
            handle = self._explicit_kwargs.get(name)
            if handle is not None:
                setattr(self, name, handle)
                continue

            # 2. Check if pattern matches name
            if self._pattern:
                handle = self._derive_from_pattern(name)
                if handle is not None:
                    setattr(self, name, handle)
                    continue

            # 3. Signal not found check if its mandatory
            if name in self.signals:
                raise AttributeError(f"Required signal '{name}' not found.")


    def _derive_from_pattern(self, sig_name: str) -> Signal | None:
        """Derive a signal handle from the pattern and signal name."""
        # Construct the target name from the pattern
        if "%" in self._pattern:
            target_name = self._pattern.replace("%", sig_name)
        else:
            raise ValueError(f"Missing signal wildcard '%' in pattern '{self._pattern}'")

        # Iterate signals in handle(default top)
        for handle in self._component:
            if is_match(target_name, handle._name):
                return self._apply_indexing(handle)

    def __str__(self) -> str:
        # Build a readable representation of available signals and their types.
        parts: list[str] = []
        for name in (self.signals + self.optional):
            parts.append(f"{name}: {getattr(self, name)}")
        return ", ".join(parts)

    def __repr__(self) -> str:
        parts: list[str] = []
        for name in (self.signals + self.optional):
            parts.append(f"<{name}={repr(getattr(self, name))}>")
        string = ", ".join(parts)
        return f"<{self.__class__.__name__} {string}>"

from __future__ import annotations

import inspect
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
    def __init__(
        self,
        interface: Interface,
        name: str,
        inputs: list[str],
        outputs: list[str],
        inouts: list[str],
        callables: list[str]
    ):
        self._name = name
        self._interface = interface

        # 1. Map Signals
        for sig in (inputs + outputs + inouts):
            setattr(self, sig, getattr(interface, sig, None))

        # 2. Map Callables (Methods from the parent interface)
        for func_name in callables:
            if hasattr(interface, func_name):
                # We fetch the method already bound to the interface instance
                method = getattr(interface, func_name)
                setattr(self, func_name, method)
            else:
                raise AttributeError(f"Method '{func_name}' listed in modport '{name}' "
                                   f"was not found in {interface.__class__.__name__}")


class ClockingBlock(ModportView):
    def __init__(
        self,
        interface: Interface,
        name: str,
        clock: str,
        inputs: list[str],
        outputs: list[str],
        inouts: list[str]
    ):
        super().__init__(interface, name, inputs, outputs, inouts, list())
        self._clock = getattr(interface, clock)

    # async def wait(self, cycles: int = 1):
    #     """Wait for N rising edges of the clock."""
    #     for _ in range(cycles):
    #         await RisingEdge(self._clock_hdl)


def modport(count: int | type | None = None):
    """
    Decorator to define a modport.
    Usage: @modport or @modport(8)
    """
    def decorator(cls):
        cls._is_modport = True
        # If count is a class (used as @modport), default to 1 instance.
        # If count is an int (used as @modport(8)), use that value.
        cls._instance_count = count if isinstance(count, int) else 1

        if not hasattr(cls, "inputs"): cls.inputs = []
        if not hasattr(cls, "outputs"): cls.outputs = []
        if not hasattr(cls, "callables"): cls.callables = []
        return cls

    # Handle the @modport (no parens) case
    if inspect.isclass(count):
        return decorator(count)
    return decorator


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
        self._kwargs = kwargs

        self._discover_and_map_signals(pattern, kwargs)
        self._create_modports()
        self._create_clocking()


    def _create_modports(self):
        for name in dir(self.__class__):
                    attr = getattr(self.__class__, name)
                    if isinstance(attr, type) and getattr(attr, "_is_modport", False):
                        # Retrieve the value from the decorator metadata
                        class_default = getattr(attr, "_instance_count", 1)
                        # Check kwargs for override, using class_default as the fallback
                        count = self._kwargs.get(f"{name}_count", class_default)

                        # Logic to create either a single view or a list of views
                        def create_view(instance_idx=None):
                            # We pass instance_idx to ModportView in case you want
                            # sub-indexing logic inside the modport signals later
                            return ModportView(
                                interface=self,
                                name=f"{name}_{instance_idx}" if instance_idx is not None else name,
                                inputs=getattr(attr, "inputs", []),
                                outputs=getattr(attr, "outputs", []),
                                inouts=getattr(attr, "inouts", []),
                                callables=getattr(attr, "callables", [])
                            )

                        if count > 1:
                            # Create a list of modports: bus.slave[0], bus.slave[1]...
                            setattr(self, name, [create_view(i) for i in range(count)])
                        else:
                            setattr(self, name, create_view())

    def _create_clocking(self):
        for name in dir(self.__class__):
            attr = getattr(self.__class__, name)
            if isinstance(attr, type):
                if getattr(attr, "_is_clocking", False):
                    clk_name = getattr(attr, "clock", [None])[0]
                    view = ClockingBlock(
                        interface=self,
                        name=name,
                        clock=clk_name,
                        inputs=getattr(attr, "inputs", []),
                        outputs=getattr(attr, "outputs", []),
                        inouts=getattr(attr, "inouts", []))
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

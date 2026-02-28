from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, TypeAlias, Union, cast, Type

try:
    from cocotb import top
except ImportError:
    top = None

from cocotb.handle import HierarchyObject
from cocotb.triggers import RisingEdge, ReadOnly, Timer, FallingEdge
import cocotb

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

from .utils import is_match, ReadOnlyManager


class ClockedSignal:
    def __init__(
        self,
        handle: Signal,
        clock_handle: Signal,
        edge_type: Type,
        input_skew: Any = None,
        output_skew: Any = None
    ):
        self._handle = handle
        self._clk = clock_handle
        self._edge_type = edge_type
        self._input_skew = input_skew
        self._output_skew = output_skew

    @property
    def value(self):
        """Getter for immediate (potentially unsafe) access."""
        return self._handle.value

    @value.setter
    def value(self, val):
        """
        Seamless Non-blocking Drive.
        Usage: cb.sig.value = 1
        """
        cocotb.start_soon(self._drive_on_edge(val))

    async def _drive_on_edge(self, val):
        await self._edge_type(self._clk)
        if self._output_skew is not None:
            await self._output_skew
        self._handle.value = val

    async def capture(self):
        """
        Synchronized Sample (Input Skew).
        Uses a shared ReadOnly manager to prevent scheduler errors.
        """
        if self._input_skew is not None:
            await self._input_skew
        else:
            # Instead of awaiting ReadOnly directly, we wait for the manager
            await ReadOnlyManager.wait()

        return self._handle.value

    def __getattr__(self, name):
        """Forward other cocotb handle methods (like ._name, etc)"""
        return getattr(self._handle, name)


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
        clock_name: str,
        edge_type: Type,
        input_skew: Any,
        output_skew: Any,
        inputs: list[str],
        outputs: list[str],
        inouts: list[str]
    ):
        # We don't use ModportView's signal mapping because we want to wrap them
        super().__init__(interface, name, [], [], [], [])

        self._clk_handle = getattr(interface, clock_name)
        self._edge_type = edge_type

        # Prepare triggers (handles both Timer objects and Trigger classes like FallingEdge)
        def prepare_trigger(trigger_val):
            if inspect.isclass(trigger_val):
                return trigger_val(self._clk_handle)
            return trigger_val

        in_trig = prepare_trigger(input_skew)
        out_trig = prepare_trigger(output_skew)

        # Wrap every signal in a ClockedSignal object
        for sig_name in (inputs + outputs + inouts):
            raw_handle = getattr(interface, sig_name)
            setattr(self, sig_name, ClockedSignal(
                raw_handle,
                self._clk_handle,
                self._edge_type,
                in_trig,
                out_trig
            ))

    async def wait(self, cycles: int = 1):
        """Equivalent to ##N"""
        for _ in range(cycles):
            await self._edge_type(self._clk_handle)


def modport(count: int | type | None = None, clocking: str | None = None):
    """
    Decorator to define a modport.
    Usage: @modport(clocking="cb") or @modport(8, clocking="cb")
    """
    def decorator(cls):
        cls._is_modport = True
        cls._instance_count = count if isinstance(count, int) else 1
        cls._clocking_name = clocking

        if not hasattr(cls, "inputs"):
            cls.inputs = []
        if not hasattr(cls, "outputs"):
            cls.outputs = []
        if not hasattr(cls, "inouts"):
            cls.inouts = []
        if not hasattr(cls, "callables"):
            cls.callables = []
        return cls

    if inspect.isclass(count):
        return decorator(count)
    return decorator


def clocking(clock: str, edge=RisingEdge, input=None, output=None):
    """
    Decorator to define a clocking block.
    Usage: @clocking(clock="clk", edge=RisingEdge, input=Timer(1, 'ns'))
    """
    def decorator(cls):
        cls._is_clocking = True
        cls._clock_name = clock
        cls._edge_type = edge
        cls._input_skew = input
        cls._output_skew = output

        if not hasattr(cls, "inputs"):
            cls.inputs = []
        if not hasattr(cls, "outputs"):
            cls.outputs = []
        if not hasattr(cls, "inouts"):
            cls.inouts = []
        return cls
    return decorator


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
        if component is None and top is not None:
            component = cast(HierarchyObject, top)
        self._component = component
        self._idx = idx
        self._kwargs = kwargs

        self._discover_and_map_signals(pattern, kwargs)
        # Sequence matters: create clocking first so modports can link to them
        self._create_clocking()
        self._create_modports()

    def _create_clocking(self):
        for name in dir(self.__class__):
            attr = getattr(self.__class__, name)
            if isinstance(attr, type) and getattr(attr, "_is_clocking", False):
                cb_inst = ClockingBlock(
                    interface=self,
                    name=name,
                    clock_name=attr._clock_name,
                    edge_type=attr._edge_type,
                    input_skew=attr._input_skew,
                    output_skew=attr._output_skew,
                    inputs=getattr(attr, "inputs", []),
                    outputs=getattr(attr, "outputs", []),
                    inouts=getattr(attr, "inouts", [])
                )
                setattr(self, name, cb_inst)

    def _create_modports(self):
        for name in dir(self.__class__):
            attr = getattr(self.__class__, name)
            if isinstance(attr, type) and getattr(attr, "_is_modport", False):
                cb_instance = getattr(self, attr._clocking_name, None) if attr._clocking_name else None

                def create_view(instance_idx=None):
                    view = ModportView(
                        interface=self,
                        name=f"{name}_{instance_idx}" if instance_idx is not None else name,
                        inputs=getattr(attr, "inputs", []),
                        outputs=getattr(attr, "outputs", []),
                        inouts=getattr(attr, "inouts", []),
                        callables=getattr(attr, "callables", [])
                    )
                    if cb_instance:
                        setattr(view, "cb", cb_instance)
                    return view

                count = self._kwargs.get(f"{name}_count", attr._instance_count)
                if count > 1:
                    setattr(self, name, [create_view(i) for i in range(count)])
                else:
                    setattr(self, name, create_view())

    def _apply_indexing(self, handle: Any) -> Any:
        if self._idx is not None and handle is not None:
            try:
                return handle[self._idx]
            except (IndexError, TypeError):
                return handle
        return handle

    def _discover_and_map_signals(self, pattern: str, kwargs: dict[str, Any]):
        self._explicit_kwargs = kwargs
        self._pattern = pattern

        for name in (self.signals + self.optional):
            handle = self._explicit_kwargs.get(name)
            if handle is not None:
                setattr(self, name, handle)
                continue

            if self._pattern:
                handle = self._derive_from_pattern(name)
                if handle is not None:
                    setattr(self, name, handle)
                    continue

            if name in self.signals:
                raise AttributeError(f"Required signal '{name}' not found.")

    def _derive_from_pattern(self, sig_name: str) -> Signal | None:
        if "%" in self._pattern:
            target_name = self._pattern.replace("%", sig_name)
        else:
            raise ValueError(f"Missing signal wildcard '%' in pattern '{self._pattern}'")

        for handle in self._component:
            if is_match(target_name, handle._name):
                return self._apply_indexing(handle)

    def __str__(self) -> str:
        parts: list[str] = []
        for name in (self.signals + self.optional):
            parts.append(f"{name}: {getattr(self, name, None)}")
        return ", ".join(parts)

    def __repr__(self) -> str:
        parts: list[str] = []
        for name in (self.signals + self.optional):
            parts.append(f"<{name}={repr(getattr(self, name, None))}>")
        string = ", ".join(parts)
        return f"<{self.__class__.__name__} {string}>"

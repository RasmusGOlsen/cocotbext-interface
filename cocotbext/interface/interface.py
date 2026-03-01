from __future__ import annotations
import inspect
import logging
from typing import TYPE_CHECKING, Any, TypeAlias, Union, Type, TypeVar, Generic, get_type_hints, Callable

import cocotb
from cocotb.handle import HierarchyObject
from cocotb.triggers import RisingEdge, ReadOnly, Timer

from .utils import ReadOnlyManager

if TYPE_CHECKING:
    from cocotb.handle import LogicArrayObject, LogicObject, ArrayObject
    Signal: TypeAlias = Union[LogicObject, LogicArrayObject, ArrayObject]

# --- 1. Directional Markers (Type Hints Only) ---
T = TypeVar("T")
class Input(Generic[T]): ...
class Output(Generic[T]): ...
class InOut(Generic[T]): ...
class Import(Generic[T]): ...

# --- 2. Synchronous Signal Wrapper ---
class ClockedSignal:
    def __init__(
        self,
        handle: Signal,
        clock: Signal,
        edge: Type,
        is_input,
        input_skew: Any = None,
        output_skew: Any = None
    ):
        self._handle = handle
        self._clk = clock
        self._edge = edge
        self._is_input = is_input
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

# --- 3. Functional Views ---
class ClockingBlock:
    def __init__(self, parent: Interface, cb_cls: type):
        self._parent = parent
        # The decorator stores the string name of the clock, e.g., "clk"
        self._clk_name = getattr(cb_cls, "_clock_name", None)
        self._clk = getattr(parent, self._clk_name, None)
        self._edge = getattr(cb_cls, "_edge_type", RisingEdge)

        if self._clk is None:
            raise AttributeError(f"ClockingBlock '{cb_cls.__name__}' references missing clock '{self._clk_name}'")

        # Resolve type hints for the nested class
        hints = get_type_hints(cb_cls, globalns=globals())

        for name, _type in hints.items():
            # Get the actual signal handle from the parent Interface
            raw_handle = getattr(parent, name, None)

            # 1. Handle Optional Signals: If raw_handle is None, keep it None
            if raw_handle is None:
                setattr(self, name, None)
                continue

            # 2. Determine Directionality: Use the Generic 'origin'
            origin = getattr(_type, "__origin__", None)
            is_input = (origin is Input)

            # 3. Wrap in ClockedSignal: This provides the .value setter and .capture() method
            setattr(self, name, ClockedSignal(
                handle=raw_handle,
                clock=self._clk,
                edge=self._edge,
                is_input=is_input
            ))

    async def wait(self, cycles: int = 1):
        """Standard ##N delay logic."""
        for _ in range(cycles):
            await self._edge(self._clk)

    def __str__(self) -> str:
        edge_name = self._edge.__name__ if hasattr(self._edge, "__name__") else str(self._edge)
        lines = [f"ClockingBlock: {self._parent.__class__.__name__}.cb (Ref: {self._clk_name} @ {edge_name})"]

        for name, attr in self.__dict__.items():
            if not name.startswith("_"):
                if attr is None:
                    lines.append(f"  {name:15} : None (Optional)")
                elif isinstance(attr, ClockedSignal):
                    try:
                        val = attr.value # Immediate sample
                    except:
                        val = "X"
                    lines.append(f"  {name:15} : {val}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<ClockingBlock {self._parent.__class__.__name__}.cb @ {self._clk_name}>"

class ModportView:
    def __init__(self, parent: Interface, mp_cls: type, index: int | None = None):
        self._parent = parent
        self._index = index

        # We use globalns=globals() to ensure 'Import' and 'Callable' types
        # are resolvable inside the nested class scope.
        hints = get_type_hints(mp_cls, globalns=globals())

        for name, _type in hints.items():
            origin = getattr(_type, "__origin__", None)

            if origin is Import:
                # 1. Map Method from parent Interface
                if hasattr(parent, name):
                    setattr(self, name, getattr(parent, name))
                else:
                    raise AttributeError(
                        f"Modport '{mp_cls.__name__}' expected imported method '{name}', "
                        f"but it was not found in {parent.__class__.__name__}"
                    )

            else:
                # 2. Map Signal Handle
                # First, get the handle from the parent (might be a Signal or None)
                raw_handle = getattr(parent, name, None)

                if raw_handle is None:
                    # Signal was optional and not found in RTL
                    setattr(self, name, None)
                    continue

                # If the handle exists and we have an index (Interface Array/Pattern)
                if index is not None:
                    try:
                        # Attempt to index (e.g., dut.data_in[0])
                        setattr(self, name, raw_handle[index])
                    except (IndexError, TypeError):
                        # Fallback if the signal isn't an array but the interface is
                        setattr(self, name, raw_handle)
                else:
                    setattr(self, name, raw_handle)

        # 3. Link to Clocking Block (Optional)
        cb_name = getattr(mp_cls, "_clocking_name", None)
        if cb_name:
            if hasattr(parent, cb_name):
                setattr(self, "cb", getattr(parent, cb_name))
            else:
                logging.warning(f"Modport {mp_cls.__name__} refers to missing clocking block '{cb_name}'")

    def __str__(self) -> str:
        idx_str = f"[{self._index}]" if self._index is not None else ""
        lines = [f"Modport View: {self._parent.__class__.__name__}.{self.__class__.__name__}{idx_str}"]

        # Filter internal attributes for a clean print
        for name in self.__dict__:
            if not name.startswith("_") and name != "cb":
                val = getattr(self, name)
                # Show signal value or 'None' for optional signals
                display_val = val.value if hasattr(val, "value") else val
                lines.append(f"  {name:15} : {display_val}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        idx_info = f"[{self._index}]" if self._index is not None else ""
        return f"<ModportView {self._parent.__class__.__name__}{idx_info}>"

# --- 4. Decorators ---
def clocking(clock: str, edge=RisingEdge):
    def decorator(cls):
        cls._is_clocking, cls._clock_name, cls._edge_type = True, clock, edge
        return cls
    return decorator

def modport(clocking: str = None, callables: list[str] = None, count: int = 1):
    def decorator(cls):
        cls._is_modport = True
        cls._clocking_name = clocking
        cls._callables = callables or []
        cls._count = count
        return cls
    return decorator

# --- 5. The Base Interface ---
class Interface:
    def __init__(self, signals: dict[str, Signal | None], index: int | None = None):
        self._index = index
        self._signals_list = list(signals.keys())
        for name, handle in signals.items():
            setattr(self, name, handle)
        self._create_clocking()
        self._create_modports()

    @classmethod
    def _get_requirements(cls) -> dict[str, Any]:
        """Returns a mapping of signal names to their default values (if any)."""
        # Annotations tell us what SHOULD exist
        hints = get_type_hints(cls, globalns=globals())
        # Class attributes tell us if there is a DEFAULT (like None)
        requirements = {}
        for name in hints:
            if name.startswith("_"): continue
            # If the class has the attribute, it's the default value.
            # Otherwise, it's 'Required' (we use a sentinel).
            requirements[name] = getattr(cls, name, "...REQUIRED...")
        return requirements

    @classmethod
    def from_signal(cls, **kwargs):
        return cls.from_pattern(parent=None, pattern=None, idx=None)

    @classmethod
    def from_entity(cls, handle: HierarchyObject):
        return cls.from_pattern(parent=handle, pattern="%", idx=None)

    @classmethod
    def from_pattern(cls, parent: HierarchyObject | None, pattern: str | None = "%", idx: int | None = None, **kwargs):
        requirements = cls._get_requirements()
        signals = {}
        handle = None
        for name, default in requirements.items():
            target = name
            # 1. Kwarg override
            if name in kwargs:
                signals[name] = kwargs[name]
                continue

            # 2. Pattern search
            if pattern is not None:
                target = pattern.replace("%", name)
                handle = getattr(parent, target, None)

            # Apply indexing
            if handle is not None and idx is not None:
                try:
                    handle = handle[idx]
                except (IndexError, TypeError):
                    pass

            # 3. Validation
            if handle is None:
                if default == "...REQUIRED...":
                    raise AttributeError(f"Required signal '{name}' (target: {target}) not found.")
                signals[name] = default
            else:
                signals[name] = handle

        return cls(signals, index=idx)

    def _create_clocking(self):
        for name, attr in inspect.getmembers(self.__class__, inspect.isclass):
            if getattr(attr, "_is_clocking", False):
                setattr(self, name, ClockingBlock(self, attr))

    def _create_modports(self):
        for name, attr in inspect.getmembers(self.__class__, inspect.isclass):
            if getattr(attr, "_is_modport", False):
                count = getattr(attr, "_count", 1)
                if count > 1:
                    setattr(self, name, [ModportView(self, attr, i) for i in range(count)])
                else:
                    setattr(self, name, ModportView(self, attr))

    def __str__(self) -> str:
        """User-friendly multi-line state summary."""
        lines = [f"Interface: {self.__class__.__name__}"]
        if hasattr(self, "_index") and self._index is not None:
            lines[0] += f" [Index: {self._index}]"

        for name in self._signals_list:
            handle = getattr(self, name)
            val = "N/A"
            try:
                val = handle.value
            except Exception:
                pass
            lines.append(f"  {name:15} : {val}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Developer-friendly one-liner."""
        sig_count = len(self._signals_list)
        idx_info = f"idx={self._index}" if getattr(self, "_index", None) is not None else "top"
        return f"<{self.__class__.__name__} ({idx_info}, {sig_count} signals)>"

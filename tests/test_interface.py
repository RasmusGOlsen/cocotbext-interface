"""
Pytest tests for the refactored Interface class.

Tests cover the new API using type hints, factory methods, and nested class decorators.
"""

import asyncio
import inspect
from typing import Iterator

import pytest

from cocotbext.interface.interface import (
    Interface,
    ClockingBlock,
    ModportView,
    ClockedSignal,
    modport,
    clocking,
    Input,
    Output,
    InOut,
    Import,
)
from cocotb.triggers import RisingEdge, FallingEdge


# ============================================================================
# Global Mocking for cocotb scheduler
# ============================================================================

@pytest.fixture(autouse=True)
def mock_cocotb_scheduler(monkeypatch):
    """Auto-patch cocotb.start_soon to avoid scheduler errors in tests.

    Properly closes coroutines to avoid RuntimeWarning about unawaited coroutines.
    """
    def mock_start_soon(coro):
        # If it's a coroutine, close it properly to avoid warning
        if inspect.iscoroutine(coro):
            coro.close()

    monkeypatch.setattr('cocotb.start_soon', mock_start_soon)


# ============================================================================
# Mock Fixtures for cocotb dependencies
# ============================================================================

class MockHandle:
    """Mock for cocotb handle with a name and optional array support."""
    def __init__(self, name: str, is_array: bool = False, array_size: int = 4):
        self._name = name
        self._is_array = is_array
        self._array_size = array_size
        self._value = 0

    @property
    def value(self):
        """Simple property to mimic cocotb handle read/write."""
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

    def __getitem__(self, index: int):
        """Support array indexing."""
        if not self._is_array:
            raise TypeError(f"Cannot index non-array handle {self._name}")
        if not (0 <= index < self._array_size):
            raise IndexError(f"Index {index} out of range for array {self._name}")
        return MockHandle(f"{self._name}[{index}]", is_array=False)

    def __iter__(self):
        """Support iteration (for HierarchyObject behavior)."""
        return iter([])

    def __repr__(self):
        return f"MockHandle('{self._name}')"


class MockHierarchy:
    """Mock for HierarchyObject that acts as a container of handles."""
    def __init__(self):
        self._children: dict[str, MockHandle] = {}

    def add_signal(self, name: str, is_array: bool = False, array_size: int = 4):
        """Add a signal to the hierarchy."""
        self._children[name] = MockHandle(name, is_array=is_array, array_size=array_size)

    def __iter__(self) -> Iterator[MockHandle]:
        """Iterate over all signals in the hierarchy."""
        return iter(self._children.values())

    def __getattr__(self, name: str):
        """Allow accessing signals as attributes (dut.clk)."""
        if name.startswith("_"):
            return super().__getattribute__(name)
        return self._children.get(name)


@pytest.fixture
def mock_hierarchy():
    """Fixture providing a mock HierarchyObject."""
    hierarchy = MockHierarchy()
    hierarchy.add_signal("clk")
    hierarchy.add_signal("rst")
    hierarchy.add_signal("data")  # Generic data signal
    hierarchy.add_signal("data_in")
    hierarchy.add_signal("data_out")
    hierarchy.add_signal("valid")
    hierarchy.add_signal("ready")
    hierarchy.add_signal("bus_data", is_array=True, array_size=8)
    return hierarchy


# ============================================================================
# Test Interface Definitions (using the new API)
# ============================================================================

class SimpleInterface(Interface):
    """Simple test interface with required and optional signals."""
    clk: Output[object]
    rst: Output[object]
    data_in: Input[object]
    status: Input[object] = None  # Optional signal


class InterfaceWithModport(Interface):
    """Test interface with modport."""
    clk: Output[object]
    data: Input[object]

    @modport()
    class master:
        data: Output[object]
        clk: Input[object]


class InterfaceWithClocking(Interface):
    """Test interface with clocking block."""
    clk: Output[object]
    data: Input[object]
    valid: Input[object]

    @clocking(clock="clk")
    class CK:
        data: Input[object]
        valid: Input[object]


class InterfaceWithMultipleModports(Interface):
    """Test interface with multiple modport instances."""
    clk: Output[object]
    data_in: Input[object]
    data_out: Output[object]

    @modport(count=2)
    class port:
        data_in: Input[object]
        data_out: Output[object]


class InterfaceWithModportAndClocking(Interface):
    """Test interface with modport linked to clocking block."""
    clk: Output[object]
    data: Input[object]
    valid: Output[object]

    @clocking(clock="clk")
    class CK:
        data: Input[object]
        valid: Output[object]

    @modport(clocking="CK")
    class master:
        data: Output[object]
        valid: Input[object]


# ============================================================================
# Tests: Factory Methods (from_entity, from_pattern)
# ============================================================================

class TestFactoryMethods:
    """Test Interface factory methods."""

    def test_from_entity_basic(self, mock_hierarchy):
        """Test creating Interface from a hierarchy handle."""
        iface = SimpleInterface.from_entity(mock_hierarchy)

        assert iface.clk._name == "clk"
        assert iface.rst._name == "rst"
        assert iface.data_in._name == "data_in"

    def test_from_entity_optional_signal(self, mock_hierarchy):
        """Test optional signal behavior with from_entity."""
        # Don't add 'status', so it should default to None
        iface = SimpleInterface.from_entity(mock_hierarchy)
        assert iface.status is None

    def test_from_entity_missing_required_signal(self, mock_hierarchy):
        """Test that missing required signal raises AttributeError."""
        # Create a minimal hierarchy without a required signal
        minimal_h = MockHierarchy()
        minimal_h.add_signal("clk")
        minimal_h.add_signal("rst")
        # Missing 'data_in', so should raise

        with pytest.raises(AttributeError, match="Required signal"):
            SimpleInterface.from_entity(minimal_h)

    def test_from_pattern_basic(self, mock_hierarchy):
        """Test creating Interface from a pattern."""
        iface = SimpleInterface.from_pattern(mock_hierarchy, pattern="%")

        assert iface.clk._name == "clk"
        assert iface.rst._name == "rst"
        assert iface.data_in._name == "data_in"

    def test_from_pattern_with_kwargs_override(self, mock_hierarchy):
        """Test that kwargs override pattern matching."""
        custom_clk = MockHandle("custom_clk")
        iface = SimpleInterface.from_pattern(
            mock_hierarchy,
            pattern="%",
            clk=custom_clk
        )

        assert iface.clk is custom_clk
        assert iface.rst._name == "rst"

    def test_from_pattern_with_index(self, mock_hierarchy):
        """Test pattern matching with array indexing."""
        iface = SimpleInterface.from_pattern(
            mock_hierarchy,
            pattern="%",
            idx=2
        )

        # All signals should be indexed (unless not arrays)
        assert iface.clk._name == "clk"


# ============================================================================
# Tests: Basic Signal Access
# ============================================================================

class TestBasicSignalAccess:
    """Test basic signal access via interface."""

    def test_signal_value_read(self, mock_hierarchy):
        """Test reading signal value."""
        mock_hierarchy._children["clk"]._value = 42
        iface = SimpleInterface.from_entity(mock_hierarchy)
        assert iface.clk.value == 42

    def test_signal_value_write(self, mock_hierarchy):
        """Test writing signal value."""
        iface = SimpleInterface.from_entity(mock_hierarchy)
        iface.clk.value = 99
        assert mock_hierarchy._children["clk"].value == 99

    def test_str_representation(self, mock_hierarchy):
        """Test __str__ method."""
        iface = SimpleInterface.from_entity(mock_hierarchy)
        str_repr = str(iface)
        assert "SimpleInterface" in str_repr
        assert "clk" in str_repr

    def test_repr_representation(self, mock_hierarchy):
        """Test __repr__ method."""
        iface = SimpleInterface.from_entity(mock_hierarchy)
        repr_str = repr(iface)
        assert "SimpleInterface" in repr_str


# ============================================================================
# Tests: Modports
# ============================================================================

class TestModports:
    """Test modport creation and functionality."""

    def test_modport_creation(self, mock_hierarchy):
        """Test that modports are created and accessible."""
        iface = InterfaceWithModport.from_entity(mock_hierarchy)

        assert hasattr(iface, "master")
        master = iface.master
        assert isinstance(master, ModportView)

    def test_modport_contains_signals(self, mock_hierarchy):
        """Test that modport contains the mapped signals."""
        iface = InterfaceWithModport.from_entity(mock_hierarchy)
        master = iface.master

        assert hasattr(master, "data")
        assert hasattr(master, "clk")

    def test_modport_repr(self, mock_hierarchy):
        """Test modport __repr__."""
        iface = InterfaceWithModport.from_entity(mock_hierarchy)
        master = iface.master
        repr_str = repr(master)
        assert "ModportView" in repr_str

    def test_modport_multiple_instances(self, mock_hierarchy):
        """Test modport with count > 1 creates a list."""
        iface = InterfaceWithMultipleModports.from_entity(mock_hierarchy)

        # 'port' should be a list of 2 ModportView instances
        port_list = iface.port
        assert isinstance(port_list, list)
        assert len(port_list) == 2
        assert all(isinstance(p, ModportView) for p in port_list)

    def test_modport_multiple_instances_access(self, mock_hierarchy):
        """Test accessing individual modport instances."""
        iface = InterfaceWithMultipleModports.from_entity(mock_hierarchy)

        port0 = iface.port[0]
        port1 = iface.port[1]

        assert hasattr(port0, "data_in")
        assert hasattr(port1, "data_in")
        # Both should reference the same underlying signal
        assert port0.data_in._name == mock_hierarchy._children["data_in"]._name
        assert port1.data_in._name == mock_hierarchy._children["data_in"]._name


# ============================================================================
# Tests: Clocking Blocks
# ============================================================================

class TestClockingBlocks:
    """Test clocking block creation and functionality."""

    def test_clocking_block_creation(self, mock_hierarchy):
        """Test that clocking blocks are created correctly."""
        iface = InterfaceWithClocking.from_entity(mock_hierarchy)

        assert hasattr(iface, "CK")
        ck = iface.CK
        assert isinstance(ck, ClockingBlock)

    def test_clocking_block_has_clock(self, mock_hierarchy):
        """Test that clocking block has reference to clock."""
        iface = InterfaceWithClocking.from_entity(mock_hierarchy)
        ck = iface.CK

        assert ck._clk._name == "clk"
        assert ck._clk_name == "clk"

    def test_clocking_block_signals_wrapped(self, mock_hierarchy):
        """Test that clocking block signals are wrapped in ClockedSignal."""
        iface = InterfaceWithClocking.from_entity(mock_hierarchy)
        ck = iface.CK

        assert isinstance(ck.data, ClockedSignal)
        assert isinstance(ck.valid, ClockedSignal)

    def test_clocking_block_wait(self, mock_hierarchy):
        """Test clocking block wait() method."""
        iface = InterfaceWithClocking.from_entity(mock_hierarchy)
        ck = iface.CK

        # Just ensure the method exists and has correct signature
        assert hasattr(ck, "wait")
        assert callable(ck.wait)

    def test_clocking_block_repr(self, mock_hierarchy):
        """Test clocking block __repr__."""
        iface = InterfaceWithClocking.from_entity(mock_hierarchy)
        ck = iface.CK

        repr_str = repr(ck)
        assert "ClockingBlock" in repr_str
        assert "clk" in repr_str


# ============================================================================
# Tests: ClockedSignal
# ============================================================================

class TestClockedSignal:
    """Test ClockedSignal synchronous signal wrapper."""

    def test_clocked_signal_value_getter(self, mock_hierarchy):
        """Test that ClockedSignal can read value."""
        mock_hierarchy._children["clk"]._value = 42
        cs = ClockedSignal(
            handle=mock_hierarchy._children["clk"],
            clock=mock_hierarchy._children["clk"],
            edge=RisingEdge,
            is_input=False
        )
        assert cs.value == 42

    def test_clocked_signal_value_setter(self, mock_hierarchy):
        """Test that ClockedSignal can set value (schedules non-blocking drive)."""
        cs = ClockedSignal(
            handle=mock_hierarchy._children["clk"],
            clock=mock_hierarchy._children["clk"],
            edge=RisingEdge,
            is_input=False
        )
        # Set value (in tests, this just calls mocked cocotb.start_soon)
        cs.value = 99
        # ClockedSignal value setter doesn't directly update; it schedules it
        # Just verify no exception was raised


    def test_clocked_signal_attribute_forwarding(self, mock_hierarchy):
        """Test that ClockedSignal forwards attributes to handle."""
        cs = ClockedSignal(
            handle=mock_hierarchy._children["clk"],
            clock=mock_hierarchy._children["clk"],
            edge=RisingEdge,
            is_input=False
        )
        # _name should be forwarded to the handle
        assert cs._name == "clk"

    def test_clocked_signal_creation_with_edge(self, mock_hierarchy):
        """Test ClockedSignal with different edge types."""
        cs_rising = ClockedSignal(
            handle=mock_hierarchy._children["data_in"],
            clock=mock_hierarchy._children["clk"],
            edge=RisingEdge,
            is_input=True
        )
        assert cs_rising._edge is RisingEdge

        cs_falling = ClockedSignal(
            handle=mock_hierarchy._children["data_in"],
            clock=mock_hierarchy._children["clk"],
            edge=FallingEdge,
            is_input=True
        )
        assert cs_falling._edge is FallingEdge

    def test_clocked_signal_input_flag(self, mock_hierarchy):
        """Test ClockedSignal tracks input direction."""
        cs_input = ClockedSignal(
            handle=mock_hierarchy._children["data_in"],
            clock=mock_hierarchy._children["clk"],
            edge=RisingEdge,
            is_input=True
        )
        assert cs_input._is_input is True

        cs_output = ClockedSignal(
            handle=mock_hierarchy._children["data_out"],
            clock=mock_hierarchy._children["clk"],
            edge=RisingEdge,
            is_input=False
        )
        assert cs_output._is_input is False

    def test_clocked_signal_capture_multiple(self, mock_hierarchy):
        """Test capturing multiple signals without scheduler error."""
        async def inner():
            # Create ClockedSignals for multiple handles
            h1 = MockHandle("sig1")
            h1._value = 11
            h2 = MockHandle("sig2")
            h2._value = 22
            clk = mock_hierarchy._children["clk"]

            cs1 = ClockedSignal(h1, clk, RisingEdge, is_input=True)
            cs2 = ClockedSignal(h2, clk, RisingEdge, is_input=True)

            # Simulate sequential captures (immediate read for testing)
            val1 = cs1.value  # immediate read
            val2 = cs2.value  # immediate read

            assert val1 == 11
            assert val2 == 22

        asyncio.run(inner())


# ============================================================================
# Tests: Modport + Clocking Integration
# ============================================================================

class TestModportClockingIntegration:
    """Test modport linked to clocking block."""

    def test_modport_has_clocking_reference(self, mock_hierarchy):
        """Test that modport can reference clocking block."""
        iface = InterfaceWithModportAndClocking.from_entity(mock_hierarchy)

        master = iface.master
        assert hasattr(master, "cb")
        cb = master.cb
        assert isinstance(cb, ClockingBlock)

    def test_clocking_accessible_through_modport(self, mock_hierarchy):
        """Test clocking block functionality through modport."""
        iface = InterfaceWithModportAndClocking.from_entity(mock_hierarchy)

        master = iface.master
        cb = master.cb

        assert cb._clk_name == "clk"
        assert hasattr(cb, "data")
        assert hasattr(cb, "valid")


# ============================================================================
# Tests: Type Hints and Directionality
# ============================================================================

class TestTypeHintsAndDirectionality:
    """Test that Input/Output type hints work correctly."""

    def test_input_type_hint(self, mock_hierarchy):
        """Test Input type hint within clocking block."""
        iface = InterfaceWithClocking.from_entity(mock_hierarchy)
        ck = iface.CK

        # Clocking block signals should be wrapped based on their type hint
        data_signal = ck.data
        assert isinstance(data_signal, ClockedSignal)
        # The type hint 'Input[object]' means is_input=True
        assert data_signal._is_input is True

    def test_output_type_hint(self, mock_hierarchy):
        """Test Output type hint within modport."""
        iface = InterfaceWithModportAndClocking.from_entity(mock_hierarchy)
        master = iface.master

        # Modport signals should map
        assert hasattr(master, "data")
        # Modport maps handles as-is (not wrapped in ClockedSignal)


# ============================================================================
# Tests: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_interface_with_index(self, mock_hierarchy):
        """Test Interface with index parameter."""
        iface = SimpleInterface.from_pattern(
            mock_hierarchy,
            pattern="%",
            idx=1
        )
        assert iface._index == 1

    def test_interface_missing_clocking_reference(self, mock_hierarchy):
        """Test modport with non-existent clocking block reference."""
        # Create interface, but don't define the clocking block that modport refers to
        class BrokenInterface(Interface):
            clk: Output[object]
            data: Input[object]

            @modport(clocking="MissingClocking")
            class port:
                data: Input[object]

        # Should not raise during creation, but warn when accessing
        iface = BrokenInterface.from_entity(mock_hierarchy)
        # Just verify it doesn't crash immediately

    def test_empty_interface(self, mock_hierarchy):
        """Test interface with no signals."""
        class EmptyInterface(Interface):
            pass

        iface = EmptyInterface.from_entity(mock_hierarchy)
        assert len(iface._signals_list) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

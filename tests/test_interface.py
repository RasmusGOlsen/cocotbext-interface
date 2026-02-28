"""
Pytest tests for the Interface class.

These tests use mocks for cocotb dependencies to allow testing in pure Python
without requiring a simulator.
"""

from typing import Iterator

import pytest
import asyncio

from cocotbext.interface import Interface, clocking, modport

from cocotbext.interface.interface import (
    ClockingBlock,
    ModportView,
)

# ============================================================================
# Mock Fixtures for cocotb dependencies
# ============================================================================

class MockHandle:
    """Mock for SimHandleBase with a name and optional array support."""
    def __init__(self, name: str, is_array: bool = False, array_size: int = 4):
        self._name = name
        self._is_array = is_array
        self._array_size = array_size
        # simulate the cocotb .value property used by ClockedSignal
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
        # Return a new mock handle representing the indexed element
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

    def __iter_children__(self):
        """Support cocotb's child iteration."""
        return iter(self._children.values())


@pytest.fixture
def mock_component():
    """Fixture providing a mock HierarchyObject component."""
    hierarchy = MockHierarchy()
    # Add some typical signals
    hierarchy.add_signal("clk")
    hierarchy.add_signal("rst")
    hierarchy.add_signal("data_in")
    hierarchy.add_signal("data_out")
    hierarchy.add_signal("valid")
    hierarchy.add_signal("ready")
    hierarchy.add_signal("bus_data", is_array=True, array_size=8)
    return hierarchy


@pytest.fixture
def mock_top():
    """Fixture providing a mock for cocotb.top."""
    return MockHierarchy()


# ============================================================================
# Test Classes
# ============================================================================

class SimpleInterface(Interface):
    """Simple test interface with required and optional signals."""
    signals = ["clk", "rst", "data_in"]
    optional = ["status"]


class InterfaceWithModport(Interface):
    """Test interface with modport."""
    signals = ["clk", "data"]

    @modport
    class master:
        inputs = ["data"]
        outputs = ["clk"]


class InterfaceWithClocking(Interface):
    """Test interface with clocking block."""
    signals = ["clk", "data", "valid"]

    @clocking(clock="clk")
    class CK:
        inputs = ["data", "valid"]
        outputs = []


# ============================================================================
# Tests: Basic Signal Discovery and Mapping
# ============================================================================

class TestInterfaceSignalDiscovery:
    """Test signal discovery and mapping."""

    def test_explicit_kwargs_mapping(self, mock_component):
        """Test mapping signals via explicit kwargs."""
        clk_handle = MockHandle("clk")
        rst_handle = MockHandle("rst")

        iface = SimpleInterface(
            component=mock_component,
            clk=clk_handle,
            rst=rst_handle,
            data_in=MockHandle("data_in")
        )

        assert iface.clk is clk_handle
        assert iface.rst is rst_handle

    def test_pattern_based_discovery(self, mock_component):
        """Test signal discovery using pattern matching."""
        iface = SimpleInterface(
            component=mock_component,
            pattern="",  # No pattern, direct names
            clk=MockHandle("clk"),
            rst=MockHandle("rst"),
            data_in=MockHandle("data_in")
        )

        assert hasattr(iface, "clk")
        assert hasattr(iface, "rst")
        assert hasattr(iface, "data_in")

    def test_optional_signals_not_required(self, mock_component):
        """Test that optional signals don't raise AttributeError if missing."""
        # This should not raise even though 'status' is not provided
        iface = SimpleInterface(
            component=mock_component,
            clk=MockHandle("clk"),
            rst=MockHandle("rst"),
            data_in=MockHandle("data_in")
        )
        # status is optional, should not raise
        assert not hasattr(iface, "status") or iface.status is None

    def test_required_signal_error(self, mock_component):
        """Test that missing required signals raise AttributeError."""
        with pytest.raises(AttributeError, match="Required signal 'clk' not found"):
            SimpleInterface(component=mock_component)


# ============================================================================
# Tests: Pattern-Based Discovery
# ============================================================================

class TestPatternDiscovery:
    """Test pattern-based signal discovery."""

    def test_missing_wildcard_in_pattern_raises_error(self, mock_component):
        """Test that pattern without wildcard raises ValueError."""
        with pytest.raises(ValueError, match="Missing signal wildcard"):
            SimpleInterface(
                component=mock_component,
                pattern="fixed_name",
                clk=MockHandle("clk"),
                rst=MockHandle("rst"),
                data_in=MockHandle("data_in")
            )

    def test_explicit_kwargs_override_pattern(self, mock_component):
        """Test that explicit kwargs take precedence over pattern."""
        explicit_clk = MockHandle("explicit_clk")
        iface = SimpleInterface(
            component=mock_component,
            pattern="my_%",
            clk=explicit_clk,
            rst=MockHandle("rst"),
            data_in=MockHandle("data_in")
        )

        # Explicit kwarg should be used
        assert iface.clk is explicit_clk


# ============================================================================
# Tests: Indexing
# ============================================================================

class TestIndexing:
    """Test array indexing functionality."""

    def test_indexing_on_explicit_kwargs(self, mock_component):
        """Test array indexing on explicitly provided handles."""
        # Create a mock array that supports indexing
        array_handle = MockHandle("bus_data", is_array=True, array_size=8)
        indexed_handle = array_handle[2]  # Get indexed version

        class ArrayInterface(Interface):
            signals = ["clk", "indexed_bus"]
            optional = []

        iface = ArrayInterface(
            component=mock_component,
            idx=2,
            clk=MockHandle("clk"),
            indexed_bus=indexed_handle
        )

        # The indexed handle should be set
        assert iface.indexed_bus._name == "bus_data[2]"

    def test_indexing_on_non_array_returns_original(self, mock_component):
        """Test that indexing on non-array signals returns the original handle."""
        clk_handle = MockHandle("clk", is_array=False)

        iface = SimpleInterface(
            component=mock_component,
            idx=1,
            clk=clk_handle,
            rst=MockHandle("rst"),
            data_in=MockHandle("data_in")
        )

        # Non-array should return original
        assert iface.clk is clk_handle


# ============================================================================
# Tests: String Representations
# ============================================================================

class TestStringRepresentations:
    """Test __str__ and __repr__ methods."""

    def test_str_representation(self, mock_component):
        """Test __str__ returns all signals."""
        # Create interface with explicit signals only (no optional ones)
        class MinimalInterface(Interface):
            signals = ["clk", "rst", "data_in"]
            optional = []

        iface = MinimalInterface(
            component=mock_component,
            clk=MockHandle("clk"),
            rst=MockHandle("rst"),
            data_in=MockHandle("data_in")
        )

        str_repr = str(iface)
        assert "clk" in str_repr
        assert "rst" in str_repr
        assert "data_in" in str_repr

    def test_repr_representation(self, mock_component):
        """Test __repr__ includes class name."""
        # Create interface with explicit signals only (no optional ones)
        class MinimalInterface(Interface):
            signals = ["clk", "rst", "data_in"]
            optional = []

        iface = MinimalInterface(
            component=mock_component,
            clk=MockHandle("clk"),
            rst=MockHandle("rst"),
            data_in=MockHandle("data_in")
        )

        repr_str = repr(iface)
        assert "MinimalInterface" in repr_str
        assert "clk" in repr_str


# ============================================================================
# Tests: Modports
# ============================================================================

class TestModports:
    """Test modport creation and functionality."""

    def test_modport_creation(self, mock_component):
        """Test that modports are created correctly."""
        iface = InterfaceWithModport(
            component=mock_component,
            clk=MockHandle("clk"),
            data=MockHandle("data")
        )

        # Check that modport exists
        assert hasattr(iface, "master")
        master = iface.master
        assert isinstance(master, ModportView)

    def test_modport_contains_signals(self, mock_component):
        """Test that modport contains the specified signals."""
        data_handle = MockHandle("data")
        clk_handle = MockHandle("clk")

        iface = InterfaceWithModport(
            component=mock_component,
            clk=clk_handle,
            data=data_handle
        )

        master = iface.master
        assert master.clk is clk_handle
        assert master.data is data_handle

    def test_modport_name(self, mock_component):
        """Test that modport has correct name."""
        iface = InterfaceWithModport(
            component=mock_component,
            clk=MockHandle("clk"),
            data=MockHandle("data")
        )

        master = iface.master
        assert master._name == "master"


# ============================================================================
# Tests: Clocking Blocks
# ============================================================================

class TestClockingBlocks:
    """Test clocking block creation and functionality."""

    def test_clocking_block_creation(self, mock_component):
        """Test that clocking blocks are created correctly."""
        iface = InterfaceWithClocking(
            component=mock_component,
            clk=MockHandle("clk"),
            data=MockHandle("data"),
            valid=MockHandle("valid")
        )

        # Check that clocking block exists
        assert hasattr(iface, "CK")
        ck = iface.CK
        assert isinstance(ck, ClockingBlock)

    def test_clocking_block_has_clock(self, mock_component):
        """Test that clocking block can access the clock signal."""
        clk_handle = MockHandle("clk")
        iface = InterfaceWithClocking(
            component=mock_component,
            clk=clk_handle,
            data=MockHandle("data"),
            valid=MockHandle("valid")
        )

        ck = iface.CK
        # clock handle stored only in _clk_handle
        assert ck._clk_handle is clk_handle

    def test_clocking_block_contains_signals(self, mock_component):
        """Test that clocking block contains input/output signals."""
        data_handle = MockHandle("data")
        valid_handle = MockHandle("valid")

        iface = InterfaceWithClocking(
            component=mock_component,
            clk=MockHandle("clk"),
            data=data_handle,
            valid=valid_handle
        )

        ck = iface.CK
        # signals should be wrapped in ClockedSignal objects
        from cocotbext.interface.interface import ClockedSignal
        assert isinstance(ck.data, ClockedSignal)
        assert ck.data._handle is data_handle
        assert isinstance(ck.valid, ClockedSignal)
        assert ck.valid._handle is valid_handle


# ============================================================================
# Tests: Decorators
# ============================================================================

class TestDecorators:
    """Test @modport and @clocking decorators."""

    def test_modport_decorator_sets_flag(self):
        """Test that @modport decorator sets _is_modport flag."""
        @modport
        class TestModport:
            pass

        assert getattr(TestModport, "_is_modport", False) is True

    def test_clocking_decorator_sets_flag(self):
        """Test that @clocking decorator sets _is_clocking flag."""
        @clocking(clock="clk")
        class TestClocking:
            pass

        assert getattr(TestClocking, "_is_clocking", False) is True


# ============================================================================
# Tests: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_signals_list(self, mock_component):
        """Test interface with no signals."""
        class EmptyInterface(Interface):
            signals = []
            optional = []

        iface = EmptyInterface(component=mock_component)
        assert str(iface) == ""

    def test_multiple_optional_signals(self, mock_component):
        """Test interface with multiple optional signals."""
        class MultiOptionalInterface(Interface):
            signals = ["clk"]
            optional = ["opt1", "opt2", "opt3"]

        iface = MultiOptionalInterface(
            component=mock_component,
            clk=MockHandle("clk")
        )
        # Should not raise
        assert hasattr(iface, "clk")

    def test_interface_inheritance(self, mock_component):
        """Test that interface can be inherited."""
        class BaseInterface(Interface):
            signals = ["clk", "rst"]

        class DerivedInterface(BaseInterface):
            signals = BaseInterface.signals + ["data"]

        iface = DerivedInterface(
            component=mock_component,
            clk=MockHandle("clk"),
            rst=MockHandle("rst"),
            data=MockHandle("data")
        )

        assert hasattr(iface, "clk")
        assert hasattr(iface, "rst")
        assert hasattr(iface, "data")


# ============================================================================
# Tests: ClockedSignal (Advanced Clocking)
# ============================================================================

class TestClockedSignal:
    """Test ClockedSignal wrapper for synchronized sampling and driving."""

    def test_clocked_signal_value_getter(self, mock_component):
        """Test that ClockedSignal can read the underlying handle value."""
        from cocotbext.interface.interface import ClockedSignal
        from cocotb.triggers import RisingEdge

        handle = MockHandle("test_signal")
        handle._value = 42
        clk = MockHandle("clk")

        cs = ClockedSignal(handle, clk, RisingEdge)
        assert cs.value == 42

    def test_clocked_signal_attribute_forwarding(self, mock_component):
        """Test that ClockedSignal forwards attributes to underlying handle."""
        from cocotbext.interface.interface import ClockedSignal
        from cocotb.triggers import RisingEdge

        handle = MockHandle("test_signal")
        clk = MockHandle("clk")

        cs = ClockedSignal(handle, clk, RisingEdge)
        # _name attribute should be forwarded
        assert cs._name == "test_signal"

    def test_clocked_signal_creation_with_rising_edge(self, mock_component):
        """Test creation of ClockedSignal with RisingEdge."""
        from cocotbext.interface.interface import ClockedSignal
        from cocotb.triggers import RisingEdge

        handle = MockHandle("data")
        clk = MockHandle("clk")

        cs = ClockedSignal(handle, clk, RisingEdge)
        assert cs._handle is handle
        assert cs._clk is clk
        assert cs._edge_type is RisingEdge

    def test_clocked_signal_creation_with_falling_edge(self, mock_component):
        """Test creation of ClockedSignal with FallingEdge."""
        from cocotbext.interface.interface import ClockedSignal
        from cocotb.triggers import FallingEdge

        handle = MockHandle("data")
        clk = MockHandle("clk")

        cs = ClockedSignal(handle, clk, FallingEdge)
        assert cs._edge_type is FallingEdge

    def test_clocked_signal_with_input_skew(self, mock_component):
        """Test ClockedSignal creation with input skew."""
        from cocotbext.interface.interface import ClockedSignal
        from cocotb.triggers import RisingEdge, Timer

        handle = MockHandle("data")
        clk = MockHandle("clk")
        input_skew = Timer(1, "ns")

        cs = ClockedSignal(handle, clk, RisingEdge, input_skew=input_skew)
        assert cs._input_skew is input_skew

    def test_clocked_signal_with_output_skew(self, mock_component):
        """Test ClockedSignal creation with output skew."""
        from cocotbext.interface.interface import ClockedSignal
        from cocotb.triggers import RisingEdge, Timer

        handle = MockHandle("data")
        clk = MockHandle("clk")
        output_skew = Timer(2, "ns")

        cs = ClockedSignal(handle, clk, RisingEdge, output_skew=output_skew)
        assert cs._output_skew is output_skew

    # --------------------------------------------------------------------
    # New tests for capturing and ReadOnlyManager
    # --------------------------------------------------------------------

    def test_readonly_manager_single_event_per_timestep(self, monkeypatch):
        """Ensure ReadOnlyManager only creates one Event per simulation time step."""

        async def inner():
            from cocotbext.interface.utils import ReadOnlyManager

            # reset state
            ReadOnlyManager._event = None
            ReadOnlyManager._last_time = -1

            # patch get_sim_time to fixed value
            monkeypatch.setattr('cocotbext.interface.utils.get_sim_time', lambda: 123)

            events = []
            class DummyEvent:
                def __init__(self):
                    events.append(self)
                async def wait(self):
                    pass
            monkeypatch.setattr('cocotbext.interface.utils.Event', DummyEvent)

            # patch start_soon to schedule the coroutine so it doesn't produce a warning
            monkeypatch.setattr('cocotbext.interface.utils.start_soon', lambda coro: asyncio.get_event_loop().create_task(coro))

            # call wait twice; should only create one DummyEvent
            await ReadOnlyManager.wait()
            await ReadOnlyManager.wait()
            assert len(events) == 1

        asyncio.run(inner())

    def test_clockedsignal_capture_multiple_signals(self, mock_component, monkeypatch):
        """Ensure ClockedSignal.capture can be used sequentially on several signals."""

        async def inner():
            from cocotbext.interface.interface import ClockedSignal, ReadOnlyManager
            from cocotb.triggers import RisingEdge

            # create two handles with different values
            h1 = MockHandle("sig1")
            h1._value = 11
            h2 = MockHandle("sig2")
            h2._value = 22
            clk = MockHandle("clk")

            cs1 = ClockedSignal(h1, clk, RisingEdge)
            cs2 = ClockedSignal(h2, clk, RisingEdge)

            # intercept ReadOnlyManager.wait calls
            calls = []
            async def dummy_wait():
                calls.append(True)
            monkeypatch.setattr('cocotbext.interface.interface.ReadOnlyManager.wait', dummy_wait)

            # capture both values sequentially
            val1 = await cs1.capture()
            val2 = await cs2.capture()

            assert val1 == 11
            assert val2 == 22
            assert len(calls) == 2  # manager was invoked for each capture

        asyncio.run(inner())


# ============================================================================
# Tests: ClockingBlock Advanced Features
# ============================================================================

class TestClockingBlockAdvanced:
    """Test advanced clocking block features."""

    def test_clocking_block_wraps_signals_in_clocked_signal(self, mock_component):
        """Test that ClockingBlock wraps signals in ClockedSignal."""
        from cocotbext.interface.interface import ClockedSignal

        iface = InterfaceWithClocking(
            component=mock_component,
            clk=MockHandle("clk"),
            data=MockHandle("data"),
            valid=MockHandle("valid")
        )

        ck = iface.CK
        assert isinstance(ck.data, ClockedSignal)
        assert isinstance(ck.valid, ClockedSignal)

    def test_clocking_block_with_falling_edge(self, mock_component):
        """Test ClockingBlock with FallingEdge."""
        from cocotb.triggers import FallingEdge

        class InterfaceWithFallingEdgeClocking(Interface):
            signals = ["clk", "data"]

            @clocking(clock="clk", edge=FallingEdge)
            class CK_FE:
                inputs = ["data"]
                outputs = []

        iface = InterfaceWithFallingEdgeClocking(
            component=mock_component,
            clk=MockHandle("clk"),
            data=MockHandle("data")
        )

        ck = iface.CK_FE
        assert isinstance(ck, ClockingBlock)
        assert ck._edge_type is FallingEdge

    def test_clocking_block_with_input_output_skews(self, mock_component):
        """Test ClockingBlock with input and output skews."""
        from cocotb.triggers import Timer

        class InterfaceWithSkews(Interface):
            signals = ["clk", "data", "valid"]

            @clocking(clock="clk", input=Timer(1, "ns"), output=Timer(1, "ns"))
            class CK:
                inputs = ["data"]
                outputs = ["valid"]

        iface = InterfaceWithSkews(
            component=mock_component,
            clk=MockHandle("clk"),
            data=MockHandle("data"),
            valid=MockHandle("valid")
        )

        ck = iface.CK
        assert ck.data._input_skew is not None
        assert ck.valid._output_skew is not None

    def test_clocking_block_has_name(self, mock_component):
        """Test that ClockingBlock has a name."""
        iface = InterfaceWithClocking(
            component=mock_component,
            clk=MockHandle("clk"),
            data=MockHandle("data"),
            valid=MockHandle("valid")
        )

        ck = iface.CK
        assert ck._name == "CK"

    def test_clocking_block_interface_reference(self, mock_component):
        """Test that ClockingBlock maintains reference to parent interface."""
        iface = InterfaceWithClocking(
            component=mock_component,
            clk=MockHandle("clk"),
            data=MockHandle("data"),
            valid=MockHandle("valid")
        )

        ck = iface.CK
        assert ck._interface is iface

    def test_clocking_block_with_inout_signals(self, mock_component):
        """Test ClockingBlock with inout signals."""
        class InterfaceWithInout(Interface):
            signals = ["clk", "tri_signal"]

            @clocking(clock="clk")
            class CK:
                inputs = []
                outputs = []
                inouts = ["tri_signal"]

        iface = InterfaceWithInout(
            component=mock_component,
            clk=MockHandle("clk"),
            tri_signal=MockHandle("tri_signal")
        )

        ck = iface.CK
        from cocotbext.interface.interface import ClockedSignal
        assert isinstance(ck.tri_signal, ClockedSignal)


# ============================================================================
# Tests: Modport Advanced Features
# ============================================================================

class TestModportAdvanced:
    """Test advanced modport features."""

    def test_modport_with_inout_signals(self, mock_component):
        """Test modport with inout signals."""
        class InterfaceWithInoutModport(Interface):
            signals = ["clk", "tri_signal"]

            @modport
            class slave:
                inputs = ["clk"]
                outputs = []
                inouts = ["tri_signal"]

        iface = InterfaceWithInoutModport(
            component=mock_component,
            clk=MockHandle("clk"),
            tri_signal=MockHandle("tri_signal")
        )

        slave = iface.slave
        assert slave.clk is not None
        assert slave.tri_signal is not None

    def test_modport_with_multiple_instances(self, mock_component):
        """Test modport with multiple instances."""
        class InterfaceWithMultiModport(Interface):
            signals = ["clk", "data"]

            @modport(count=3)
            class port:
                inputs = ["data"]
                outputs = ["clk"]

        iface = InterfaceWithMultiModport(
            component=mock_component,
            clk=MockHandle("clk"),
            data=MockHandle("data")
        )

        ports = iface.port
        assert isinstance(ports, list)
        assert len(ports) == 3
        assert all(isinstance(p, ModportView) for p in ports)

    def test_modport_multiple_instance_names(self, mock_component):
        """Test that multiple modport instances are named correctly."""
        class InterfaceWithMultiModport(Interface):
            signals = ["clk", "data"]

            @modport(count=2)
            class port:
                inputs = ["data"]
                outputs = ["clk"]

        iface = InterfaceWithMultiModport(
            component=mock_component,
            clk=MockHandle("clk"),
            data=MockHandle("data")
        )

        ports = iface.port
        assert ports[0]._name == "port_0"
        assert ports[1]._name == "port_1"

    def test_modport_with_callables(self, mock_component):
        """Test modport that exposes callable methods."""
        class InterfaceWithCallables(Interface):
            signals = ["clk", "data"]

            def reset(self):
                """Reset method."""
                return "reset_called"

            def enable(self):
                """Enable method."""
                return "enable_called"

            @modport
            class controller:
                inputs = []
                outputs = ["clk"]
                callables = ["reset", "enable"]

        iface = InterfaceWithCallables(
            component=mock_component,
            clk=MockHandle("clk"),
            data=MockHandle("data")
        )

        controller = iface.controller
        assert controller.reset() == "reset_called"
        assert controller.enable() == "enable_called"

    def test_modport_missing_callable_raises_error(self, mock_component):
        """Test that modport with missing callable raises AttributeError."""
        class InterfaceWithMissingCallable(Interface):
            signals = ["clk", "data"]

            @modport
            class controller:
                inputs = []
                outputs = ["clk"]
                callables = ["nonexistent_method"]

        with pytest.raises(AttributeError, match="Method 'nonexistent_method' listed in modport"):
            InterfaceWithMissingCallable(
                component=mock_component,
                clk=MockHandle("clk"),
                data=MockHandle("data")
            )

    def test_modport_linked_to_clocking_block(self, mock_component):
        """Test modport that links to a clocking block."""
        class InterfaceWithClockingAndModport(Interface):
            signals = ["clk", "data", "valid"]

            @clocking(clock="clk")
            class CK:
                inputs = ["data"]
                outputs = ["valid"]

            @modport(clocking="CK")
            class dut:
                inputs = ["data"]
                outputs = ["valid"]
                callables = []

        iface = InterfaceWithClockingAndModport(
            component=mock_component,
            clk=MockHandle("clk"),
            data=MockHandle("data"),
            valid=MockHandle("valid")
        )

        dut = iface.dut
        assert hasattr(dut, "cb")
        assert dut.cb is iface.CK

    def test_modport_count_via_kwargs(self, mock_component):
        """Test modport instance count override via kwargs."""
        class InterfaceWithModportCount(Interface):
            signals = ["clk", "data"]

            @modport(count=2)
            class port:
                inputs = ["data"]
                outputs = ["clk"]

        iface = InterfaceWithModportCount(
            component=mock_component,
            clk=MockHandle("clk"),
            data=MockHandle("data"),
            port_count=4  # Override the count
        )

        ports = iface.port
        assert isinstance(ports, list)
        assert len(ports) == 4

    def test_modport_single_instance_not_list(self, mock_component):
        """Test that single modport instance is not a list."""
        class InterfaceWithSingleModport(Interface):
            signals = ["clk", "data"]

            @modport(count=1)
            class port:
                inputs = ["data"]
                outputs = ["clk"]

        iface = InterfaceWithSingleModport(
            component=mock_component,
            clk=MockHandle("clk"),
            data=MockHandle("data")
        )

        port = iface.port
        # Single instance should not be wrapped in list
        assert isinstance(port, ModportView)
        assert not isinstance(port, list)

    def test_modport_decorator_with_count_parameter(self):
        """Test @modport decorator with count parameter."""
        @modport(count=5)
        class TestModport:
            pass

        assert getattr(TestModport, "_instance_count", 1) == 5

    def test_modport_decorator_with_clocking_parameter(self):
        """Test @modport decorator with clocking parameter."""
        @modport(clocking="CB")
        class TestModport:
            pass

        assert getattr(TestModport, "_clocking_name", None) == "CB"

    def test_modport_all_signals_accessible(self, mock_component):
        """Test that all signal types are accessible in modport."""
        class InterfaceWithAllSignalTypes(Interface):
            signals = ["clk", "data_in", "data_out", "tri_signal"]

            @modport
            class port:
                inputs = ["data_in"]
                outputs = ["data_out"]
                inouts = ["tri_signal"]

        data_in_handle = MockHandle("data_in")
        data_out_handle = MockHandle("data_out")
        tri_handle = MockHandle("tri_signal")

        iface = InterfaceWithAllSignalTypes(
            component=mock_component,
            clk=MockHandle("clk"),
            data_in=data_in_handle,
            data_out=data_out_handle,
            tri_signal=tri_handle
        )

        port = iface.port
        assert port.data_in is data_in_handle
        assert port.data_out is data_out_handle
        assert port.tri_signal is tri_handle


# ============================================================================
# Tests: Clocking Decorator
# ============================================================================

class TestClockingDecorator:
    """Test @clocking decorator parameters."""

    def test_clocking_decorator_stores_clock_name(self):
        """Test that @clocking decorator stores clock name."""
        @clocking(clock="my_clk")
        class TestClocking:
            pass

        assert getattr(TestClocking, "_clock_name", None) == "my_clk"

    def test_clocking_decorator_stores_edge_type(self):
        """Test that @clocking decorator stores edge type."""
        from cocotb.triggers import FallingEdge

        @clocking(clock="clk", edge=FallingEdge)
        class TestClocking:
            pass

        assert getattr(TestClocking, "_edge_type", None) is FallingEdge

    def test_clocking_decorator_stores_input_skew(self):
        """Test that @clocking decorator stores input skew."""
        from cocotb.triggers import Timer

        skew = Timer(1, "ns")

        @clocking(clock="clk", input=skew)
        class TestClocking:
            pass

        assert getattr(TestClocking, "_input_skew", None) is skew

    def test_clocking_decorator_stores_output_skew(self):
        """Test that @clocking decorator stores output skew."""
        from cocotb.triggers import Timer

        skew = Timer(1, "ns")

        @clocking(clock="clk", output=skew)
        class TestClocking:
            pass

        assert getattr(TestClocking, "_output_skew", None) is skew

    def test_clocking_decorator_initializes_signal_lists(self):
        """Test that @clocking decorator initializes signal lists."""
        @clocking(clock="clk")
        class TestClocking:
            pass

        assert hasattr(TestClocking, "inputs")
        assert hasattr(TestClocking, "outputs")
        assert hasattr(TestClocking, "inouts")


# ============================================================================
# Tests: Modport Decorator
# ============================================================================

class TestModportDecorator:
    """Test @modport decorator parameters."""

    def test_modport_decorator_initializes_signal_lists(self):
        """Test that @modport decorator initializes signal lists."""
        @modport
        class TestModport:
            pass

        assert hasattr(TestModport, "inputs")
        assert hasattr(TestModport, "outputs")
        assert hasattr(TestModport, "inouts")
        assert hasattr(TestModport, "callables")

    def test_modport_decorator_default_instance_count(self):
        """Test that @modport decorator sets default instance count to 1."""
        @modport
        class TestModport:
            pass

        assert getattr(TestModport, "_instance_count", 1) == 1

    def test_modport_decorator_with_type_argument(self):
        """Test @modport decorator applied directly to class (without parens)."""
        @modport
        class TestModport:
            inputs = ["a"]
            outputs = ["b"]

        assert getattr(TestModport, "_is_modport", False) is True


# ============================================================================
# Tests: Integration (Clocking + Modport)
# ============================================================================

class TestClockingModportIntegration:
    """Test integration of clocking and modport features."""

    def test_interface_with_multiple_clocking_blocks(self, mock_component):
        """Test interface with multiple clocking blocks."""
        class InterfaceWithMultiClocking(Interface):
            signals = ["clk1", "clk2", "data"]

            @clocking(clock="clk1")
            class CK1:
                inputs = ["data"]
                outputs = []

            @clocking(clock="clk2")
            class CK2:
                inputs = ["data"]
                outputs = []

        iface = InterfaceWithMultiClocking(
            component=mock_component,
            clk1=MockHandle("clk1"),
            clk2=MockHandle("clk2"),
            data=MockHandle("data")
        )

        assert isinstance(iface.CK1, ClockingBlock)
        assert isinstance(iface.CK2, ClockingBlock)
        assert iface.CK1._clk_handle._name == "clk1"
        assert iface.CK2._clk_handle._name == "clk2"

    def test_interface_with_multiple_modports_and_clocking(self, mock_component):
        """Test interface with multiple modports linked to clocking."""
        class InterfaceWithMultiModportClocking(Interface):
            signals = ["clk", "data", "valid"]

            @clocking(clock="clk")
            class CK:
                inputs = ["data"]
                outputs = ["valid"]

            @modport(clocking="CK")
            class master:
                inputs = []
                outputs = ["data", "valid"]

            @modport(clocking="CK")
            class slave:
                inputs = ["data", "valid"]
                outputs = []

        iface = InterfaceWithMultiModportClocking(
            component=mock_component,
            clk=MockHandle("clk"),
            data=MockHandle("data"),
            valid=MockHandle("valid")
        )

        master = iface.master
        slave = iface.slave

        assert hasattr(master, "cb")
        assert hasattr(slave, "cb")
        assert master.cb is iface.CK
        assert slave.cb is iface.CK

    def test_complex_interface_scenario(self, mock_component):
        """Test complex real-world interface scenario."""
        class AXILiteInterface(Interface):
            signals = ["aclk", "aresetn", "awaddr", "awvalid", "awready",
                      "wdata", "wvalid", "wready", "bvalid", "bready",
                      "araddr", "arvalid", "arready", "rdata", "rvalid", "rready"]

            def write_transaction(self, addr, data):
                return f"write at {addr}: {data}"

            def read_transaction(self, addr):
                return f"read from {addr}"

            @clocking(clock="aclk")
            class cb:
                inputs = ["awready", "wready", "bvalid", "arready", "rvalid", "rdata"]
                outputs = ["awaddr", "awvalid", "wdata", "wvalid", "bready",
                          "araddr", "arvalid", "rready"]

            @modport(clocking="cb")
            class master:
                inputs = ["awready", "wready", "bvalid", "arready", "rvalid", "rdata"]
                outputs = ["awaddr", "awvalid", "wdata", "wvalid", "bready",
                          "araddr", "arvalid", "rready"]
                callables = ["write_transaction", "read_transaction"]

        iface = AXILiteInterface(
            component=mock_component,
            aclk=MockHandle("aclk"),
            aresetn=MockHandle("aresetn"),
            awaddr=MockHandle("awaddr"),
            awvalid=MockHandle("awvalid"),
            awready=MockHandle("awready"),
            wdata=MockHandle("wdata"),
            wvalid=MockHandle("wvalid"),
            wready=MockHandle("wready"),
            bvalid=MockHandle("bvalid"),
            bready=MockHandle("bready"),
            araddr=MockHandle("araddr"),
            arvalid=MockHandle("arvalid"),
            arready=MockHandle("arready"),
            rdata=MockHandle("rdata"),
            rvalid=MockHandle("rvalid"),
            rready=MockHandle("rready")
        )

        master = iface.master
        assert master.write_transaction(0x1000, 0xDEADBEEF) == "write at 4096: 3735928559"
        assert master.read_transaction(0x1000) == "read from 4096"
        assert hasattr(master, "cb")
        from cocotbext.interface.interface import ClockedSignal
        # modport exposes raw signals from interface
        assert isinstance(master.awaddr, MockHandle)
        # clocking block wraps signals in ClockedSignal
        assert isinstance(master.cb.awaddr, ClockedSignal)
        # ClockedSignal stores the MockHandle internally
        assert master.cb.awaddr._handle is master.awaddr

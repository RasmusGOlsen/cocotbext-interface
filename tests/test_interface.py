"""
Pytest tests for the Interface class.

These tests use mocks for cocotb dependencies to allow testing in pure Python
without requiring a simulator.
"""

from typing import Iterator

import pytest

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
        self._value = 0

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

    @clocking
    class CK:
        clock = ["clk"]
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
        assert ck._clock is clk_handle

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
        assert ck.data is data_handle
        assert ck.valid is valid_handle


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
        @clocking
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

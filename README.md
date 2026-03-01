# Interface Documentation

This interface package provides a Pythonic implementation of SystemVerilog-style
**Interface**, **Modport**, and **Clocking Block** for cocotb. It is designed
to bridge the gap between hardware-centric verification concepts and the
flexibility of Python, ensuring race-free simulations and clean, reusable
verification IP.

---

## 1. Project Overview

In standard cocotb, signal assignments are immediate and reads are subject to
simulator delta-cycle races. This package introduces a timing-accurate
**Clocking Block** mechanism.

By mimicking the SystemVerilog `interface` structure, verification engineers
can:

* **Group Signals:** Define structural signal groups once in a central class.
* **Restrict Access:** Use **Modports** to define directional access for
  different VIP components (e.g., Sources, Sinks and Monitors).
* **Enforce Synchronicity:** Use **Clocking Blocks** to ensure signals are
  driven and sampled at the correct time relative to clock edges, supporting
  both time-based and event-based skews.
* **Define Functional APIs:** Using Python methods (`def` and `async def`)
  within the interface to build API tasks for implementing interface protocols
  such as AXI, Avalon, I2C, SPI, etc.

---

## 2. Part I: Designing the Interface

This section covers how to define the structure, timing, and access rules for a
interface protocol.

### 2.1 The Interface Class

To define a bus, inherit from the Interface class. Signals are now defined as
class attributes using type hints, which determines whether they are required
or optional during RTL binding.

#### Defining Signals

The framework uses the presence of a default value to distinguish between
signal types:

**Mandatory Signals**: Define these with a type hint (e.g., `LogicArrayObject`).
If no default value is assigned, the framework will raise an error if the
signal cannot be found in the RTL.

**Optional Signals**: Assign None as a default value. This indicates that the
signal may or may not exist in the design.

```python
class MyBus(Interface):
    # Mandatory signal: Must be present in the RTL
    data: LogicArrayObject

    # Optional signal: Defaults to None if not found
    rdy: LogicObject | None = None
```

These attribute names are used directly by the framework to search for matching signal names within the RTL hierarchy.

### 2.2 Defining Clocking Blocks

Use the `@clocking` decorator on an inner class. This defines the temporal
behavior for synchronous signals.

* `clock`: The name of the clock signal in the `signals` list.
* `edge`: The trigger (e.g., `RisingEdge`, `FallingEdge`, `Edge`).
* `input`: Input skew (Sample delay). Can be a `Timer` or another `Trigger`.
* `output`: Output skew (Drive delay).

### 2.3 Defining Modports

Use the `@modport` decorator to group signals and clocking blocks for specific
roles like `source`, `sink` or `monitor`.

* `name`: Name of the modport
* `clocking`: Name of the clocking block to link.
* `Input`/`Output`/`InOut`: Signals accessible in this modport.
* `Import`: Method names (APIs) exposed to this modport.

### 2.4 Complete Definition Example

```python
from cocotb.handles import LogicArrayObject, LogicObject
from cocotb.triggers import RisingEdge, FallingEdge, Timer
from interface_framework import Interface, modport, clocking, Import, Input, Output

class AxiStream(Interface):
    clk:    LogicObject
    rst_n:  LogicObject
    tdata:  LogicArrayObject
    tvalid: LogicObject
    tready: LogicObject

    async def reset(self):
        self.rst_n.value = 0
        await Timer(10, 'ns')
        self.rst_n.value = 1

    # Define Source-side timing
    @clocking(clock="clk", edge=RisingEdge, input=Timer(1, 'ns'), output=Timer(2, 'ns'))
    class source_cb:
        tready: Input[LogicObject]
        tdata:  Output[LogicArrayObject]
        tvalid: Output[LogicObject]

    # Define Sink-side timing
    @clocking(clock="clk", edge=RisingEdge, input=Timer(1, 'ns'))
    class sink_cb:
        tdata:  Input[LogicArrayObject]
        tvalid: Input[LogicObject]
        tready: Output[LogicObject]

    @modport(clocking="source_cb")
    class source:
        rst_n: Output[LogicObject]
        reset: Import[Callable]

    @modport(clocking="sink_cb")
    class sink:
        reset: Import[Callable]
```

---

## 3. Part II: Using the Interface

This section covers how to connect the interface to the DUT and use it in a
test.

### 3.1 Instantiation & Connection

Interfaces are connected to the RTL using named constructors. These methods
automatically map your class attributes to the corresponding signals in the HDL
hierarchy.

#### Connection Methods

##### 1. Direct Mapping (`from_entity`)

Use `from_entity` when the signal names in the RTL match your class attribute
names exactly under a specific hierarchy level. This is a strict connection
method that does not support overrides or pattern substitutions.

```python
# Connects to dut.tdata and
dut.rdy directly
bus = MyBus.from_entity(dut)
```

##### 2. Explicit Assignment (`from_signal`)

The `from_signal` constructor allows for manual binding of signal handles to
attributes. This is used when signals do not follow a pattern or exist in
different hierarchies.

* **Behavior:** Uses keyword arguments to map handles to class attributes.
* **Validation:** Still enforces that all mandatory signals defined in the
  class are provided.

```python
# Manual connection: Explicitly pass handles for each attribute
bus = MyBus.from_signal(
    tdata = dut.top.data_bus,
    rdy   = dut.extra_logic.ready_bit
)
```

##### 3. Pattern Matching (`from_pattern`)

The `from_pattern` method is used when signals follow a specific naming
convention. The pattern argument must contain the % wildcard, which is
substituted with each attribute name defined in your class.

```python
# Replaces % with attribute names: e.g. 'u_axi_tdata'
bus = MyBus.from_pattern(dut, pattern="u_axi_%")
```

In addition to the % wildcard, `from_pattern` supports flexible discovery:

* **Globbing:** Use * (any characters), ? (single character), or + (one or more
  to match signals.
* **Regex:** Wrap the pattern in /.../ (e.g., /u_axi_.*_%/) for complex matching
  logic.

```python
# Simple substitution: Matches 'u_axi_tdata'
bus = MyBus.from_pattern(dut, pattern="u_axi_%")

# Globbing: Matches 'u_axi_0_tdata' or 'u_axi_stage1_tdata'
bus = MyBus.from_pattern(dut, pattern="u_axi_*_%")

# Regex: Matches signals with specific numeric suffixes
bus = MyBus.from_pattern(dut, pattern="/u_axi_[0-9]_%/")
```

#### Advanced Pattern Options

While `from_entity` is strict, `from_pattern` allows for flexibility when the
RTL structure is non-standard:

* **Keyword Overrides:** Pass a signal handle as a keyword argument to skip the
  pattern search for that specific attribute.Indexing: Use the `idx` argument to
  index into all signals discovered via the pattern (e.g., `dut.u_axi_tdata[1]`).

```python
# Connect to index 1 and manually override 'rdy'
bus = MyBus.from_pattern(dut, pattern="u_axi_%", idx=1, rdy=dut.global_rdy)
```

> [!TIP] To confirm your signals are correctly bound, you can print `print(bus)`
> or `print(f"{bus=}")` the bus object to inspect the resolved RTL paths.

### 3.2 Synchronous Driving (Non-Blocking)

When using a modport's clocking block, driving a signal schedules an update for
the next clock edge + output skew. It does not block the current coroutine.

```python
# Drive data through the source modport
bus.source.src_cb.tdata.value = 0xABCD
bus.source.src_cb.tvalid.value = 1
```

### 3.3 Synchronous Sampling (Blocking)

To read a signal synchronously, you **must** use `await ...capture()`. This
ensures the simulation waits for the clock edge and the defined input skew
before returning the value.

```python
# Wait until sink is ready
while await bus.source.src_cb.tready.capture() == 0:
    await bus.source.src_cb.wait() # Wait for 1 clock cycle
```

### 3.4 Using Interface APIs

Methods defined in the interface and imported in the modport can be called
directly.

```python
# Call the reset task defined by the VIP developer
await bus.source.reset()
```

---

## 4. Technical Summary

| SystemVerilog Concept | Framework Implementation | Behavior |
| :--- | :--- | :--- |
| `interface` | `class MyBus(Interface):` | Structural container. |
| `.*` | `pattern="%"` | Substitution-based wildcard discovery. |
| `.clk(sys_clk)` | `clk=dut.sys_clk` | Explicit named mapping override. |
| `modport` | `@modport` | Role-based grouping (Source/Sink). |
| `clocking` | `@clocking` | Temporal skews and edge triggers. |
| `cb.sig <= val` | `cb_name.sig.value = val` | Non-blocking drive (Setter). |
| `val = cb.sig` | `val = await cb_name.sig.capture()` | Synchronous sample (Coroutine). |
| `##N` | `await cb_name.wait(N)` | Cycle-based delay. |

---

## 5. Best Practices

1. Define separate clocking blocks for Source and Sink roles to account for
   different signal access and physical skews.
2. Don't give access to synchronous signals in the `modport`, this forces the
   user to use the clocking block.
3. Leverage the `%` wildcard in patterns to avoid manually connecting dozens of
   signals.

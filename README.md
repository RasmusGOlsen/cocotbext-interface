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

Inherit from `Interface` and define a the signals. Mandatory signal are defined
in the `signals` list and optional signals are defined in the `optional` list.
These are the signal names that can be searched for in the RTL.

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
* `inputs`/`outputs`: Lists of signals accessible in this modport.
* `callables`: List of method names (APIs) exposed to this modport.

### 2.4 Complete Definition Example

```python
from cocotb.triggers import RisingEdge, FallingEdge, Timer
from interface_framework import Interface, modport, clocking

class AxiStream(Interface):
    signals = ["clk", "rst_n", "tdata", "tvalid", "tready"]

    async def reset(self):
        self.rst_n.value = 0
        await Timer(10, 'ns')
        self.rst_n.value = 1

    # Define Source-side timing
    @clocking(clock="clk", edge=RisingEdge, input=Timer(1, 'ns'), output=Timer(2, 'ns'))
    class source_cb:
        inputs = ["tready"]
        outputs = ["tdata", "tvalid"]

    # Define Sink-side timing
    @clocking(clock="clk", edge=RisingEdge, input=Timer(1, 'ns'))
    class sink_cb:
        inputs = ["tdata", "tvalid"]
        outputs = ["tready"]

    @modport(clocking="source_cb")
    class source:
        outputs = ["rst_n"]
        callables = ["reset"]

    @modport(clocking="sink_cb")
    class sink:
        callables = ["reset"]
```

---

## 3. Part II: Using the Interface

This section covers how to connect the interface to the DUT and use it in a
test.

### 3.1 Instantiation & Connection

The `pattern` argument is optional, but if used must contain the `%` wildcard.
The `%` is substituted with each name in the `signals` and `optional` lists.
You can also use globbing or regex (if wrapped in `/.../`).

```python
@cocotb.test()
async def test_tx(dut):
    # Pattern matching + Explicit Override
    # Replaces % with signal names: e.g. 'u_axi_tdata'
    bus = AxiStream(dut, pattern="u_axi_%", clk=dut.sys_clock)
```

> [!TIP]
> To confirm your signal are correctly connected and the interface, you can
> the `bus` object to inspect it.
>
> ```python
> print(bus)
> print(f"{bus=}")
> ```

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

Methods defined in the interface and exposed in the modport `callables` can be
called directly.

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

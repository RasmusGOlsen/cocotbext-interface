from abc import ABC, abstractmethod

from .interface import Interface


class MemoryMappedInterface(Interface, ABC):

    @abstractmethod
    async def read(self, address, size=4, **kwargs):
        """Basic blocking read. Returns data."""
        pass

    @abstractmethod
    async def write(self, address, data, strobe=None, **kwargs):
        """Basic blocking write. 'strobe' allows for byte-enables."""
        pass

    async def burst_read(self, address, length, size=4, **kwargs):
        """
        Default fallback: Executes 'length' individual reads.
        Override this for AXI/Avalon to use native burst hardware features.
        """
        results = []
        for i in range(length):
            data = await self.read(address + (i * size), size=size, **kwargs)
            results.append(data)
        return results

    async def burst_write(self, address, data_list, size=4, **kwargs):
        """
        Default fallback: Executes individual writes from a list.
        """
        for i, data in enumerate(data_list):
            await self.write(address + (i * size), data, size=size, **kwargs)

    @abstractmethod
    def is_busy(self) -> bool:
        """Non-blocking check: Is the interface currently performing an operation?"""
        pass

    @abstractmethod
    async def wait_until_idle(self) -> None:
        """Blocks until is_busy() is False."""
        pass

    @abstractmethod
    async def reset(self):
        pass

    @abstractmethod
    def set_backpressure(self, enable: bool) -> None:
        pass

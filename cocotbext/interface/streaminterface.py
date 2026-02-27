from abc import ABC, abstractmethod
from typing import Any

from .interface import Interface


class StreamInterface(Interface, ABC):
    @abstractmethod
    async def send(self, data):
        pass

    @abstractmethod
    async def receive(self) -> Any:
        pass

    @abstractmethod
    def set_backpressure(self, enable: bool) -> None:
        pass

    @abstractmethod
    async def reset(self) -> None:
        pass

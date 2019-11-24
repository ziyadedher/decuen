"""Workaround for MyPy issue with not being able to store callable functions properly as instance attributes."""

from typing import Any, Generic, TypeVar

CallableType = TypeVar("CallableType")


class FunctionProperty(Generic[CallableType]):
    """Wrapper class to provide ability to store functions as class properties."""

    def __get__(self, oself: Any, owner: Any) -> CallableType:
        """Magic."""
        ...

    def __set__(self, oself: Any, value: CallableType) -> None:
        """Magic."""
        ...

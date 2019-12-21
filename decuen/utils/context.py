"""Utilities for managing environmental context."""

import functools
from typing import Callable, ClassVar, List, Optional, TypeVar, Union, cast

from gym import Env, Space  # type: ignore


class EnvironmentalContext:
    """Context manager for environmental spaces.

    Use this to create custom contexts that can subsequently either be used as a context manager to enclose a scope in
    context, or be passed into the universal context for use as the default context. This context is then used by the
    `with_context` function wrapper to inject state and action space environmental context into function that need it.

    For example, we can create a scope-encapsulated context by using an instance of this class as a context manager:
    >>> with EnvironmentalContext(state_space=..., action_space=...) as ctx:
    >>>     ...
    Notice that if either <state_space> or <action_space> is unspecified in the initialization, this context manager
    is considered partial. Functions requiring context will still require the unspecified spaces or else an error will
    be raised.

    We can also directly set an environmental context to be the universal context:
    >>> CTX.set_context(EnvironmentalContext(state_space=..., action_space=...))
    """

    _state_space: Optional[Space]
    _action_space: Optional[Space]
    _prev_contexts: List['EnvironmentalContext']

    def __init__(self, *, state_space: Optional[Space] = None, action_space: Optional[Space] = None) -> None:
        """Initialize an envionmental context with optional state and action spaces."""
        self._state_space = state_space
        self._action_space = action_space
        self._prev_contexts = []

    @property
    def state_space(self) -> Space:
        """Access the state space associated with this environmental context."""
        if self._state_space is None:
            return ValueError("no state space in context")
        return self._state_space

    @state_space.setter
    def state_space(self, space: Space) -> None:
        self._state_space = space

    @property
    def action_space(self) -> Space:
        """Access the action space associated with this environmental context."""
        if self._action_space is None:
            return ValueError("no action space in context")
        return self._action_space

    @action_space.setter
    def action_space(self, space: Space) -> None:
        self._action_space = space

    def __enter__(self):
        """Use this environmental context as a context manager."""
        self._prev_contexts.append(CTX.get_context())
        CTX.set_context(self)

    def __exit__(self, *args):
        """Exit this environmental context manager and restore the previous context."""
        CTX.set_context(self._prev_contexts.pop())


class UniversalContext:
    """Singleton encapsulating the universal environmental context.

    Simply provides an interface to get and set the current context. Note that this management is not thread-safe.
    """

    _context: ClassVar[EnvironmentalContext] = EnvironmentalContext()

    @classmethod
    def get_context(cls) -> EnvironmentalContext:
        """Get the current environmental context."""
        return cls._context

    @classmethod
    def set_context(cls, context: EnvironmentalContext) -> None:
        """Set the current environmental context."""
        cls._context = context


CTX = UniversalContext


def get_context() -> EnvironmentalContext:
    """Get the current universal context."""
    return CTX.get_context()


def set_context(context: Union[EnvironmentalContext, Env]) -> None:
    """Set the current universal context.

    Can set based on an `EnvironmentalContext` object or an OpenAI Gym `Env` object.
    """
    if isinstance(context, Env):
        context = EnvironmentalContext(state_space=context.observation_space, action_space=context.action_space)
    CTX.set_context(context)


FuncType = TypeVar("FuncType", bound=Callable)


def with_context(func: FuncType) -> FuncType:
    """Inject the current environmental context into this function.

    This function was constructed to be used as a decorator to initialization functions that expect the state and action
    space of the environment they are working in. Using this functions requires the last two arguments of the wrapped
    function to respectively be `state_space` and `action_space` and they must be keyword-only arguments.

    Using this function injects the current envrionmental context into the function by way of those keyword arguments.
    For this function to have any effect, the current context must be populated which can be done by ways of the
    universal singleton context `UniversalContext` aliased by `CTX` or by directly using an `EnvironmentalContext`
    context manager to associate a context with a scope.
    """
    state_space_kwarg_name = "state_space"
    action_space_kwarg_name = "action_space"

    num_kwargs = func.__code__.co_kwonlyargcount
    kwarg_names = func.__code__.co_varnames

    if num_kwargs < 2:
        raise ValueError(
            "wrapper {with_context.__name__} requires function to accept at least two keyword-only arguments")
    if kwarg_names[-2] != state_space_kwarg_name:
        raise ValueError(
            "wrapper {with_context.__name__} requires second-last argument to be named '{state_space_kwarg_name}'")
    if kwarg_names[-1] != action_space_kwarg_name:
        raise ValueError(
            "wrapper {with_context.__name__} requires last argument to be named '{action_space_kwarg_name}'")

    @functools.wraps(func)
    def context_wrapper(*args, **kwargs):
        if state_space_kwarg_name not in kwargs:
            kwargs[state_space_kwarg_name] = CTX.get_context().state_space
        if action_space_kwarg_name not in kwargs:
            kwargs[action_space_kwarg_name] = CTX.get_context().action_space
        func(*args, **kwargs)
    return cast(FuncType, context_wrapper)


# pylint: disable=too-few-public-methods
class Contextful:
    """Thin interface class to introduce contextful state and action spaces to any class."""

    _state_space: Space
    _action_space: Space

    def __init__(self) -> None:
        """Initialize a contextful object and populate the state an action spaces based on context."""
        self._state_space = get_context().state_space
        self._action_space = get_context().action_space

    @property
    def state_space(self) -> Space:
        """Access the state space associated with this environmental context."""
        return self._state_space

    @property
    def action_space(self) -> Space:
        """Access the action space associated with this environmental context."""
        return self._action_space
# pylint: enable=too-few-public-methods

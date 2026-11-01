"""Session lifetime management for processor compositions."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import logging
from threading import RLock


_sequential = ContextVar('manuscript_sequential_models', default=False)


def check_memory_policy(policy):
    if not isinstance(policy, str) or policy not in ('keep', 'sequential'):
        raise ValueError("memory_policy must be 'keep' or 'sequential'")
    return policy


def _walk(processor, sequential=False, parallel=False, ancestors=()):
    if id(processor) in ancestors:
        raise ValueError('Processor compositions must not contain cycles')
    if isinstance(processor, ModelComposition):
        sequential = sequential or processor.memory_policy == 'sequential'
        parallel = parallel or bool(getattr(processor, 'parallel', False))
        if sequential and parallel:
            raise ValueError("memory_policy='sequential' cannot be combined with parallel=True, including nested compositions")
        for child in processor._memory_children():
            if child is not None:
                yield from _walk(child, sequential, parallel, (*ancestors, id(processor)))
    else:
        if sequential and not callable(getattr(processor, 'close', None)):
            raise TypeError(f"{type(processor).__name__} must implement close() for memory_policy='sequential'; stateless processors may use a no-op close()")
        yield processor


def validate_composition(processor):
    list(_walk(processor, _sequential.get()))


def _release(processor):
    close = getattr(processor, 'close', None)
    if callable(close):
        close()
        if getattr(processor, '_runtime_sessions', None):
            raise RuntimeError(f'{type(processor).__name__} could not release all runtime sessions')


def _release_children(processor):
    errors, seen = [], set()
    for child in _walk(processor):
        if id(child) in seen:
            continue
        seen.add(id(child))
        try:
            _release(child)
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise errors[0]


@contextmanager
def processor_scope(processor):
    """Release a leaf after execution, preserving an original inference error."""
    if not _sequential.get() or isinstance(processor, ModelComposition):
        yield
        return
    try:
        yield
    except BaseException:
        try:
            _release(processor)
        except Exception:
            logging.getLogger(__name__).exception('Failed to close processor after prediction failed')
        raise
    else:
        _release(processor)


def managed_prediction(method):
    @wraps(method)
    def predict(self, *args, **kwargs):
        inherited = _sequential.get()
        sequential = inherited or self.memory_policy == 'sequential'
        # Serialize repeated calls on the same sequential composition. Members
        # shared with other compositions/direct calls need caller-side locking.
        with self._memory_lock if sequential else nullcontext():
            validate_composition(self)
            if sequential and not inherited:
                # Models may have been warmed before this policy was selected.
                # Release all participants before starting the first one.
                _release_children(self)
            token = _sequential.set(sequential)
            try:
                return method(self, *args, **kwargs)
            finally:
                _sequential.reset(token)
    return predict


class ModelComposition:
    """Common close/context-manager support for built-in compositions."""

    def _init_memory_policy(self, policy):
        self.memory_policy = check_memory_policy(policy)
        self._memory_lock = RLock()

    def __getstate__(self):
        return {key: value for key, value in vars(self).items() if key != "_memory_lock"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._memory_lock = RLock()

    def _memory_children(self):
        raise NotImplementedError

    def close(self):
        """Release participant sessions; later predictions can load them again."""
        with self._memory_lock:
            _release_children(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

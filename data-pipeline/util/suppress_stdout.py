import io
import sys
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = original_stdout

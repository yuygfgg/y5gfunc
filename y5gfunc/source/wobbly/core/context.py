"""
Processing contexts and helpers for Wobbly parser.
"""

from contextlib import contextmanager
from typing import Iterator, Dict, Any, Optional

from ..errors import WobblyError, WobblyProcessError


@contextmanager
def safe_processing(stage: str, details: Optional[Dict[str, Any]] = None) -> Iterator[None]:
    """
    Safe processing context manager for catching and transforming exceptions
    
    Args:
        stage: Processing stage name
        details: Optional details to include in error
        
    Yields:
        Nothing
    """
    try:
        yield
    except WobblyError:
        # Already a WobblyError, re-raise
        raise
    except Exception as e:
        # Convert other exceptions to WobblyProcessError
        raise WobblyProcessError(
            f"Error during {stage}",
            stage=stage,
            details=details,
            cause=e
        )
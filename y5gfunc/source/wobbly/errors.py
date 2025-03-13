"""
Error type definitions for Wobbly parser.
"""

from typing import Optional, Any


class WobblyError(Exception):
    """Base error type for Wobbly parser"""
    def __init__(self, message: str, cause: Optional[Exception] = None):
        self.cause = cause
        super().__init__(f"{message}" + (f" Caused by: {cause}" if cause else ""))


class WobblyParseError(WobblyError):
    """Error when parsing Wobbly project files"""
    pass


class WobblyProcessError(WobblyError):
    """Error when processing Wobbly projects"""
    def __init__(
        self, 
        message: str, 
        stage: str = "unknown", 
        details: Optional[Any] = None, 
        cause: Optional[Exception] = None
    ):
        self.stage = stage
        self.details = details
        super_msg = f"{message} (stage: {stage})"
        super().__init__(super_msg, cause)


class WobblyInputError(WobblyError):
    """Input file related errors"""
    pass
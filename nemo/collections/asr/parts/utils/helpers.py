from typing import Optional, Union, List, Dict, Any, Tuple, Callable, TypeVar, Generic, Type, cast, Sequence, Mapping



def exists(val: Any) -> bool:
    """
    Returns True if val is not None, False otherwise.
    """
    return val is not None

def isfalse(val: Any) -> bool:
    """
    Returns True if val is False, False otherwise.
    """
    return val is False


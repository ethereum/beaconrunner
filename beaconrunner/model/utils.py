import copy
from dataclasses import field

def default(obj):
    return field(default_factory=lambda: copy.copy(obj))

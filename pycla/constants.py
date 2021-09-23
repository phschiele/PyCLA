from enum import Enum


class Direction(Enum):
    LOWER = "lower"
    HIGHER = "higher"


class VariableState(Enum):
    IN = 0
    UP = 1
    LOW = 2


class SimplexReturn(Enum):
    OPTIMAL = "optimal"
    DEGENERATE = "degenerate"


class SimplexPhase(Enum):
    ONE = 1
    TWO = 2

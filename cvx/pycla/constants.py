from __future__ import annotations


class Direction:
    LOWER = "lower"
    HIGHER = "higher"


class VariableState:
    IN = 0
    UP = 1
    LOW = 2


class SimplexReturn:
    OPTIMAL = "optimal"
    DEGENERATE = "degenerate"


class SimplexPhase:
    ONE = 1
    TWO = 2

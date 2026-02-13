from swesmith.bug_gen.procedural.base import ProceduralModifier

from swesmith.bug_gen.procedural.python.classes import (
    ClassRemoveBasesModifier,
    ClassRemoveFuncsModifier,
    ClassShuffleMethodsModifier,
)
from swesmith.bug_gen.procedural.python.control_flow import (
    ControlIfElseInvertModifier,
    ControlShuffleLinesModifier,
)
from swesmith.bug_gen.procedural.python.operations import (
    OperationBreakChainsModifier,
    OperationChangeConstantsModifier,
    OperationChangeModifier,
    OperationSwapOperandsModifier,
)
from swesmith.bug_gen.procedural.python.remove import (
    RemoveAssignModifier,
    RemoveConditionalModifier,
    RemoveLoopModifier,
    RemoveWrapperModifier,
)

MODIFIERS_PYTHON: list[ProceduralModifier] = [
    ClassRemoveBasesModifier(likelihood=0.25),
    ClassRemoveFuncsModifier(likelihood=0.15),
    ClassShuffleMethodsModifier(likelihood=0.25),
    ControlIfElseInvertModifier(likelihood=0.25),
    ControlShuffleLinesModifier(likelihood=0.25),
    RemoveAssignModifier(likelihood=0.25),
    RemoveConditionalModifier(likelihood=0.25),
    RemoveLoopModifier(likelihood=0.25),
    RemoveWrapperModifier(likelihood=0.25),
    OperationBreakChainsModifier(likelihood=0.4),
    OperationChangeConstantsModifier(likelihood=0.4),
    OperationChangeModifier(likelihood=0.4),
    OperationSwapOperandsModifier(likelihood=0.4),
]

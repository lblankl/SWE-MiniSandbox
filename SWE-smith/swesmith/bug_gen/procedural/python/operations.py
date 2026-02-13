import libcst

from swesmith.bug_gen.procedural.base import CommonPMs
from swesmith.bug_gen.procedural.python.base import PythonProceduralModifier


FLIPPED_OPERATORS = {
    libcst.Add: libcst.Subtract,
    libcst.And: libcst.Or,
    libcst.BitAnd: libcst.BitOr,
    libcst.BitAnd: libcst.BitXor,
    libcst.BitOr: libcst.BitAnd,
    libcst.BitXor: libcst.BitAnd,
    libcst.Divide: libcst.Multiply,
    libcst.Equal: libcst.NotEqual,
    libcst.FloorDivide: libcst.Modulo,
    libcst.GreaterThan: libcst.LessThan,
    libcst.GreaterThanEqual: libcst.LessThanEqual,
    libcst.In: libcst.NotIn,
    libcst.Is: libcst.IsNot,
    libcst.IsNot: libcst.Is,
    libcst.LeftShift: libcst.RightShift,
    libcst.LessThan: libcst.GreaterThan,
    libcst.LessThanEqual: libcst.GreaterThanEqual,
    libcst.Modulo: libcst.FloorDivide,
    libcst.Multiply: libcst.Divide,
    libcst.NotEqual: libcst.Equal,
    libcst.NotIn: libcst.In,
    libcst.Or: libcst.And,
    libcst.Power: libcst.Multiply,
    libcst.RightShift: libcst.LeftShift,
    libcst.Subtract: libcst.Add,
}


class OperationChangeModifier(PythonProceduralModifier):
    explanation: str = CommonPMs.OPERATION_CHANGE.explanation
    name: str = CommonPMs.OPERATION_CHANGE.name
    conditions: list = CommonPMs.OPERATION_CHANGE.conditions

    class Transformer(PythonProceduralModifier.Transformer):
        def leave_BinaryOperation(self, original_node, updated_node):
            if self.flip():
                if isinstance(updated_node.operator, (libcst.Add, libcst.Subtract)):
                    updated_node = updated_node.with_changes(
                        operator=self.parent.rand.choice(
                            [libcst.Add(), libcst.Subtract()]
                        )
                    )
                elif isinstance(
                    updated_node.operator,
                    (libcst.Multiply, libcst.Divide, libcst.FloorDivide, libcst.Modulo),
                ):
                    updated_node = updated_node.with_changes(
                        operator=self.parent.rand.choice(
                            [
                                libcst.Multiply(),
                                libcst.Divide(),
                                libcst.FloorDivide(),
                                libcst.Modulo(),
                            ]
                        )
                    )
                elif isinstance(
                    updated_node.operator, (libcst.BitAnd, libcst.BitOr, libcst.BitXor)
                ):
                    updated_node = updated_node.with_changes(
                        operator=self.parent.rand.choice(
                            [libcst.BitAnd(), libcst.BitOr(), libcst.BitXor()]
                        )
                    )
                elif isinstance(
                    updated_node.operator, (libcst.LeftShift, libcst.RightShift)
                ):
                    updated_node = updated_node.with_changes(
                        operator=self.parent.rand.choice(
                            [libcst.LeftShift(), libcst.RightShift()]
                        )
                    )
                elif isinstance(updated_node.operator, (libcst.Power, libcst.Multiply)):
                    updated_node = updated_node.with_changes(
                        operator=self.parent.rand.choice(
                            [libcst.Power(), libcst.Multiply()]
                        )
                    )
            return updated_node


class OperationFlipOperatorModifier(PythonProceduralModifier):
    explanation: str = CommonPMs.OPERATION_FLIP_OPERATOR.explanation
    name: str = CommonPMs.OPERATION_FLIP_OPERATOR.name
    conditions: list = CommonPMs.OPERATION_FLIP_OPERATOR.conditions

    class Transformer(PythonProceduralModifier.Transformer):
        def _flip_operator(self, updated_node):
            op_type = type(updated_node.operator)
            if op_type in FLIPPED_OPERATORS:
                # Create a new operator of the flipped type
                new_op_class = FLIPPED_OPERATORS[op_type]
                new_op = new_op_class()

                # Return the binary operation with the flipped operator
                return updated_node.with_changes(operator=new_op)
            return updated_node

        def leave_BinaryOperation(self, original_node, updated_node):
            return self._flip_operator(updated_node) if self.flip() else updated_node

        def leave_BooleanOperation(self, original_node, updated_node):
            return self._flip_operator(updated_node) if self.flip() else updated_node


class OperationSwapOperandsModifier(PythonProceduralModifier):
    explanation: str = CommonPMs.OPERATION_SWAP_OPERANDS.explanation
    name: str = CommonPMs.OPERATION_SWAP_OPERANDS.name
    conditions: list = CommonPMs.OPERATION_SWAP_OPERANDS.conditions

    class Transformer(PythonProceduralModifier.Transformer):
        def leave_BinaryOperation(self, original_node, updated_node):
            if self.flip():
                updated_node = updated_node.with_changes(
                    left=updated_node.right, right=updated_node.left
                )
            return updated_node

        def leave_BooleanOperation(self, original_node, updated_node):
            if self.flip():
                updated_node = updated_node.with_changes(
                    left=updated_node.right, right=updated_node.left
                )
            return updated_node


class OperationBreakChainsModifier(PythonProceduralModifier):
    explanation: str = CommonPMs.OPERATION_BREAK_CHAINS.explanation
    name: str = CommonPMs.OPERATION_BREAK_CHAINS.name
    conditions: list = CommonPMs.OPERATION_BREAK_CHAINS.conditions

    class Transformer(PythonProceduralModifier.Transformer):
        def leave_BinaryOperation(self, original_node, updated_node):
            if self.flip():
                if isinstance(updated_node.left, libcst.BinaryOperation):
                    updated_node = updated_node.with_changes(
                        left=updated_node.left.left
                    )
                elif isinstance(updated_node.right, libcst.BinaryOperation):
                    updated_node = updated_node.with_changes(
                        right=updated_node.right.right
                    )
            return updated_node


class OperationChangeConstantsModifier(PythonProceduralModifier):
    explanation: str = CommonPMs.OPERATION_CHANGE_CONSTANTS.explanation
    name: str = CommonPMs.OPERATION_CHANGE_CONSTANTS.name
    conditions: list = CommonPMs.OPERATION_CHANGE_CONSTANTS.conditions

    class Transformer(PythonProceduralModifier.Transformer):
        def leave_BinaryOperation(self, original_node, updated_node):
            if self.flip():
                if isinstance(updated_node.left, libcst.Integer):
                    try:
                        left_value = int(updated_node.left.value)
                    except ValueError:
                        left_value = int(updated_node.left.value, 16)
                    updated_node = updated_node.with_changes(
                        left=updated_node.left.with_changes(
                            value=str(left_value + self.parent.rand.choice([-1, 1]))
                        )
                    )
                if isinstance(updated_node.right, libcst.Integer):
                    try:
                        right_value = int(updated_node.right.value)
                    except ValueError:
                        right_value = int(updated_node.right.value, 16)
                    updated_node = updated_node.with_changes(
                        right=updated_node.right.with_changes(
                            value=str(right_value + self.parent.rand.choice([-1, 1]))
                        )
                    )
            return updated_node

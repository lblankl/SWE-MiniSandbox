import tree_sitter_go as tsgo

from swesmith.bug_gen.procedural.base import CommonPMs
from swesmith.bug_gen.procedural.golang.base import GolangProceduralModifier
from swesmith.constants import BugRewrite, CodeEntity
from tree_sitter import Language, Parser

GO_LANGUAGE = Language(tsgo.language())

# Mapping of Go binary operators to their alternatives
FLIPPED_OPERATORS = {
    "+": "-",
    "-": "+",
    "*": "/",
    "/": "*",
    "%": "*",  # Modulo to multiplication (common mistake)
    "<<": ">>",
    ">>": "<<",
    "&": "|",
    "|": "&",
    "^": "&",  # XOR to AND
    "==": "!=",
    "!=": "==",
    "<": ">",
    "<=": ">=",
    ">": "<",
    ">=": "<=",
    "&&": "||",
    "||": "&&",
}

# Operator groups for systematic changes
ARITHMETIC_OPS = ["+", "-", "*", "/", "%"]
BITWISE_OPS = ["&", "|", "^", "<<", ">>"]
COMPARISON_OPS = ["==", "!=", "<", "<=", ">", ">="]
LOGICAL_OPS = ["&&", "||"]


class OperationChangeModifier(GolangProceduralModifier):
    explanation: str = CommonPMs.OPERATION_CHANGE.explanation
    name: str = CommonPMs.OPERATION_CHANGE.name
    conditions: list = CommonPMs.OPERATION_CHANGE.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Apply operation changes to Go binary expressions."""
        if not self.flip():
            return None

        # Parse the code
        parser = Parser(GO_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        # Find and modify binary operations
        modified_code = self._change_operations(code_entity.src_code, tree.root_node)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _change_operations(self, source_code: str, node) -> str:
        """Recursively find and change binary operations."""
        modifications = []

        def collect_binary_ops(n):
            if n.type == "binary_expression":
                # Find the operator child
                operator_node = None
                for child in n.children:
                    if child.type in [
                        "+",
                        "-",
                        "*",
                        "/",
                        "%",
                        "<<",
                        ">>",
                        "&",
                        "|",
                        "^",
                        "==",
                        "!=",
                        "<",
                        "<=",
                        ">",
                        ">=",
                        "&&",
                        "||",
                    ]:
                        operator_node = child
                        break

                if operator_node and self.flip():
                    op = operator_node.text.decode("utf-8")
                    new_op = self._get_alternative_operator(op)
                    if new_op != op:
                        modifications.append((operator_node, new_op))

            for child in n.children:
                collect_binary_ops(child)

        collect_binary_ops(node)

        # Apply modifications from right to left to preserve positions
        modified_code = source_code
        for operator_node, new_op in sorted(
            modifications, key=lambda x: x[0].start_byte, reverse=True
        ):
            start_byte = operator_node.start_byte
            end_byte = operator_node.end_byte
            modified_code = (
                modified_code[:start_byte] + new_op + modified_code[end_byte:]
            )

        return modified_code

    def _get_alternative_operator(self, op: str) -> str:
        """Get an alternative operator from the same category."""
        if op in ARITHMETIC_OPS:
            return self.rand.choice(ARITHMETIC_OPS)
        elif op in BITWISE_OPS:
            return self.rand.choice(BITWISE_OPS)
        elif op in COMPARISON_OPS:
            return self.rand.choice(COMPARISON_OPS)
        elif op in LOGICAL_OPS:
            return self.rand.choice(LOGICAL_OPS)
        return op


class OperationFlipOperatorModifier(GolangProceduralModifier):
    explanation: str = CommonPMs.OPERATION_FLIP_OPERATOR.explanation
    name: str = CommonPMs.OPERATION_FLIP_OPERATOR.name
    conditions: list = CommonPMs.OPERATION_FLIP_OPERATOR.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Apply operator flipping to Go binary expressions."""
        if not self.flip():
            return None

        # Parse the code
        parser = Parser(GO_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        # Find and flip binary operations
        modified_code = self._flip_operators(code_entity.src_code, tree.root_node)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _flip_operators(self, source_code: str, node) -> str:
        """Recursively find and flip binary operations."""
        modifications = []

        def collect_binary_ops(n):
            if n.type == "binary_expression":
                # Find the operator child
                operator_node = None
                for child in n.children:
                    if child.type in FLIPPED_OPERATORS:
                        operator_node = child
                        break

                if operator_node and self.flip():
                    op = operator_node.text.decode("utf-8")
                    if op in FLIPPED_OPERATORS:
                        modifications.append((operator_node, FLIPPED_OPERATORS[op]))

            for child in n.children:
                collect_binary_ops(child)

        collect_binary_ops(node)

        # Apply modifications from right to left to preserve positions
        modified_code = source_code
        for operator_node, new_op in sorted(
            modifications, key=lambda x: x[0].start_byte, reverse=True
        ):
            start_byte = operator_node.start_byte
            end_byte = operator_node.end_byte
            modified_code = (
                modified_code[:start_byte] + new_op + modified_code[end_byte:]
            )

        return modified_code


class OperationSwapOperandsModifier(GolangProceduralModifier):
    explanation: str = CommonPMs.OPERATION_SWAP_OPERANDS.explanation
    name: str = CommonPMs.OPERATION_SWAP_OPERANDS.name
    conditions: list = CommonPMs.OPERATION_SWAP_OPERANDS.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Apply operand swapping to Go binary expressions."""
        if not self.flip():
            return None

        # Parse the code
        parser = Parser(GO_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        # Find and swap operands in binary operations
        modified_code = self._swap_operands(code_entity.src_code, tree.root_node)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _swap_operands(self, source_code: str, node) -> str:
        """Recursively find and swap operands in binary operations."""
        modifications = []

        def collect_binary_ops(n):
            if n.type == "binary_expression" and len(n.children) >= 3:
                if self.flip():
                    left_operand = n.children[0]
                    operator = None
                    right_operand = None

                    # Find operator and right operand
                    for i, child in enumerate(n.children[1:], 1):
                        if child.type in [
                            "+",
                            "-",
                            "*",
                            "/",
                            "%",
                            "<<",
                            ">>",
                            "&",
                            "|",
                            "^",
                            "==",
                            "!=",
                            "<",
                            "<=",
                            ">",
                            ">=",
                            "&&",
                            "||",
                        ]:
                            operator = child
                            if i + 1 < len(n.children):
                                right_operand = n.children[i + 1]
                            break

                    if left_operand and operator and right_operand:
                        modifications.append((n, left_operand, operator, right_operand))

            for child in n.children:
                collect_binary_ops(child)

        collect_binary_ops(node)

        # Apply modifications from right to left to preserve positions
        modified_code = source_code
        for expr_node, left, op, right in sorted(
            modifications, key=lambda x: x[0].start_byte, reverse=True
        ):
            start_byte = expr_node.start_byte
            end_byte = expr_node.end_byte

            left_text = left.text.decode("utf-8")
            op_text = op.text.decode("utf-8")
            right_text = right.text.decode("utf-8")

            # Swap the operands
            new_expr = f"{right_text} {op_text} {left_text}"
            modified_code = (
                modified_code[:start_byte] + new_expr + modified_code[end_byte:]
            )

        return modified_code


class OperationBreakChainsModifier(GolangProceduralModifier):
    explanation: str = CommonPMs.OPERATION_BREAK_CHAINS.explanation
    name: str = CommonPMs.OPERATION_BREAK_CHAINS.name
    conditions: list = CommonPMs.OPERATION_BREAK_CHAINS.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Apply chain breaking to Go binary expressions."""
        if not self.flip():
            return None

        # Parse the code
        parser = Parser(GO_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        # Find and break chains in binary operations
        modified_code = self._break_chains(code_entity.src_code, tree.root_node)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _break_chains(self, source_code: str, node) -> str:
        """Recursively find and break chains in binary operations."""
        modifications = []

        def collect_binary_ops(n):
            if n.type == "binary_expression" and self.flip():
                # Look for nested binary expressions
                left_operand = n.children[0] if n.children else None
                right_operand = None

                # Find the right operand
                for i, child in enumerate(n.children[1:], 1):
                    if child.type not in [
                        "+",
                        "-",
                        "*",
                        "/",
                        "%",
                        "<<",
                        ">>",
                        "&",
                        "|",
                        "^",
                        "==",
                        "!=",
                        "<",
                        "<=",
                        ">",
                        ">=",
                        "&&",
                        "||",
                    ]:
                        right_operand = child
                        break

                # If left operand is a binary expression, replace with its left operand
                if left_operand and left_operand.type == "binary_expression":
                    inner_left = (
                        left_operand.children[0] if left_operand.children else None
                    )
                    if inner_left:
                        modifications.append((n, inner_left))
                # If right operand is a binary expression, replace with its right operand
                elif right_operand and right_operand.type == "binary_expression":
                    inner_right = None
                    for child in reversed(right_operand.children):
                        if child.type not in [
                            "+",
                            "-",
                            "*",
                            "/",
                            "%",
                            "<<",
                            ">>",
                            "&",
                            "|",
                            "^",
                            "==",
                            "!=",
                            "<",
                            "<=",
                            ">",
                            ">=",
                            "&&",
                            "||",
                        ]:
                            inner_right = child
                            break
                    if inner_right:
                        modifications.append((n, inner_right))

            for child in n.children:
                collect_binary_ops(child)

        collect_binary_ops(node)

        # Apply modifications from right to left to preserve positions
        modified_code = source_code
        for expr_node, replacement in sorted(
            modifications, key=lambda x: x[0].start_byte, reverse=True
        ):
            start_byte = expr_node.start_byte
            end_byte = expr_node.end_byte
            replacement_text = replacement.text.decode("utf-8")
            modified_code = (
                modified_code[:start_byte] + replacement_text + modified_code[end_byte:]
            )

        return modified_code


class OperationChangeConstantsModifier(GolangProceduralModifier):
    explanation: str = CommonPMs.OPERATION_CHANGE_CONSTANTS.explanation
    name: str = CommonPMs.OPERATION_CHANGE_CONSTANTS.name
    conditions: list = CommonPMs.OPERATION_CHANGE_CONSTANTS.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Apply constant changes to Go binary expressions."""
        if not self.flip():
            return None

        # Parse the code
        parser = Parser(GO_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        # Find and modify constants in binary operations
        modified_code = self._change_constants(code_entity.src_code, tree.root_node)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _change_constants(self, source_code: str, node) -> str:
        """Recursively find and modify constants in binary operations."""
        modifications = []

        def collect_binary_ops(n):
            if n.type == "binary_expression":
                for child in n.children:
                    if child.type == "int_literal" and self.flip():
                        try:
                            value = int(child.text.decode("utf-8"))
                            new_value = value + self.rand.choice([-1, 1])
                            modifications.append((child, str(new_value)))
                        except ValueError:
                            pass  # Skip invalid integer literals
                    elif child.type == "float_literal" and self.flip():
                        try:
                            value = float(child.text.decode("utf-8"))
                            delta = self.rand.choice([-0.1, 0.1, -1.0, 1.0])
                            new_value = value + delta
                            modifications.append((child, str(new_value)))
                        except ValueError:
                            pass  # Skip invalid float literals

            for child in n.children:
                collect_binary_ops(child)

        collect_binary_ops(node)

        # Apply modifications from right to left to preserve positions
        modified_code = source_code
        for const_node, new_value in sorted(
            modifications, key=lambda x: x[0].start_byte, reverse=True
        ):
            start_byte = const_node.start_byte
            end_byte = const_node.end_byte
            modified_code = (
                modified_code[:start_byte] + new_value + modified_code[end_byte:]
            )

        return modified_code

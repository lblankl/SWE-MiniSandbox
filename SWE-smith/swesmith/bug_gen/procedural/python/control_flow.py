import libcst

from swesmith.bug_gen.procedural.base import CommonPMs
from swesmith.bug_gen.procedural.python.base import PythonProceduralModifier


class ControlIfElseInvertModifier(PythonProceduralModifier):
    explanation: str = CommonPMs.CONTROL_IF_ELSE_INVERT.explanation
    name: str = CommonPMs.CONTROL_IF_ELSE_INVERT.name
    conditions: list = CommonPMs.CONTROL_IF_ELSE_INVERT.conditions
    min_complexity: int = 5

    class Transformer(PythonProceduralModifier.Transformer):
        def leave_If(
            self, original_node: libcst.If, updated_node: libcst.If
        ) -> libcst.If:
            if not self.flip():
                return updated_node

            # Only proceed if there's an else branch to swap with
            if not updated_node.orelse:
                return updated_node

            # We need to handle standard else blocks
            if isinstance(updated_node.orelse, libcst.Else):
                # Store the original bodies
                if_body = updated_node.body
                else_body = updated_node.orelse.body

                # Create a new else clause with the original if body
                new_else = libcst.Else(
                    body=if_body,
                    whitespace_before_colon=updated_node.orelse.whitespace_before_colon,
                )

                # Return a new If statement with swapped bodies
                return updated_node.with_changes(body=else_body, orelse=new_else)

            # Skip elif cases for now
            return updated_node


class ControlShuffleLinesModifier(PythonProceduralModifier):
    explanation: str = CommonPMs.CONTROL_SHUFFLE_LINES.explanation
    name: str = CommonPMs.CONTROL_SHUFFLE_LINES.name
    conditions: list = CommonPMs.CONTROL_SHUFFLE_LINES.conditions
    max_complexity: int = 10

    class Transformer(PythonProceduralModifier.Transformer):
        def leave_FunctionDef(
            self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
        ) -> libcst.FunctionDef:
            # Skip modification if random check fails
            if not self.flip():
                return updated_node

            # Make sure we're working with an indented block
            if not isinstance(updated_node.body, libcst.IndentedBlock):
                return updated_node

            # Get the body statements
            body = list(updated_node.body.body)

            # Don't shuffle if there are fewer than 2 statements
            if len(body) < 2:
                return updated_node

            # Create a shuffled copy of the statements
            shuffled_body = body.copy()
            self.parent.rand.shuffle(shuffled_body)

            # Create a new indented block with the shuffled statements
            new_body = libcst.IndentedBlock(
                body=tuple(shuffled_body),
                indent=updated_node.body.indent,
                header=updated_node.body.header,
                footer=updated_node.body.footer,
            )

            # Return the updated function with the new body
            return updated_node.with_changes(body=new_body)

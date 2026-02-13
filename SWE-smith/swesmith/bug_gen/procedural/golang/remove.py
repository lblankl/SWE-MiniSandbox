import tree_sitter_go as tsgo

from swesmith.bug_gen.procedural.base import CommonPMs
from swesmith.bug_gen.procedural.golang.base import GolangProceduralModifier
from swesmith.constants import BugRewrite, CodeEntity
from tree_sitter import Language, Parser

GO_LANGUAGE = Language(tsgo.language())


class RemoveLoopModifier(GolangProceduralModifier):
    explanation: str = CommonPMs.REMOVE_LOOP.explanation
    name: str = CommonPMs.REMOVE_LOOP.name
    conditions: list = CommonPMs.REMOVE_LOOP.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Remove loop statements from the Go code."""
        if not self.flip():
            return None

        # Parse the code
        parser = Parser(GO_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        # Find and remove loop statements
        modified_code = self._remove_loops(code_entity.src_code, tree.root_node)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _remove_loops(self, source_code: str, node) -> str:
        """Recursively find and remove loop statements."""
        removals = []

        def collect_loops(n):
            if n.type == "for_statement":
                if self.flip():
                    removals.append(n)
            for child in n.children:
                collect_loops(child)

        collect_loops(node)

        if not removals:
            return source_code

        # Apply removals from end to start to preserve byte offsets
        modified_source = source_code
        for loop_node in reversed(removals):
            start_byte = loop_node.start_byte
            end_byte = loop_node.end_byte

            # Remove the entire loop statement
            modified_source = modified_source[:start_byte] + modified_source[end_byte:]

        return modified_source


class RemoveConditionalModifier(GolangProceduralModifier):
    explanation: str = CommonPMs.REMOVE_CONDITIONAL.explanation
    name: str = CommonPMs.REMOVE_CONDITIONAL.name
    conditions: list = CommonPMs.REMOVE_CONDITIONAL.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Remove conditional statements from the Go code."""
        if not self.flip():
            return None

        # Parse the code
        parser = Parser(GO_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        # Find and remove conditional statements
        modified_code = self._remove_conditionals(code_entity.src_code, tree.root_node)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _remove_conditionals(self, source_code: str, node) -> str:
        """Recursively find and remove conditional statements."""
        removals = []

        def collect_conditionals(n):
            if n.type == "if_statement":
                if self.flip():
                    removals.append(n)
            for child in n.children:
                collect_conditionals(child)

        collect_conditionals(node)

        if not removals:
            return source_code

        # Apply removals from end to start to preserve byte offsets
        modified_source = source_code
        for if_node in reversed(removals):
            start_byte = if_node.start_byte
            end_byte = if_node.end_byte

            # Remove the entire if statement
            modified_source = modified_source[:start_byte] + modified_source[end_byte:]

        return modified_source


class RemoveAssignModifier(GolangProceduralModifier):
    explanation: str = CommonPMs.REMOVE_ASSIGNMENT.explanation
    name: str = CommonPMs.REMOVE_ASSIGNMENT.name
    conditions: list = CommonPMs.REMOVE_ASSIGNMENT.conditions

    def modify(self, code_entity: CodeEntity) -> BugRewrite:
        """Remove assignment statements from the Go code."""
        if not self.flip():
            return None

        # Parse the code
        parser = Parser(GO_LANGUAGE)
        tree = parser.parse(bytes(code_entity.src_code, "utf8"))

        # Find and remove assignment statements
        modified_code = self._remove_assignments(code_entity.src_code, tree.root_node)

        if modified_code == code_entity.src_code:
            return None

        return BugRewrite(
            rewrite=modified_code,
            explanation=self.explanation,
            strategy=self.name,
        )

    def _remove_assignments(self, source_code: str, node) -> str:
        """Recursively find and remove assignment statements."""
        removals = []

        def collect_assignments(n):
            # Go assignment types include:
            # - assignment_statement (=)
            # - short_var_declaration (:=)
            # - inc_statement (++)
            # - dec_statement (--)
            if n.type in [
                "assignment_statement",
                "short_var_declaration",
                "inc_statement",
                "dec_statement",
            ]:
                if self.flip():
                    removals.append(n)
            for child in n.children:
                collect_assignments(child)

        collect_assignments(node)

        if not removals:
            return source_code

        # Apply removals from end to start to preserve byte offsets
        modified_source = source_code
        for assign_node in reversed(removals):
            start_byte = assign_node.start_byte
            end_byte = assign_node.end_byte

            # Remove the entire assignment statement
            modified_source = modified_source[:start_byte] + modified_source[end_byte:]

        return modified_source

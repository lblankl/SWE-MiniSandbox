import libcst

from abc import ABC
from swesmith.bug_gen.procedural.base import ProceduralModifier
from swesmith.constants import BugRewrite, CodeEntity


class PythonProceduralModifier(ProceduralModifier, ABC):
    """Base class for Python-specific procedural modifications using LibCST."""

    class Transformer(libcst.CSTTransformer):
        """Nested LibCST transformer that has access to parent modifier."""

        def __init__(self, parent_modifier):
            self.parent = parent_modifier
            super().__init__()

        def flip(self) -> bool:
            """Delegate to parent's flip method."""
            return self.parent.flip()

    def modify(self, code_entity: CodeEntity) -> BugRewrite | None:
        try:
            module = libcst.parse_module(code_entity.src_code)
        except libcst.ParserSyntaxError:
            # Failed to parse code - syntax errors, malformed code, etc.
            return None

        changed = False
        transformer = self.Transformer(self)

        try:
            for _ in range(self.max_attempts):
                modified = module.visit(transformer)
                if module.code != modified.code:
                    changed = True
                    break
        except (AttributeError, TypeError, ValueError):
            return None

        if not changed:
            return None

        return BugRewrite(
            rewrite=modified.code,
            explanation=self.explanation,
            cost=0.0,
            strategy=self.name,
        )

from __future__ import annotations

import ast
import contextlib
import io
from typing import Any

from docutils import nodes
from sphinx.util.docutils import SphinxDirective


def _split_last_expr(tree: ast.Module) -> tuple[ast.Module | None, ast.Expression | None]:
    """Return the module without the last Expr node and the last Expr (if any)."""
    if not tree.body:
        return tree, None
    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        prefix = ast.Module(body=tree.body[:-1], type_ignores=tree.type_ignores)
        expr = ast.Expression(body=last.value)
        return prefix, expr
    return tree, None


class JupyterExecuteDirective(SphinxDirective):
    """Lightweight code-execution directive with persistent namespace per document."""

    has_content = True

    def run(self) -> list[nodes.Node]:
        if not self.content:
            return []

        code = "\n".join(self.content)
        literal = nodes.literal_block(code, code)
        literal["language"] = "python"

        env = self.env
        namespace_store = env.temp_data.setdefault("jupyter_execute_namespaces", {})
        ns = namespace_store.setdefault(env.docname, {"__name__": f"jupyter_exec_{env.docname}"})

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = self._execute(code, ns)

        output_nodes: list[nodes.Node] = [literal]

        stdout_text = stdout.getvalue()
        if stdout_text:
            out_block = nodes.literal_block(stdout_text, stdout_text)
            out_block["classes"].append("jupyter-output")
            output_nodes.append(out_block)

        if result is not None:
            result_text = repr(result)
            res_block = nodes.literal_block(result_text, result_text)
            res_block["classes"].append("jupyter-result")
            output_nodes.append(res_block)

        return output_nodes

    def _execute(self, code: str, namespace: dict[str, Any]) -> Any:
        filename = self.env.doc2path(self.env.docname)
        tree = ast.parse(code, filename=filename, mode="exec")
        tree = ast.fix_missing_locations(tree)
        prefix, expr = _split_last_expr(tree)

        if prefix and prefix.body:
            exec(compile(prefix, filename, "exec"), namespace, namespace)

        if expr is not None:
            compiled = compile(expr, filename, "eval")
            return eval(compiled, namespace, namespace)

        if not prefix or not prefix.body:
            if expr is None:
                exec(compile(tree, filename, "exec"), namespace, namespace)
        return None


def setup(app):
    app.add_directive("jupyter-execute", JupyterExecuteDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}

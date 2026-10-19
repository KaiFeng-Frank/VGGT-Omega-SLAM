"""(a) The ported get_similarity / mean_top_quarter equal VGGT_SPARK's (numerically and by AST)."""
import ast
import inspect
import textwrap

import numpy as np
import pytest
torch = pytest.importorskip("torch", reason="attention integration tests need PyTorch (CPU is sufficient)")

fork = pytest.importorskip("vggt.models.aggregator", reason="needs pinned VGGT_SPARK sources; run tools/setup_sources.py")
from vos import omega_backend as ob


def _qk(tokens_per_img, heads=4, head_dim=16, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    n = 2 * tokens_per_img
    q = torch.randn(1, heads, n, head_dim, generator=g)
    k = torch.randn(1, heads, n, head_dim, generator=g)
    return k.to(dtype), (q * head_dim ** -0.5).to(dtype)


@pytest.mark.parametrize("token_offset,tokens_per_img", [(5, 5 + 40), (17, 17 + 60), (17, 17 + 1032)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_get_similarity_equals_fork(token_offset, tokens_per_img, seed, capsys):
    k, q = _qk(tokens_per_img, seed=seed)
    ours = ob.get_similarity(k, q, token_offset=token_offset)
    theirs = fork.get_similarity(k, q, token_offset=token_offset)
    capsys.readouterr()  # the fork prints a debug line
    assert ours == theirs  # identical operations -> bit-identical


def test_get_similarity_equals_fork_bf16(capsys):
    k, q = _qk(17 + 50, dtype=torch.bfloat16)
    assert ob.get_similarity(k, q, token_offset=17) == fork.get_similarity(k, q, token_offset=17)
    capsys.readouterr()


def test_mean_top_quarter_equals_fork():
    rng = np.random.default_rng(0)
    for n in (7, 64, 1033):
        arr = rng.normal(size=n).astype(np.float32)
        assert ob.mean_top_quarter(arr) == fork.mean_top_quarter(arr)


def _normalised_ast(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    func = tree.body[0]

    class DropPrints(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "print":
                return None
            return node

    func = DropPrints().visit(func)
    # docstrings/comments carry no behaviour; drop a leading docstring if present
    if func.body and isinstance(func.body[0], ast.Expr) and isinstance(getattr(func.body[0], "value", None), ast.Constant) \
            and isinstance(func.body[0].value.value, str):
        func.body = func.body[1:]
    func.name = "f"
    return ast.dump(func)


@pytest.mark.parametrize("name", ["get_similarity", "mean_top_quarter"])
def test_port_is_verbatim_modulo_debug_prints(name):
    assert _normalised_ast(getattr(ob, name)) == _normalised_ast(getattr(fork, name))


def test_scaled_ratio_makes_upstream_test_equal_to_raw_vs_tau():
    from vos.omega_backend import UPSTREAM_TEST

    for tau in (0.3, 0.8, 0.95, 1.7):
        for raw in np.linspace(0.05, 2.5, 97):
            scaled = UPSTREAM_TEST * float(raw) / tau
            assert (scaled < 0.95) == (raw < tau)

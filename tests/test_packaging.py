# Copyright © 2026 Apple Inc.

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _setup_constants():
    tree = ast.parse((ROOT / "setup.py").read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return constants


def test_custom_kernel_build_dependencies_match_setup_versions():
    constants = _setup_constants()
    pyproject = (ROOT / "pyproject.toml").read_text()
    readme = (ROOT / "README.md").read_text()
    benchmark_docs = (ROOT / "docs" / "glm52-prefill-benchmark.md").read_text()
    native_kernel_docs = (ROOT / "docs" / "native-kernels.md").read_text()
    vendored_kernel_docs = (
        ROOT / "mlx_lm" / "custom_kernels" / "glm_moe_dsa" / "README.md"
    ).read_text()

    assert f'"nanobind=={constants["NANOBIND_VERSION"]}"' in pyproject
    expected_mlx = "\"mlx=={}; platform_system == 'Darwin'\"".format(
        constants["MIN_MLX_VERSION"]
    )
    assert expected_mlx in pyproject
    assert f"MLX {constants['MIN_MLX_VERSION']}" in readme
    assert f"mlx=={constants['MIN_MLX_VERSION']}" in readme
    assert f"mlx>={constants['MIN_MLX_VERSION']}" in benchmark_docs
    assert f"MLX {constants['MIN_MLX_VERSION']}" in native_kernel_docs
    assert f"nanobind {constants['NANOBIND_VERSION']}" in native_kernel_docs
    assert f"MLX {constants['MIN_MLX_VERSION']}" in vendored_kernel_docs
    assert f"nanobind {constants['NANOBIND_VERSION']}" in vendored_kernel_docs
    assert ".venv/bin/python" in vendored_kernel_docs

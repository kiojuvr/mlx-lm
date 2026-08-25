# GLM MoE DSA Custom Kernels

This package vendors the GLM MoE DSA native custom kernels from oMLX so this
GLM-5.2-focused fork can build and run the native DSA indexer, sparse MLA,
q8/q4 V-up, routed-MoE weighted reduction and affine gate/up prefill fusion,
and experimental q4 q projection routes without a runtime dependency on the
changing oMLX repository.

Vendored source:

- Source project: oMLX
- Commit observed during vendoring: `eca4a7c`
- Source path: `omlx/custom_kernels/glm_moe_dsa`
- License: Apache-2.0; see `LICENSE` in this directory.

From the repository root, build with:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
MLX_LM_WITH_CUSTOM_KERNEL=1 \
uv pip install --python .venv/bin/python --no-deps -e .
```

The build requires MLX 0.32.2 and nanobind 2.15.0. See
[Native GLM Kernels](../../../docs/native-kernels.md) for Xcode selection,
Metal Toolchain verification, and the full build and smoke-test procedure.

The CMake target still emits an `omlx_glm_kernels.metallib` file name because
the vendored C++ sources look up that library name internally.

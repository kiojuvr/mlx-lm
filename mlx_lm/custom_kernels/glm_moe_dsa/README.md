# GLM MoE DSA Custom Kernels

This package vendors the GLM MoE DSA native custom kernels from oMLX so this
GLM-5.2-focused fork can build and run the native DSA indexer, sparse MLA, and
q8 V-up routes without a runtime dependency on the changing oMLX repository.

Vendored source:

- Repository: `/Volumes/USB-SSD-2/omlx`
- Commit observed during vendoring: `eca4a7c`
- Source path: `omlx/custom_kernels/glm_moe_dsa`
- License: Apache-2.0; see `LICENSE` in this directory.

Build with:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
MLX_LM_WITH_CUSTOM_KERNEL=1 \
uv pip install --python /Users/kioju/.venvs/mlx-glm52/bin/python --no-deps -e .
```

The CMake target still emits an `omlx_glm_kernels.metallib` file name because
the vendored C++ sources look up that library name internally.

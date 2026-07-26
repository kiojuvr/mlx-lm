# Copyright © 2024 Apple Inc.

import os
import sys
from pathlib import Path

from setuptools import setup

package_dir = Path(__file__).parent / "mlx_lm"
sys.path.append(str(package_dir))

from _version import __version__

MIN_MLX_VERSION = "0.32.0"
NANOBIND_VERSION = "2.13.0"
CUSTOM_KERNEL_FLAG = "--with-custom-kernel"
TRUTHY = {"1", "true", "yes", "on"}
DEFAULT_CUSTOM_KERNEL_DEPLOYMENT_TARGET = "15.0"


def _with_custom_kernel() -> bool:
    if CUSTOM_KERNEL_FLAG in sys.argv:
        sys.argv.remove(CUSTOM_KERNEL_FLAG)
        return True
    return (
        os.environ.get("MLX_LM_WITH_CUSTOM_KERNEL", "").strip().lower() in TRUTHY
        or os.environ.get("OMLX_WITH_CUSTOM_KERNEL", "").strip().lower() in TRUTHY
    )


def _custom_kernel_build_kwargs() -> dict:
    if not _with_custom_kernel():
        return {}
    if sys.platform != "darwin":
        raise RuntimeError("GLM custom kernels can only be built on macOS.")

    target = (
        os.environ.get("MLX_LM_CUSTOM_KERNEL_DEPLOYMENT_TARGET")
        or os.environ.get("MACOSX_DEPLOYMENT_TARGET")
        or DEFAULT_CUSTOM_KERNEL_DEPLOYMENT_TARGET
    )
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", target)
    cmake_args = os.environ.get("CMAKE_ARGS", "").strip()
    extra_cmake_args = []
    if "CMAKE_OSX_DEPLOYMENT_TARGET" not in cmake_args:
        extra_cmake_args.append(f"-DCMAKE_OSX_DEPLOYMENT_TARGET={target}")
    if "Python_EXECUTABLE" not in cmake_args:
        extra_cmake_args.append(f"-DPython_EXECUTABLE={sys.executable}")
    if extra_cmake_args:
        appended_args = " ".join(extra_cmake_args)
        os.environ["CMAKE_ARGS"] = (
            f"{cmake_args} {appended_args}".strip() if cmake_args else appended_args
        )

    from mlx import extension

    return {
        "ext_modules": [
            extension.CMakeExtension(
                "mlx_lm.custom_kernels.glm_moe_dsa._ext",
                sourcedir="mlx_lm/custom_kernels/glm_moe_dsa/csrc",
            ),
        ],
        "cmdclass": {"build_ext": extension.CMakeBuild},
    }


setup(
    name="mlx-lm",
    version=__version__,
    description="LLMs with MLX and the Hugging Face Hub",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author_email="mlx@group.apple.com",
    author="MLX Contributors",
    url="https://github.com/ml-explore/mlx-lm",
    license="MIT",
    install_requires=[
        f"mlx>={MIN_MLX_VERSION}; platform_system == 'Darwin'",
        "numpy",
        "transformers>=5.7.0",
        "sentencepiece",
        "protobuf",
        "pyyaml",
        "jinja2",
        "Pillow",
    ],
    packages=[
        "mlx_lm",
        "mlx_lm.models",
        "mlx_lm.quant",
        "mlx_lm.tuner",
        "mlx_lm.tool_parsers",
        "mlx_lm.chat_templates",
        "mlx_lm.custom_kernels",
        "mlx_lm.custom_kernels.glm_moe_dsa",
    ],
    package_data={
        "mlx_lm.custom_kernels.glm_moe_dsa": [
            "*.metallib",
            "*.dylib",
            "*.so",
            "LICENSE",
            "README.md",
        ],
    },
    python_requires=">=3.8",
    extras_require={
        "test": ["datasets", "lm-eval"],
        "train": ["datasets", "tqdm"],
        "evaluate": ["lm-eval", "tqdm"],
        "custom-kernel": ["cmake>=3.27", f"nanobind=={NANOBIND_VERSION}"],
        "cuda13": [f"mlx[cuda13]>={MIN_MLX_VERSION}"],
        "cuda12": [f"mlx[cuda12]>={MIN_MLX_VERSION}"],
        "cpu": [f"mlx[cpu]>={MIN_MLX_VERSION}"],
    },
    entry_points={
        "console_scripts": [
            "mlx_lm = mlx_lm.cli:main",
            "mlx_lm.awq = mlx_lm.quant.awq:main",
            "mlx_lm.dwq = mlx_lm.quant.dwq:main",
            "mlx_lm.dynamic_quant = mlx_lm.quant.dynamic_quant:main",
            "mlx_lm.gptq = mlx_lm.quant.gptq:main",
            "mlx_lm.benchmark = mlx_lm.benchmark:main",
            "mlx_lm.cache_prompt = mlx_lm.cache_prompt:main",
            "mlx_lm.chat = mlx_lm.chat:main",
            "mlx_lm.convert = mlx_lm.convert:main",
            "mlx_lm.evaluate = mlx_lm.evaluate:main",
            "mlx_lm.fuse = mlx_lm.fuse:main",
            "mlx_lm.generate = mlx_lm.generate:main",
            "mlx_lm.lora = mlx_lm.lora:main",
            "mlx_lm.perplexity = mlx_lm.perplexity:main",
            "mlx_lm.server = mlx_lm.server:main",
            "mlx_lm.share = mlx_lm.share:main",
            "mlx_lm.manage = mlx_lm.manage:main",
            "mlx_lm.upload = mlx_lm.upload:main",
        ]
    },
    **_custom_kernel_build_kwargs(),
)

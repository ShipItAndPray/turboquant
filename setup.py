from setuptools import setup, find_packages

setup(
    name="turboquant",
    version="0.2.0",
    description="Compress Any LLM Up to 6x in One Command — GGUF, GPTQ, AWQ with Ollama/vLLM targets, HuggingFace publishing, and quality eval",
    url="https://github.com/ShipItAndPray/turboquant",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="TurboQuant",
    py_modules=["turboquant"],
    python_requires=">=3.9",
    install_requires=[
        "transformers>=4.36.0",
        "huggingface-hub>=0.20.0",
        "torch>=2.0.0",
    ],
    extras_require={
        "gguf": ["llama-cpp-python>=0.2.0"],
        "gptq": ["auto-gptq>=0.7.0", "datasets>=2.14.0"],
        "awq": ["autoawq>=0.2.0"],
        "all": ["llama-cpp-python>=0.2.0", "auto-gptq>=0.7.0", "autoawq>=0.2.0", "datasets>=2.14.0"],
    },
    entry_points={
        "console_scripts": [
            "turboquant=turboquant:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
)

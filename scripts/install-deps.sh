#!/bin/sh

# installing guidance for deps
pip install pytest guidance huggingface_hub tokenizers jsonschema maturin[zig] \
    torch transformers==4.52.1 bitsandbytes ipython psutil mypy llama_cpp_python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip uninstall -y guidance

# print out versions
rustc --version
python --version

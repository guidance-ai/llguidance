#!/bin/sh

# installing guidance for deps
pip install pytest guidance huggingface_hub tokenizers jsonschema maturin[zig] \
    torch transformers==4.52.1 bitsandbytes ipython psutil mypy llama_cpp_python
pip uninstall -y guidance

# print out versions
rustc --version
python --version

import llguidance.llamacpp
import llama_cpp
import os
import requests # type: ignore
from typing import Any

def get_llama_vocab_file(pytestconfig: Any) -> str:
    url = "https://raw.githubusercontent.com/ggml-org/llama.cpp/f4ab2a41476600a98067a9474ea8f9e6db41bcfa/models/ggml-vocab-llama-bpe.gguf"
    cache_dir = pytestconfig.cache.makedir("llama_vocab")
    file_name = "vocab.gguf"
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(file_path):
        r = requests.get(url)
        r.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(r.content)

    return file_path


def test_llama_cpp(pytestconfig: Any) -> None:
    filepath = get_llama_vocab_file(pytestconfig)
    p = llama_cpp.llama_model_default_params()
    p.vocab_only = True
    model = llama_cpp.llama_model_load_from_file(filepath.encode(), p)
    assert model is not None
    vocab = llama_cpp.llama_model_get_vocab(model)
    assert vocab is not None
    llt = llguidance.llamacpp.lltokenizer_from_vocab(vocab)
    for s in [
            "Hello world!", "Hello world! こんにちは世界！", "wave 👋", "heart 👋💖",
            "1`a`b`c`d`e`f`g`h`i"
    ]:
        toks = llt.tokenize_str(s)
        print(llt.dbg_tokens(toks))
        assert llt.decode_str(toks) == s
    toks = llt.tokenize_bytes(b"\x8b")
    print(llt.dbg_tokens(toks))
    print(toks)
    assert len(toks) == 1
    assert llt.decode_bytes(toks) == b"\x8b"

import llguidance.llamacpp
import llama_cpp
from huggingface_hub import hf_hub_download


def test_llama_cpp() -> None:
    filepath = hf_hub_download(
        repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    )
    p = llama_cpp.llama_model_params()
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

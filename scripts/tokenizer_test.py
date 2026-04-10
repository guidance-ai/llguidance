import llguidance as llg
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer_json_path = hf_hub_download(repo_id=model_name, filename="tokenizer.json")
with open(tokenizer_json_path, "r") as f:
    llg_tokenizer = llg.LLTokenizer(f.read())


def tokenize(s: str) -> list[int]:
    r = tokenizer(s, return_tensors="pt", add_special_tokens=False)
    return r["input_ids"][0].tolist()


def test_tokenize(s: str):
    t = tokenize(s)
    dbg = llg_tokenizer.dbg_tokens(t)
    print(f"{repr(s)} -> {t} {dbg}")


test_tokenize("additional_properties")
test_tokenize('{"_properties')
test_tokenize("<|end_of_text|>_properties")
test_tokenize("_properties")
test_tokenize("\u0002_properties")

"""markdown
# Toolcalls

This document explains how to design and parse toolcall outputs produced by different LLMs. Models often use distinct output formats, so you can either:

- build a model-specific grammar tailored to the model's expected format (recommended for best accuracy), or
- use prompt-injection to coerce self-hosted models into a consistent, universal format.

Why tailor a grammar to each model format
- Models are often trained or fine-tuned on particular output styles and token sequences; they learn to continue or reproduce those formats reliably.
- If you ask a model to produce a format it was not trained to emit, the request becomes out-of-distribution: the model may produce malformed JSON, partial outputs, or unrelated text.
- Model-specific grammars align your parser with the model's native tendencies, greatly reducing parsing errors and increasing robustness.
- Prompt-injection can work but requires explicit, repeatable instructions and verification; when you cannot control training or prompt fidelity, a model-specific grammar is the safer choice.

Practical guidance
- Prefer model-specific grammars for production integrations where reliability matters.
- If using prompt-injection to standardize output, include clear examples, strict instructions, and validate outputs with test prompts.
- Always validate and sanitize model outputs before acting on them, and add fallbacks for malformed or incomplete toolcall outputs.

The sections below show examples for building grammars for model-specific outputs and discuss how to approach a unified interface.

## Example tool definitions

Consider the following example tool definitions (Python-style):

```py
tools = [
    {
        "name": "add",
        "description": "Adds two floating-point numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "The first number."},
                "y": {"type": "number", "description": "The second number."},
            },
            "required": ["x", "y"],
        },
        "returns": {"type": "number", "description": "The sum."},
    },
    {
        "name": "mul",
        "description": "Multiplies two floating-point numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "The first number."},
                "y": {"type": "number", "description": "The second number."},
            },
            "required": ["x", "y"],
        },
        "returns": {"type": "number", "description": "The product."},
    },
]

funcs = [
    {
        "type": "object",
        "properties": {
            "name": {"const": tool["name"]},
            "arguments": tool["parameters"],
        },
        "required": ["name", "arguments"],
    }
    for tool in tools
]
```

## Model-specific grammars

When targeting one model or a model family, match its native toolcall format — it's the most reliable.

Simple instructions
- Inspect a few real outputs (for single and multiple toolcalls) to capture delimiters, wrappers, and spacing.
- Define grammar rules that mirror exact wrappers and the JSON shape (single vs. multi-toolcall).

### Phi-4-Mini (example)

Phi-4-Mini produces toolcalls as a single JSON array wrapped by special delimiters. Example output:

```
<|tool_call|>[{"name": "add", "arguments": {"x": 123345432, "y": 4563464236}}, {"name": "mul", "arguments": {"x": 874284, "y": 912429}}]<|/tool_call|>
```

A corresponding JSON schema and grammar might look like:

```py
schema = {
  "type": "array",
  "items": {"anyOf": funcs},
  "minItems": 1,
}

grammar = f"""
start: <|tool_call|> tools <|/tool_call|>
tools: %json {json.dumps(schema)}
"""
```

### Qwen-3 (example)

Qwen-3 may emit multiple JSON objects as separate lines, each wrapped in a toolcall block. Example output:

```
<tool_call>
{"name": "add", "arguments": {"x": 123345432, "y": 4563464236}}
</tool_call>
<tool_call>
{"name": "mul", "arguments": {"x": 874284, "y": 912429}}
</tool_call>
```

One way to express that with a grammar is:

```py
grammar = f"""
start: /(tool)+/
tool: <tool_call> "\n" (func1 | func2) "\n" <|/tool_call|>
func1: %json {json.dumps(funcs[0])}
func2: %json {json.dumps(funcs[1])}
"""
```

Note: the exact grammar depends on the parser you use and the precise output formatting of the model.

## Unified interface (prompt-injection)

When integrating with multiple models, maintaining individual grammars for each can be cumbersome and error-prone. 
A unified interface streamlines this process by injecting clear, custom output instructions directly into the prompt. 
This approach guides models to emit toolcall outputs in a consistent, predictable structure—regardless of their native tendencies. 
With a standardized output format, you can define a single grammar for parsing, simplifying downstream integration and reducing maintenance overhead.

Below is an example used in [VLLM](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_phi4_mini.jinja):

```py
system_prompt += r"""
In addition to plain text responses, you can choose to call one or more of the provided functions.

Use the following rule to decide when to call a function:
  * if the response can be generated from your internal knowledge (e.g., as in the case of queries like "What is the capital of Poland?"), do so
  * if you need external information that can be obtained by calling one or more of the provided functions, generate a function calls

If you decide to call functions:
  * prefix function calls with functools marker (no closing marker required)
  * all function calls should be generated in a single JSON list formatted as functools[{"name": [function name], "arguments": [function arguments as JSON]}, ...]
  * follow the provided JSON schema. Do not hallucinate arguments or values. Do to blindly copy values from the provided samples
  * respect the argument type formatting. E.g., if the type is number and format is float, write value 7 as 7.0
  * make sure you pick the right functions that match the user intent
"""

system_prompt += json.dumps(funcs, indent=2)
```

The injected prompt directs the model to produce toolcalls in the following standardized format:

```
functools[{"name": [function name], "arguments": [function arguments as JSON]}, ...]
```

This structure ensures that all function calls are grouped within a single JSON array, prefixed by the `functools` marker. Each entry in the array represents a function call, specifying the function's name and its arguments as a JSON object. By enforcing this format, you can reliably parse toolcall outputs across different models using a unified grammar.

You can then define a single, unified grammar to parse toolcall outputs from any model that follows your standardized format. This grammar expects the output to begin with the `functools` marker, followed by a JSON array of function calls that conform to your schema. For example:

```py
schema = {
  "type": "array",
  "items": {"anyOf": funcs},
  "minItems": 1,
}

grammar = f"""
start: "functools" tools
tools: %json {json.dumps(schema)}
"""
```

**Note:** This is just one possible approach. You can adjust the injected prompt to produce any output format you prefer, such as a plain JSON array of toolcalls without the `functools` prefix, depending on your integration needs.
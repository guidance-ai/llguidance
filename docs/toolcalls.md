# Tool calls

This document explains how to design and parse tool call outputs produced by different LLMs. Models often use distinct output formats, so you can either:

- build a model-specific grammar tailored to the model's expected format (recommended for best accuracy), or
- use prompt-injection to coerce self-hosted models into a consistent, universal format.

### Why tailor a grammar to each model format?

A common mistake is assuming that simply providing a grammar will make a model generate the expected output format, even without clear context in the input prompt. This often leads to out-of-distribution behavior, as illustrated below:

**Example: Out-of-distribution output**

- **Input:** "Tell me the capital and population of France"
- **Grammar:** `{"name": string, "population": int}`

If you do not provide explicit context or examples in the prompt, the model may default to its usual conversational style, such as:

> "The capital of France is Paris and its population is 2,050,000."

But if you force the model to start its answer with `{ "name": "`, a pattern it likely hasn't seen in training, it may try to continue in natural language, producing something like:

> `{ "name": "The capital of France is Paris and its population is 2,050,000`

However, without any context, the model does not know that it has to close the string with a `"`, it may continue generating text for the "name" key until it reaches the token limit. Although this output technically remains a valid JSON string value, it often leads to incomplete JSON objects or irrelevant content, such as:

> `{ "name": "The capital of France is Paris and its population is 2,050,000. Paris is famous for its iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, its world-class art. 撤撤撤撤撤撤撤撤撤撤撤撤撤撤撤撤撤撤撤撤"`

This happens because the model is operating outside the distribution of outputs it was trained on. Models are typically trained or fine-tuned on specific output styles and token sequences, so they reliably reproduce those formats. When forced to generate unfamiliar formats, they may produce invalid JSON, incomplete outputs, or unrelated text.

**Preventing Out-of-Distribution Tool Call Outputs**

To avoid out-of-distribution (OOD) outputs when using grammar for tool-calling, you can use one of two main strategies:

1. **Tailor the grammar to each model’s native output format.**  
  - This approach maximizes reliability by matching the model’s expected output structure, reducing parsing errors and unexpected results.
  - Recommended for production scenarios or when you need robust, predictable parsing.

2. **Use careful prompt injection to introduce a new, standardized tool call format.**  
  - Provide explicit, repeatable instructions and concrete examples in the prompt to guide the model to emit outputs in your desired format.
  - Use a matching grammar to parse these outputs.
  - This approach enables a unified interface across multiple models, but requires thorough validation and testing to ensure models consistently follow the new format.

The following sections provide examples for both approaches: building grammars for model-specific outputs and designing a unified interface with prompt-injection.

## Example tool definitions

Consider the following example tool definitions (Python-style):

```py
phi4_tools = [
  {
      "name": "add",
      "description": "Adds two floating-point numbers together.",
      "parameters": {
          "type": "object",
          "properties": {
              "x": {"type": "number", "description": "The first number to add."},
              "y": {"type": "number", "description": "The second number to add."},
          },
          "required": ["x", "y"],
      },
      "returns": {"type": "number", "description": "The sum of the two input numbers."},
  },
  {
      "name": "mul",
      "description": "Multiplies two floating-point numbers together.",
      "parameters": {
          "type": "object",
          "properties": {
              "x": {"type": "number", "description": "The first number to multiply."},
              "y": {"type": "number", "description": "The second number to multiply."},
          },
          "required": ["x", "y"],
      },
      "returns": {"type": "number", "description": "The product of the two input numbers."},
  }
]

# format passing to openai client
tools = [{"type": "function", "function": tool} for tool in phi4_tools]

tool_descs = []
for tool in tools:
  tool = tool["function"]
  json_desc = {
      "type": "object",
      "properties": {
          "name": {"const": tool["name"]},
          "arguments": tool["parameters"],
      },
      "required": ["name", "arguments"],
  }
  tool_descs.append(json_desc)
```

## Model-specific grammars

When targeting one model or a model family, match its native tool call format — it's the most reliable.

Simple instructions
- Read model documentation and carefully inspect a few real outputs (for single and multiple tool calls) to capture delimiters, wrappers, and spacing.
- Define grammar rules that mirror exact wrappers and the JSON shape (single vs. multi-tool call).

### Phi-4-Mini (example)

Phi-4-Mini produces tool calls as a single JSON array wrapped by special delimiters. Example output:

```
<|tool_call|>[{"name": "add", "arguments": {"x": 123345432, "y": 4563464236}}, {"name": "mul", "arguments": {"x": 874284, "y": 912429}}]<|/tool_call|>
```

A corresponding JSON schema and grammar might look like:

```py
schema = {
  "type": "array",
  "items": {"anyOf": tool_descs},
  "minItems": 1,
}

grammar = f"""
start: <|tool_call|> tools <|/tool_call|>
tools: %json {json.dumps(schema)}
"""
```

### Qwen-3 (example)

Qwen-3 may emit multiple JSON objects as separate lines, each wrapped in a tool call block. Example output:

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
start: tool*
tool: <tool_call> "\\n" (func1 | func2) "\\n" </tool_call> ("\\n")*
func1: %json {json.dumps(json_funcs[0])}
func2: %json {json.dumps(json_funcs[1])}
"""
```

Note: the exact grammar depends on the parser you use and the precise output formatting of the model.

## Unified interface (prompt-injection)

When integrating with multiple models, maintaining individual grammars for each can be cumbersome and error-prone. 
A unified interface streamlines this process by injecting clear, custom output instructions directly into the prompt. 
This approach guides models to emit tool call outputs in a consistent, predictable structure—regardless of their native tendencies. 
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

The injected prompt directs the model to produce tool calls in the following standardized format:

```
functools[{"name": [function name], "arguments": [function arguments as JSON]}, ...]
```

This structure ensures that all function calls are grouped within a single JSON array, prefixed by the `functools` marker. Each entry in the array represents a function call, specifying the function's name and its arguments as a JSON object. By enforcing this format, you can reliably parse tool call outputs across different models using a unified grammar.

You can then define a single, unified grammar to parse tool call outputs from any model that follows your standardized format. This grammar expects the output to begin with the `functools` marker, followed by a JSON array of function calls that conform to your schema. For example:

```py
schema = {
  "type": "array",
  "items": {"anyOf": tool_descs},
  "minItems": 1,
}

grammar = f"""
start: "functools" tools
tools: %json {json.dumps(schema)}
"""
```

**Note:** 
- This is just one possible approach. You can adjust the injected prompt to produce any output format you prefer, such as a plain JSON array of tool calls without the `functools` prefix, depending on your integration needs.
- While prompt-injection can unify outputs, it may not be as reliable as model-specific grammars as the model was trained intensively for its own format. However, careful prompting and, if possible, fine-tuning can help improve consistency and close the gap.
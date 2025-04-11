# JSON schema -> llguidance converter

This sub-module converts JSON schema to llguidance grammar.

It aims to either produce a grammar conformant to the JSON schema semantics (Draft 2020-12), or give an error,
but [see below](#departures-from-json-schema-semantics) for some known differences.

There are various limits on the size of the input schema and the resulting grammar.
However, we've successfully processed schemas up to 4 MB in size.

## Supported JSON schema features

Following JSON schema features are supported.

Core features:

- `anyOf`
- `oneOf` - converted to `anyOf` only when provably equivalent
- `allOf` - intersection of certain schemas is not supported right now
- `$ref` - external/remote refs unsupported
- `const`
- `enum`
- `type` - both single type and array of types
- sibling keys - when schema has keywords in addition to `anyOf`, `allOf`, `$ref`, the result is intersection

Array features:

- `items`
- `prefixItems`
- `minItems`
- `maxItems`

Object features:

- `properties` - order of properties is fixed to the order in schema
- `additionalProperties`
- `required`

String features:

- `minLength`
- `maxLength`
- `pattern`
- `format`, with the following formats: `date-time`, `time`, `date`, `duration`, `email`, `hostname`, `ipv4`, `ipv6`, `uuid`,

Number features (for both integer and number):

- `minimum`
- `maximum`
- `exclusiveMinimum`
- `exclusiveMaximum`
- `multipleOf`

## Departures from JSON schema semantics

- order of object properties is fixed, see below
- string `format` is enforced by default, with unrecognized or unimplemented formats returning errors

## Whitespace handling

By default any whitespace is allowed inside of the JSON object.
Whitespace is not allowed before the first `{` or after the last `}`.
You can modify your grammar easily to allow initial or trailing whitespace.

You can set top-level `"x-guidance"` key to control this.
Following keys are available inside of it:

- `item_separator`, defaults to `":"`
- `key_separator`, defaults to `","`
- `whitespace_flexible`, defaults to `true`; set to `false` to enforce compact JSON representation
- `whitespace_pattern`, optional string, overrides `whitespace_flexible`;
  `whitespace_flexible: true` is equivalent to `whitespace_pattern: r"[\x20\x0A\x0D\x09]+"`
- `coerce_one_of`, defaults to `false`; when set to `true`, the `"oneOf"` will be treated as `"anyOf"`
- `lenient`, defaults to `false`; when set to `true`, the unsupported keywords and formats will be ignored; implies `coerce_one_of: true`

For example:

```json
{
   "x-guidance": {
      "whitespace_flexible": false
   },
   "type": "object",
   "properties": {
      "a": {
         "type": "string"
      }
   }
}
```

The `"x-guidance"` key is only recognized at the top level of the schema.


## Property order

### TL;DR

Properties follow order in `properties` map.
When schemas are merged with `allOf` etc., the `properties` maps are merged in order.

Easiest way to override this, is to include `"my_property": true` in appropriate position in `"properties"`,
before `anyOf/allOf/oneOf/$ref`.

### Details, best ignored

While this algorithm may not be the easiest to implement, we judge it to be the least surprising to the user.
Basically, the schema is processed line-by-line, left-to-right, and property order is fixed as we go.

The enforced property order during generation is as follows:
1. Each property in the `"properties"` object, in order of appearance
2. Each property in `"required"`, in order of appearance (if not already in `"properties"` they are constrained with `"additionalPropertie"`)

When two schemas are joined (more than two is defined inductively), the resulting `"properties"` object will have order given by:
1. Each property in the left schema, in order of appearance
2. Each property in the right schema, in order of appearance (if not already in the left schema)
3. Recursive cases: 
   - If the left schema defines a property that is also found in the right schema, the schema of the resulting merged property will be the result of merging the two properties with the same left,right precedence
   - If the left schema defines a property NOT found in the right schema, it will be merged on the right by the right schema's `additionalProperties`
   - If the right schema defines a property NOT found in the left schema, it will be merged on the left by the left schema's `additionalProperties`

When two schemas are joined, the resulting `"required"` array will have order given by:
1. Each property in the left `"required"`, in order of appearance
2. Each property in the right `"required"`, in order of appearance (if not in the left array)

**Note**: precedence in `"properties"` and `"required"` are tracked *separately* as schemas are merged, with the order imposed by the final schema prioritizing `"properties"` over `"required"`.

When a schema is built from multiple [applicators](https://json-schema.org/draft/2020-12/vocab/applicator), the applicators are processed *in order of appearance*.

E.g., if a schema contains both `"properties"` and `"allOf"` (which has another `"properties"` definition nested inside), the resulting constraints will depend on which of these keys appears first. The one that appears first will have precedence.

These semantics extend even to applicators that violate the typical [keyword independence](https://json-schema.org/draft/2020-12/json-schema-core#section-10.1) semantics of JSON keywords.

E.g., even though the behavior of `additionalProperties` is defined in terms of `"properties"` and `"patternProperties"` its position in a schema determines its precedence independently of the location of `"properties"`. If it applies to a property defined in a subschema of `"allOf"` or `"anyOf"`, whether it applies before or after said definition is determined by its position relative to the `"allOf"` or `"anyOf"` keyword.

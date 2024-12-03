# JSON schema -> llguidance converter

This sub-module converts JSON schema to llguidance grammar.

It aims to either produce a grammar conformat to the JSON schema semantics, or give an error,
but [see below](#departures-from-json-schema-semantics) for some known differences.

There are various limits on the size of the input schema and the resulting grammar.
However, we've successfully processed schemas up to 4 MB in size.

## Supported JSON schema features

Following JSON schema features are supported.

Core features:

- `anyOf`
- `oneOf` - not supported right now, use `anyOf` instead, [issue](https://github.com/microsoft/llguidance/issues/77)
- `allOf` - intersection of certain schemas is not supported right now
- `$ref` - within the document only
- `const`
- `enum`
- `type` - both single type and array of types
- sibling properties - when schema has keywords in addition to `anyOf`, `allOf`, `$ref`, the result is intersection

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
- `pattern` (though we always anchor them, [issue](https://github.com/microsoft/llguidance/issues/66))
- `format`, with the following formats: `date-time`, `time`, `date`, `duration`, `email`, `hostname`, `ipv4`, `ipv6`, `uuid`,

Number features (for both integer and number):

- `minimum`
- `maximum`
- `exclusiveMinimum`
- `exclusiveMaximum`

## Departures from JSON schema semantics

- order of properties in `properties` is fixed to the order in schema (`required` can be used to skip some of them)

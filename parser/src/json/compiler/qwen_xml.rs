//! Qwen XML parameter encoding, inspired by XGrammar's JSON schema styles.
//! <https://github.com/mlc-ai/xgrammar>

use super::*;
use anyhow::ensure;

fn rx(pattern: &str) -> RegexAst {
    RegexAst::Regex(pattern.to_string())
}

fn without_closing_tag(value: RegexAst) -> RegexAst {
    RegexAst::And(vec![
        value,
        RegexAst::Not(Box::new(rx("(?s:.*)</parameter>(?s:.*)"))),
    ])
}

fn padded_string(value: RegexAst) -> RegexAst {
    // The format strips exactly one leading and one trailing LF, even when
    // either belongs to the value. Encode those cases with an extra LF.
    if let RegexAst::Literal(text) = &value {
        return RegexAst::Concat(vec![
            rx(if text.starts_with('\n') { "\n" } else { "\n?" }),
            value.clone(),
            rx(if text.ends_with('\n') { "\n" } else { "\n?" }),
        ]);
    }
    let mut alternatives = Vec::with_capacity(4);
    for leading in [false, true] {
        for trailing in [false, true] {
            let mut constraints = vec![value.clone()];
            if !leading {
                constraints.push(rx("(?s:[^\n].*)?"));
            }
            if !trailing {
                constraints.push(rx("(?s:.*[^\n])?"));
            }
            alternatives.push(RegexAst::Concat(vec![
                rx(if leading { "\n" } else { "" }),
                RegexAst::And(constraints),
                rx(if trailing { "\n" } else { "" }),
            ]));
        }
    }
    RegexAst::Or(alternatives)
}

fn check_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && !name
                .chars()
                .any(|ch| ch.is_whitespace() || ch == '<' || ch == '>'),
        "Invalid Qwen XML parameter name: {name:?}"
    );
    Ok(())
}

fn root_object<'a>(
    schema: &'a Schema,
    definitions: &'a HashMap<String, Schema>,
) -> Result<&'a ObjectSchema> {
    let mut current = schema;
    for _ in 0..128 {
        match current {
            Schema::Object(obj) => return Ok(obj),
            Schema::Ref(path) => {
                current = definitions
                    .get(path)
                    .ok_or_else(|| anyhow!("Missing reference: {path}"))?
            }
            _ => bail!("Qwen XML parameter encoding requires an object schema"),
        }
    }
    bail!("Recursive root reference in Qwen XML object schema")
}

fn validate_object(obj: &ObjectSchema) -> Result<bool> {
    ensure!(
        obj.pattern_properties.is_empty(),
        "Top-level patternProperties is not supported by Qwen XML"
    );
    let extras = match obj.additional_properties.schema_ref() {
        Schema::Any => true,
        Schema::Unsatisfiable(_) => false,
        _ => bail!("Schema-valued top-level additionalProperties is not supported by Qwen XML"),
    };
    let optional = obj
        .properties
        .keys()
        .any(|name| !obj.required.contains(name));
    ensure!(
        !optional || (obj.min_properties == 0 && obj.max_properties.is_none()),
        "min/maxProperties with optional declared properties is not supported"
    );
    Ok(extras)
}

pub(crate) fn prepare_tool_schema(mut schema: Value) -> Result<Value> {
    let built = build_schema(schema.clone(), &JsonCompileOptions::default())?;
    let obj = root_object(&built.schema, &built.definitions)?;
    validate_object(obj)?;
    if obj.additional_properties.is_none() {
        // Apply the tool policy only at the root. In particular, a shared $ref
        // must retain its original semantics when used inside a nested value.
        let root = schema
            .as_object_mut()
            .ok_or_else(|| anyhow!("Expected object schema"))?;
        let properties = root
            .entry("properties")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| anyhow!("properties must be an object"))?;
        for name in obj.properties.keys() {
            properties.entry(name).or_insert(Value::Bool(true));
        }
        root.insert("additionalProperties".to_string(), Value::Bool(false));
    }
    schema["x-guidance"] = json!({"style": "qwen_xml"});
    Ok(schema)
}

impl Compiler {
    pub(super) fn gen_xml_object(
        &mut self,
        schema: &Schema,
        definitions: &HashMap<String, Schema>,
    ) -> Result<NodeRef> {
        self.builder.check_limits()?;
        let obj = root_object(schema, definitions)?;
        let extras = validate_object(obj)?;

        let mut items = Vec::with_capacity(obj.properties.len() + 1);
        let mut names = Vec::with_capacity(obj.properties.len());
        for name in obj.properties.keys().chain(
            obj.required
                .iter()
                .filter(|name| !obj.properties.contains_key(*name)),
        ) {
            check_name(name)?;
            names.push(RegexAst::Literal(name.to_string()));
            let required = obj.required.contains(name);
            let value_schema = self.pattern_cache.property_schema(obj, name)?;
            let item = match self.gen_xml_parameter(name, value_schema, definitions, 0) {
                Ok(node) => node,
                Err(error)
                    if !required && error.downcast_ref::<UnsatisfiableSchemaError>().is_some() =>
                {
                    continue
                }
                Err(error) => return Err(error.context(format!("parameter {name:?}"))),
            };
            items.push(if required {
                item
            } else {
                self.builder.optional(item)
            });
        }

        let minimum = obj.min_properties.saturating_sub(obj.required.len());
        let maximum = obj
            .max_properties
            .map(|n| n.saturating_sub(obj.required.len()));
        if extras && maximum != Some(0) {
            let name = RegexAst::And(vec![
                rx("[^\\s<>]+"),
                RegexAst::Not(Box::new(RegexAst::Or(names))),
            ]);
            let item = self.gen_xml_text(RegexAst::Concat(vec![
                RegexAst::Literal("<parameter=".to_string()),
                name,
                RegexAst::Literal(">".to_string()),
            ]))?;
            items.push(self.builder.repeat(item, minimum, maximum));
        } else {
            ensure!(
                minimum == 0,
                "minProperties exceeds the available parameters"
            );
        }
        Ok(self.builder.join(&items))
    }

    fn gen_xml_parameter(
        &mut self,
        name: &str,
        schema: &Schema,
        definitions: &HashMap<String, Schema>,
        depth: usize,
    ) -> Result<NodeRef> {
        ensure!(
            depth < 128,
            "Recursive parameter reference without a JSON container"
        );
        self.builder.check_limits()?;
        let opener = format!("<parameter={name}>");
        let value = match schema {
            Schema::Ref(path) => {
                let target = definitions
                    .get(path)
                    .ok_or_else(|| anyhow!("Missing reference: {path}"))?;
                return self.gen_xml_parameter(name, target, definitions, depth + 1);
            }
            Schema::AnyOf(options) => {
                let mut nodes = Vec::with_capacity(options.len());
                for option in options {
                    match self.gen_xml_parameter(name, option, definitions, depth + 1) {
                        Ok(node) => nodes.push(node),
                        Err(error)
                            if error.downcast_ref::<UnsatisfiableSchemaError>().is_some() => {}
                        Err(error) => return Err(error),
                    }
                }
                ensure!(!nodes.is_empty(), "No satisfiable parameter alternatives");
                return Ok(self.builder.select(&nodes));
            }
            Schema::Any => return self.gen_xml_text(RegexAst::Literal(opener)),
            Schema::String(options) => {
                if options.min_length == 0
                    && options.max_length.is_none()
                    && options.regex.is_none()
                {
                    return self.gen_xml_text(RegexAst::Literal(opener));
                }
                let raw = self.gen_string(options.clone())?;
                without_closing_tag(padded_string(raw))
            }
            Schema::Null | Schema::Boolean(_) => RegexAst::Concat(vec![
                rx("\n?"),
                self.regex_compile(schema)?.unwrap(),
                rx("\n?"),
            ]),
            Schema::Number(_) => RegexAst::Concat(vec![
                rx("[ \\t\\r\\n]*"),
                self.regex_compile(schema)?.unwrap(),
                rx("[ \\t\\r\\n]*"),
            ]),
            _ => {
                let open = self.builder.string(&opener);
                let value = self.gen_json(schema)?;
                let close = self.builder.string("</parameter>");
                return Ok(self.builder.join(&[open, value, close]));
            }
        };
        self.ast_lexeme(RegexAst::Concat(vec![
            RegexAst::Literal(opener),
            value,
            RegexAst::Literal("</parameter>".to_string()),
        ]))
    }

    fn gen_xml_text(&mut self, opener: RegexAst) -> Result<NodeRef> {
        // The safe subset exposes a repeat that the slicer can recognize; the
        // full lexeme also accepts '<' without allowing the closing delimiter.
        let mut nodes = Vec::with_capacity(2);
        for value in [rx("[^<]*"), without_closing_tag(rx("(?s:.*)"))] {
            nodes.push(self.ast_lexeme(RegexAst::Concat(vec![
                opener.clone(),
                value,
                RegexAst::Literal("</parameter>".to_string()),
            ]))?);
        }
        Ok(self.builder.select(&nodes))
    }
}

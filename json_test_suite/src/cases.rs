use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize, Debug)]
pub struct Case {
    pub schema: Value,
    pub tests: Vec<Test>,
}

#[derive(Deserialize, Debug)]
pub struct Test {
    pub data: Value,
    pub valid: bool,
}

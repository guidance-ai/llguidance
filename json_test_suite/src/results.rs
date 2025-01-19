use serde::Serialize;

#[derive(Serialize, Debug)]
pub struct TestResult {
    pub valid: bool,
    pub success: bool,
    pub error: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct CaseResult {
    pub tests: Vec<TestResult>,
}

#[derive(Serialize, Debug)]
pub struct CategoryResult {
    pub category: String,
    pub cases: Vec<CaseResult>,
}

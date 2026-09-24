//! Tests for the `_immut` (non-advancing) mask / ff reads and the safe
//! snapshot mechanism.
//!
//! The postulates under test (each a by a single-variable observation):
//!  1. When the state is settled (`settle`), `compute_mask_immut` equals the
//!     existing `compute_mask` (the immut read is correct).
//!  2. When the, `compute_ff_tokens_immut` equals the existing
//!     `compute_ff_tokens` (the immut ff-read is correct).
//!  3. The `_immut` reads are non-perturbing (repeated calls leave the mask
//!     unchanged, no `ff_tokens_cache` observer effect).
//!  4. `snapshot` does not perturb the original, and `restore` is exact
//!     (the state after restore equals the pre-op state).

use lazy_static::lazy_static;
use llg_test_utils::get_tok_env;
use llguidance::{
    api::TopLevelGrammar,
    earley::SlicedBiasComputer,
    toktrie::{InferenceCapabilities, SimpleVob},
    Matcher, ParserFactory,
};

lazy_static! {
    static ref PARSER_FACTORY: ParserFactory = {
        let tok_env = get_tok_env().clone();
        let mut fact = ParserFactory::new(
            &tok_env,
            InferenceCapabilities {
                ff_tokens: true,
                backtrack: false,
                conditional_ff_tokens: false,
                fork: false,
            },
            &SlicedBiasComputer::general_slices(),
        )
        .unwrap();
        fact.quiet();
        fact
    };
}

fn create_matcher(grammar: &str, max_tokens: Option<usize>) -> Matcher {
    let mut grm = TopLevelGrammar::from_lark(grammar.to_string());
    grm.max_tokens = max_tokens;
    let parser = PARSER_FACTORY.create_parser(grm);
    Matcher::new(parser)
}

fn masks_equal(a: &SimpleVob, b: &SimpleVob) -> bool {
    a.as_slice() == b.as_slice()
}

/// A grammar with a forced literal run (the ff tokens are non-empty at the start).
const GRM: &str = r#"start: "a" "b" "c""#;

#[test]
fn test_immut_mask_equals_settled_mask() {
    let mut m = create_matcher(GRM, None);
    m.settle();
    let mask_immut = m.compute_mask_immut().unwrap();
    let mask_existing = m.compute_mask().unwrap();
    assert!(
        masks_equal(&mask_immut, &mask_existing),
        "settled immut mask must equal the existing mask"
    );
}

#[test]
fn test_immut_ff_equals_existing_ff() {
    let mut m = create_matcher(GRM, None);
    m.settle();
    let ff_immut = m.compute_ff_tokens_immut();
    let ff_existing = m.compute_ff_tokens();
    assert_eq!(ff_immut, ff_existing, "settled immut ff must equal the existing ff");
    assert!(!ff_immut.is_empty(), "the forced literal run is non-empty at the start");
}

#[test]
fn test_immut_reads_are_non_perturbing() {
    let mut m = create_matcher(GRM, None);
    m.settle();
    let mask_before = m.compute_mask_immut().unwrap();
    // repeated immut reads (the no observer effect)
    let _ = m.compute_mask_immut().unwrap();
    let _ = m.compute_ff_tokens_immut();
    let _ = m.compute_mask_immut().unwrap();
    let mask_after = m.compute_mask_immut().unwrap();
    assert!(
        masks_equal(&mask_before, &mask_after),
        "immut reads must not perturb the mask"
    );
}

#[test]
fn test_snapshot_non_perturbing_and_restore_exact() {
    let mut m = create_matcher(GRM, None);
    m.settle();
    let mask_pre = m.compute_mask_immut().unwrap();

    let snap = m.snapshot();
    // the snapshot must not perturb the original
    let mask_post_snap = m.compute_mask_immut().unwrap();
    assert!(masks_equal(&mask_pre, &mask_post_snap), "snapshot must not perturb");

    // a mutating op (the existing ff-read settles + primes the cache)
    let _ = m.compute_ff_tokens();

    // the exact restore
    m.restore(snap);
    let mask_restored = m.compute_mask_immut().unwrap();
    assert!(
        masks_equal(&mask_pre, &mask_restored),
        "restore must be exact (the pre-op state)"
    );
}
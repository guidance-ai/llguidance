use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use llguidance::{api::ParserLimits, derivre::RegexAst, earley::lexerspec::LexerSpec};

/// Counts requested allocation sizes on the measured thread, including reallocations.
/// Large buffers are tracked separately to catch storage proportional to lexeme count.
#[derive(Clone, Copy, Debug, Default)]
struct Allocations {
    calls: usize,
    bytes: usize,
    large_calls: usize,
    large_bytes: usize,
}

thread_local! {
    static ALLOCATIONS: Cell<Option<Allocations>> = const { Cell::new(None) };
}

/// Delegates to the system allocator while measuring only explicitly scoped work.
struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Records without allocating, so instrumentation cannot recursively invoke itself.
fn record(size: usize) {
    let _ = ALLOCATIONS.try_with(|counter| {
        if let Some(mut allocations) = counter.get() {
            allocations.calls += 1;
            allocations.bytes += size;
            if size >= 64 * 1024 {
                allocations.large_calls += 1;
                allocations.large_bytes += size;
            }
            counter.set(Some(allocations));
        }
    });
}

// SAFETY: All allocation and deallocation operations use the same system allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    /// Records the request and preserves the system allocator's layout contract.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }

    /// Counts zeroed allocations as well as ordinary allocations.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    /// Releases memory with the original allocator and layout.
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    /// Counts the full requested size when a buffer grows or shrinks.
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
}

/// Measures one operation without including its caller's setup or result destruction.
fn measure<T>(f: impl FnOnce() -> T) -> (T, Allocations) {
    ALLOCATIONS.with(|counter| assert!(counter.replace(Some(Allocations::default())).is_none()));
    let result = f();
    let allocations = ALLOCATIONS.with(|counter| counter.take().unwrap());
    (result, allocations)
}

/// Ordinary grammars must not pay for interval configurations or matcher tables.
/// Thousands of distinct lexemes expose even one additional slot per lexeme.
#[test]
fn ordinary_lexer_allocations() {
    let (spec, build) = measure(|| {
        let mut spec = LexerSpec::new().unwrap();
        spec.setup_lexeme_class(RegexAst::NoMatch).unwrap();
        for idx in 0..4096 {
            spec.add_simple_literal(String::new(), &format!("keyword_{idx:04}"), false)
                .unwrap();
        }
        spec
    });
    let (spec_copy, spec_clone) = measure(|| spec.clone());
    let (mut lexer, compile) = measure(|| spec.to_regex_vec(&mut ParserLimits::default()).unwrap());
    let initial = lexer.initial_state(&spec.all_lexemes());
    let mut state = initial;
    for &byte in b"keyword_0042" {
        state = lexer.transition(state, byte);
        assert!(!state.is_dead());
    }
    assert!(lexer.state_desc(state).greedy_accepting.is_some());
    let (lexer_copy, lexer_clone) = measure(|| lexer.clone());
    println!("build: {build:?}\nspec clone: {spec_clone:?}\ncompile: {compile:?}\nlexer clone: {lexer_clone:?}");
    std::hint::black_box((spec_copy, lexer_copy));

    // Measured before adding %int_ranges on 64-bit targets with the default
    // hasher. Allow small fixed changes, but less than a pointer per lexeme
    // (32 KiB), so an optional feature cannot silently add a dense table.
    #[cfg(all(target_pointer_width = "64", feature = "ahash"))]
    for (name, actual, bytes, large_bytes) in [
        ("build", build, 3_563_172, 2_935_552),
        ("spec clone", spec_clone, 933_766, 787_036),
        ("compile", compile, 1_467_930, 98_740),
        ("lexer clone", lexer_clone, 2_600_376, 2_489_404),
    ] {
        assert!(actual.bytes <= bytes + 16 * 1024, "{name}: {actual:?}");
        assert!(
            actual.large_bytes <= large_bytes + 16 * 1024,
            "{name}: {actual:?}"
        );
    }
}

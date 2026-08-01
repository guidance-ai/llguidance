//! Lexer backtracking when a longer greedy lexeme match fails.
//!
//! While a greedy lexeme remains live, an earlier accepting position may still
//! be needed if the longer match later dies. This module retains that earlier
//! parser state and replays subsequent bytes from it when necessary.

use super::{
    BiasComputer, LexerResult, Parser, ParserRecognizer, ParserState, ParserStats, PreLexeme,
    Recognizer, SharedState, SimpleVob, TokTrie, TokenId,
};

#[derive(Clone)]
struct Snapshot {
    trigger: usize,
    state: ParserState,
    fallback_state: Option<Box<LexerBacktracking>>,
}

#[derive(Clone, Copy)]
enum Commit<'a> {
    Token(&'a [u8], TokenId),
    Byte(u8),
}

impl Commit<'_> {
    fn apply(self, state: &mut ParserState) -> bool {
        match self {
            Self::Token(bytes, token) => {
                let ok = matches!(state.apply_token(bytes, token), Ok(0));
                state.token_idx += usize::from(ok);
                ok
            }
            Self::Byte(byte) => state.try_push_byte_definitive(Some(byte)) == (true, 0),
        }
    }
}

#[derive(Clone, Default)]
pub(super) struct LexerBacktracking {
    // State before a fallback became definitive, used to undo it on rollback.
    rollback_snapshot: Option<Box<Snapshot>>,
    // Latest accepting lexer-stack position in the current parser row.
    accepting_idx: Option<usize>,
    // Number of lexer-stack entries already checked for an accepting position.
    checked_stack_len: usize,
    // Whether we already tried constructing the fallback at `accepting_idx`.
    fallback_tried: bool,
    // Parser state obtained by ending the lexeme at `accepting_idx`.
    fallback: Option<Box<Snapshot>>,
}

struct FallbackContext<'a> {
    state: &'a mut ParserState,
    fallback: &'a mut LexerBacktracking,
}

fn with_fallback<T>(state: &mut ParserState, f: impl FnOnce(&mut FallbackContext<'_>) -> T) -> T {
    let mut fallback = state.shared_box.lexer_backtracking.take().unwrap();
    let result = f(&mut FallbackContext {
        state,
        fallback: &mut fallback,
    });
    state.shared_box.lexer_backtracking = Some(fallback);
    result
}

fn remaining_items(state: &ParserState) -> Option<usize> {
    (state.max_all_items != usize::MAX)
        .then(|| state.max_all_items.saturating_sub(state.stats.all_items))
}

fn set_remaining_items(state: &mut ParserState, remaining: Option<usize>) {
    state.max_all_items = remaining.map_or(usize::MAX, |remaining| {
        state.stats.all_items.saturating_add(remaining)
    });
}

// Merge actual parser work. Wall-clock time and DFA fuel are recorded by the
// outer operation against the single shared lexer.
fn merge_work_stats(owner: &mut ParserState, before: &ParserStats, branch: &ParserState) {
    let delta = branch.stats.delta(before);
    owner.stats.rows += delta.rows;
    owner.stats.cached_rows += delta.cached_rows;
    owner.stats.all_items += delta.all_items;
    owner.stats.slices_applied += delta.slices_applied;
    owner.stats.trie_nodes_walked += delta.trie_nodes_walked;
    owner.stats.definitive_bytes += delta.definitive_bytes;
    owner.stats.lexer_ops += delta.lexer_ops;
    owner.stats.num_lex_errors += delta.num_lex_errors;
    owner.stats.num_lexemes += delta.num_lexemes;
}

fn clone_without_shared(state: &mut ParserState) -> ParserState {
    let shared = std::mem::take(&mut state.shared_box);
    let clone = state.clone();
    state.shared_box = shared;
    clone
}

fn run_branch<T>(
    owner: &mut ParserState,
    branch: &mut ParserState,
    shared: &mut Box<SharedState>,
    f: impl FnOnce(&mut ParserState) -> T,
) -> T {
    set_remaining_items(branch, remaining_items(owner));
    let before = branch.stats.clone();
    std::mem::swap(&mut branch.shared_box, shared);
    let result = f(branch);
    std::mem::swap(&mut branch.shared_box, shared);
    merge_work_stats(owner, &before, branch);
    result
}

// Snapshots retain parser and fallback state, but borrow the live lexer
// only while they are being advanced.
fn with_snapshot<T>(
    owner: &mut ParserState,
    snapshot: &mut Snapshot,
    f: impl FnOnce(&mut ParserState) -> T,
) -> T {
    let mut shared = std::mem::take(&mut owner.shared_box);
    shared.lexer_backtracking = snapshot.fallback_state.take();
    let result = run_branch(owner, &mut snapshot.state, &mut shared, f);
    snapshot.fallback_state = shared.lexer_backtracking.take();
    owner.shared_box = shared;
    result
}

fn commit_lexeme_at(state: &mut ParserState, checkpoint: usize, pre: PreLexeme) -> bool {
    if !state.has_pending_lexeme_bytes() || !state.advance_parser(pre) {
        return false;
    }
    let next = state.lexer_stack.pop().unwrap();
    state.lexer_stack[checkpoint] = next;
    true
}

fn hidden_boundary(
    state: &ParserState,
    shared: &mut SharedState,
    step: &LexerResult,
) -> Option<(usize, PreLexeme)> {
    let LexerResult::State(next, _) = step else {
        return None;
    };
    if !shared.lexer_mut().try_lexeme_end(*next).is_error() || !state.has_pending_lexeme_bytes() {
        return None;
    }
    let checkpoint = state.lexer_stack.len() - 1;
    let item = state.lexer_stack[checkpoint];
    let LexerResult::Lexeme(pre) = shared.lexer_mut().try_lexeme_end(item.lexer_state) else {
        return None;
    };
    Some((checkpoint, pre))
}

fn branch_is_accepting(state: &ParserState, shared: &mut SharedState) -> bool {
    if !state.has_pending_lexeme_bytes() {
        return false;
    }
    let item = state.lexer_state();
    matches!(
        shared.lexer_mut().try_lexeme_end(item.lexer_state),
        LexerResult::Lexeme(_)
    )
}

impl FallbackContext<'_> {
    fn snapshot_with(&mut self, fallback: LexerBacktracking) -> Snapshot {
        Snapshot {
            trigger: 0,
            state: clone_without_shared(self.state),
            fallback_state: Some(Box::new(fallback)),
        }
    }

    fn snapshot(&mut self) -> Box<Snapshot> {
        let fallback = std::mem::take(self.fallback);
        Box::new(self.snapshot_with(fallback))
    }

    fn restore(&mut self, mut saved: Box<Snapshot>) {
        let stats = std::mem::take(&mut self.state.stats);
        let metrics = std::mem::take(&mut self.state.metrics);
        let parser_error = self.state.parser_error.take();
        let max_all_items = self.state.max_all_items;

        saved.state.shared_box = std::mem::take(&mut self.state.shared_box);
        *self.state = saved.state;
        self.state.stats = stats;
        self.state.metrics = metrics;
        self.state.parser_error = parser_error;
        self.state.max_all_items = max_all_items;
        *self.fallback = *saved.fallback_state.take().unwrap();
    }

    fn latest_accepting(&mut self) -> Option<(usize, PreLexeme)> {
        self.refresh_accepting();
        let idx = self.fallback.accepting_idx?;
        let item = self.state.lexer_stack[idx];
        let LexerResult::Lexeme(pre) = self.state.lexer_mut().try_lexeme_end(item.lexer_state)
        else {
            return None;
        };
        Some((idx, pre))
    }

    fn try_recover(&mut self, byte: Option<u8>, flush_end: bool) -> Option<(bool, usize)> {
        let (checkpoint, mut pre) = self.latest_accepting()?;
        let mut bytes = self.state.lexer_stack[checkpoint + 1..]
            .iter()
            .map(|state| state.byte)
            .collect::<Option<Vec<_>>>()?;
        let existing = bytes.len();
        bytes.extend(byte);
        let (&first, rest) = bytes.split_first()?;
        let previous = self.snapshot();
        let prefix = self.state.bytes.len().checked_sub(existing)?;
        self.state.bytes.truncate(prefix);
        self.state
            .byte_to_token_idx
            .truncate(prefix.min(self.state.byte_to_token_idx.len()));
        self.state.row_infos.truncate(self.state.num_rows());
        self.state.last_force_bytes_len = usize::MAX;
        self.state.rows_valid_end = self.state.num_rows();
        self.state.lexer_stack.truncate(checkpoint + 1);
        self.state.lexer_stack_top_eos = false;
        self.state.lexer_stack_flush_position = 0;
        self.discard_fallback();
        pre.byte = Some(first);
        pre.byte_next_row = true;
        let mut ok = self.state.advance_parser(pre);
        let mut backtrack = 0;
        if ok {
            self.state.bytes.push(first);
        }
        for &byte in rest {
            if !ok {
                break;
            }
            let result = self.state.try_push_byte_definitive(Some(byte));
            (ok, backtrack) = if result.0 {
                result
            } else {
                self.recover(Some(byte), false)
            };
        }
        if ok && backtrack == 0 {
            let end = (prefix + existing).min(previous.state.byte_to_token_idx.len());
            if prefix < end {
                self.state
                    .byte_to_token_idx
                    .extend_from_slice(&previous.state.byte_to_token_idx[prefix..end]);
            }
        }
        if ok && backtrack == 0 && flush_end {
            ok = self.state.flush_lexer() || self.recover(None, true).0;
        }
        if !ok || backtrack > 0 {
            self.restore(previous);
            return Some((false, backtrack));
        }
        let mut previous = previous;
        previous.trigger = previous.state.bytes.len() + usize::from(byte.is_some());
        self.fallback.rollback_snapshot = Some(previous);
        Some((true, backtrack))
    }

    fn recover(&mut self, byte: Option<u8>, flush_end: bool) -> (bool, usize) {
        self.try_recover(byte, flush_end).unwrap_or((false, 0))
    }

    fn commit_fallback_prefix(&mut self) -> bool {
        let Some((checkpoint, pre)) = self.latest_accepting() else {
            return false;
        };
        if checkpoint + 1 == self.state.lexer_stack.len() {
            commit_lexeme_at(self.state, checkpoint, pre)
        } else {
            self.recover(None, false).0
        }
    }

    fn make_fallback_snapshot(&mut self) -> Option<Box<Snapshot>> {
        let mut snapshot = Box::new(self.snapshot_with(LexerBacktracking::default()));
        let ok = with_snapshot(self.state, &mut snapshot, |state| {
            with_fallback(state, |fallback| fallback.commit_fallback_prefix())
        });
        ok.then(|| {
            snapshot.fallback_state.as_mut().unwrap().rollback_snapshot = None;
            snapshot
        })
    }

    fn refresh_accepting(&mut self) {
        if self.fallback.checked_stack_len > self.state.lexer_stack.len() {
            self.discard_fallback();
        }
        let previous = self.fallback.accepting_idx;
        let mut accepting_idx = self.fallback.accepting_idx;
        for idx in self.fallback.checked_stack_len..self.state.lexer_stack.len() {
            let item = self.state.lexer_stack[idx];
            if accepting_idx.is_some_and(|idx| self.state.lexer_stack[idx].row_idx != item.row_idx)
            {
                accepting_idx = None;
            }
            if matches!(
                self.state.lexer_mut().try_lexeme_end(item.lexer_state),
                LexerResult::Lexeme(_)
            ) {
                accepting_idx = Some(idx);
            }
        }
        self.fallback.checked_stack_len = self.state.lexer_stack.len();
        if accepting_idx.is_some_and(|idx| {
            self.state.lexer_stack[idx].row_idx != self.state.lexer_state().row_idx
        }) {
            accepting_idx = None;
        }
        if accepting_idx != previous {
            self.fallback.accepting_idx = accepting_idx;
            self.fallback.fallback_tried = false;
            self.fallback.fallback = None;
        }
    }

    fn refresh_fallback(&mut self) {
        self.refresh_accepting();
        if self.fallback.accepting_idx.is_some() && !self.fallback.fallback_tried {
            self.fallback.fallback_tried = true;
            self.fallback.fallback = self.make_fallback_snapshot();
        }
    }

    fn committed(&mut self, commit: Commit<'_>) {
        if let Some(mut fallback) = self.fallback.fallback.take() {
            let ok = with_snapshot(self.state, &mut fallback, |state| {
                let ok = commit.apply(state);
                if ok && state.shared_box.lexer_backtracking.is_some() {
                    with_fallback(state, |fallback| fallback.committed(commit));
                }
                ok
            });
            if ok {
                self.fallback.fallback = Some(fallback);
            }
        }
        self.refresh_fallback();
    }

    fn token_committed(&mut self, bytes: &[u8], token: TokenId) {
        self.committed(Commit::Token(bytes, token));
    }

    fn forced_byte_committed(&mut self, byte: u8) {
        self.committed(Commit::Byte(byte));
    }

    fn discard_fallback(&mut self) {
        self.fallback.accepting_idx = None;
        self.fallback.checked_stack_len = 0;
        self.fallback.fallback_tried = false;
        self.fallback.fallback = None;
    }

    fn visit_fallbacks(&mut self, f: &mut impl FnMut(&mut ParserState)) {
        self.refresh_fallback();
        if let Some(mut fallback) = self.fallback.fallback.take() {
            with_snapshot(self.state, &mut fallback, |state| {
                f(state);
                if state.shared_box.lexer_backtracking.is_some() {
                    with_fallback(state, |fallback| fallback.visit_fallbacks(f));
                }
            });
            self.fallback.fallback = Some(fallback);
        }
    }

    fn visit_hidden_fallbacks(&mut self, f: &mut impl FnMut(&mut ParserState)) {
        self.refresh_fallback();
        let hidden = self
            .fallback
            .accepting_idx
            .is_some_and(|idx| idx + 1 < self.state.lexer_stack.len());
        if !hidden {
            return;
        }
        if let Some(mut fallback) = self.fallback.fallback.take() {
            with_snapshot(self.state, &mut fallback, |state| {
                f(state);
                if state.shared_box.lexer_backtracking.is_some() {
                    with_fallback(state, |fallback| fallback.visit_hidden_fallbacks(f));
                }
            });
            self.fallback.fallback = Some(fallback);
        }
    }

    fn collect_hidden_fallbacks(&mut self, fallbacks: &mut Vec<ParserState>) {
        self.visit_hidden_fallbacks(&mut |state| {
            fallbacks.push(clone_without_shared(state));
        });
    }

    fn accepting_allows_eos(&mut self) -> bool {
        self.refresh_accepting();
        self.fallback.accepting_idx.is_some_and(|idx| {
            let item = self.state.lexer_stack[idx];
            self.state.lexer_mut().allows_eos(item.lexer_state)
        })
    }

    fn prepare_rollback(&mut self, target: usize) {
        while let Some(undo) = self
            .fallback
            .rollback_snapshot
            .take_if(|undo| target < undo.trigger)
        {
            self.restore(undo);
        }
        self.discard_fallback();
    }
}

fn visit_persisted_fallbacks(state: &ParserState, f: &mut impl FnMut(&ParserState)) {
    let Some(fallback) = state.shared_box.lexer_backtracking.as_deref() else {
        return;
    };
    visit_persisted_fallback(fallback, f);
}

fn visit_persisted_fallback(fallback: &LexerBacktracking, f: &mut impl FnMut(&ParserState)) {
    let Some(snapshot) = fallback.fallback.as_deref() else {
        return;
    };
    f(&snapshot.state);
    if let Some(nested) = snapshot.fallback_state.as_deref() {
        visit_persisted_fallback(nested, f);
    }
}

pub(super) fn temperature(state: &ParserState) -> Option<f32> {
    let mut temperature = state.temperature();
    visit_persisted_fallbacks(state, &mut |branch| {
        if let Some(branch_temperature) = branch.temperature() {
            temperature = Some(temperature.map_or(branch_temperature, |current| {
                current.max(branch_temperature)
            }));
        }
    });
    temperature
}

fn hidden_fallbacks(state: &mut ParserState) -> Vec<ParserState> {
    let mut fallbacks = Vec::new();
    if state.shared_box.lexer_backtracking.is_some() {
        with_fallback(state, |fallback| {
            fallback.collect_hidden_fallbacks(&mut fallbacks)
        });
    }
    fallbacks
}

// Keep fallback dispatch out of line so feature-off parser wrappers stay small.
#[inline(never)]
pub(super) fn visit_fallbacks(state: &mut ParserState, mut f: impl FnMut(&mut ParserState)) {
    if state.shared_box.lexer_backtracking.is_some() {
        with_fallback(state, |fallback| fallback.visit_fallbacks(&mut f));
    }
}

// The primary parse plus the fallback states which are already hidden behind
// a longer greedy attempt.
struct FallbackBranch {
    state: ParserState,
    active: bool,
    active_history: Vec<bool>,
    pending_step: Option<LexerResult>,
}

// A trie-local fallback can replace older descendants. Keep the old suffix so
// pop_bytes() can restore exactly the state at the parent trie prefix.
struct FallbackChange {
    depth: usize,
    parent: usize,
    replaced: Vec<FallbackBranch>,
}

struct FallbackRecognizer<'a> {
    owner: &'a mut ParserState,
    shared: Box<SharedState>,
    branches: Vec<FallbackBranch>,
    changes: Vec<FallbackChange>,
    depth: usize,
}

impl<'a> FallbackRecognizer<'a> {
    fn new(owner: &'a mut ParserState) -> Self {
        let mut states = vec![clone_without_shared(owner)];
        states.extend(hidden_fallbacks(owner));

        let fallback = owner.shared_box.lexer_backtracking.take();
        let shared = std::mem::take(&mut owner.shared_box);
        owner.shared_box.lexer_backtracking = fallback;

        Self {
            owner,
            shared,
            branches: states
                .into_iter()
                .map(|state| FallbackBranch {
                    state,
                    active: true,
                    active_history: Vec::new(),
                    pending_step: None,
                })
                .collect(),
            changes: Vec::new(),
            depth: 0,
        }
    }

    // Compute each lexer transition once. Most trie edges remain inside the
    // current lexeme, so reusing this result avoids a second DFA transition.
    fn prepare_steps(&mut self, byte: u8) {
        let shared = &mut self.shared;
        for branch in &mut self.branches {
            branch.pending_step = branch.active.then(|| {
                let current = branch.state.lexer_state().lexer_state;
                shared.lexer_mut().advance(current, byte, false)
            });
        }
    }

    // Save a boundary only when this byte hides an accepting state behind a
    // live, non-accepting continuation. Dead transitions are already handled by
    // the lexer's normal one-byte lookahead; another accepting state supersedes
    // the old boundary.
    fn add_hidden_fallback(&mut self, byte: u8) {
        for idx in 0..self.branches.len() {
            let Some(step) = self.branches[idx].pending_step.as_ref() else {
                continue;
            };
            let Some((checkpoint, pre)) =
                hidden_boundary(&self.branches[idx].state, &mut self.shared, step)
            else {
                continue;
            };

            let mut state = clone_without_shared(&mut self.branches[idx].state);
            let ok = run_branch(self.owner, &mut state, &mut self.shared, |state| {
                commit_lexeme_at(state, checkpoint, pre)
            });
            if !ok {
                continue;
            }

            let replaced = self.branches.split_off(idx + 1);
            let current = state.lexer_state().lexer_state;
            let step = self.shared.lexer_mut().advance(current, byte, false);
            self.branches.push(FallbackBranch {
                state,
                active: true,
                active_history: Vec::new(),
                pending_step: Some(step),
            });
            self.changes.push(FallbackChange {
                depth: self.depth + 1,
                parent: idx,
                replaced,
            });
            break;
        }
    }

    fn apply_step(&mut self, idx: usize, byte: u8, step: LexerResult) -> bool {
        if super::ITEM_TRACE {
            self.branches[idx].state.trace_byte_stack.push(byte);
        }
        let current = self.branches[idx].state.lexer_state();
        let ok = match step {
            LexerResult::State(next_state, step_byte) => {
                debug_assert_eq!(step_byte, byte);
                self.branches[idx]
                    .state
                    .lexer_stack
                    .push(super::LexerState {
                        row_idx: current.row_idx,
                        lexer_state: next_state,
                        byte: Some(step_byte),
                    });
                true
            }
            LexerResult::Error => false,
            step => run_branch(
                self.owner,
                &mut self.branches[idx].state,
                &mut self.shared,
                |state| state.advance_lexer_or_parser(step, current),
            ),
        };
        if super::ITEM_TRACE && !ok {
            self.branches[idx].state.trace_byte_stack.pop();
        }
        ok
    }

    // Once a higher-priority branch is accepting again, older fallback
    // descendants cannot win maximal munch and can be dropped for this prefix.
    fn prune_superseded(&mut self) {
        if self.branches.len() < 2 {
            return;
        }
        let mut keep = self.branches.len();
        for idx in 0..self.branches.len() - 1 {
            if !self.branches[idx].active {
                continue;
            }
            if branch_is_accepting(&self.branches[idx].state, &mut self.shared) {
                keep = idx + 1;
                break;
            }
        }
        if keep < self.branches.len() {
            let replaced = self.branches.split_off(keep);
            self.changes.push(FallbackChange {
                depth: self.depth,
                parent: keep - 1,
                replaced,
            });
        }
    }

    fn revert_changes_after(&mut self, depth: usize) {
        while self
            .changes
            .last()
            .is_some_and(|change| change.depth > depth)
        {
            let change = self.changes.pop().unwrap();
            self.branches.truncate(change.parent + 1);
            self.branches.extend(change.replaced);
        }
    }

    fn revert_all_changes(&mut self) {
        while let Some(change) = self.changes.pop() {
            self.branches.truncate(change.parent + 1);
            self.branches.extend(change.replaced);
        }
    }

    fn for_each_branch(&mut self, mut f: impl FnMut(&mut ParserState)) {
        let owner = &mut self.owner;
        let shared = &mut self.shared;
        for branch in &mut self.branches {
            run_branch(owner, &mut branch.state, shared, &mut f);
        }
    }
}

impl Drop for FallbackRecognizer<'_> {
    fn drop(&mut self) {
        let fallback = self.owner.shared_box.lexer_backtracking.take();
        self.owner.shared_box = std::mem::take(&mut self.shared);
        self.owner.shared_box.lexer_backtracking = fallback;
    }
}

impl Recognizer for FallbackRecognizer<'_> {
    fn pop_bytes(&mut self, num: usize) {
        for _ in 0..num {
            let owner = &mut self.owner;
            let shared = &mut self.shared;
            for branch in &mut self.branches {
                let previous = branch.active_history.pop().unwrap();
                if branch.active {
                    run_branch(owner, &mut branch.state, shared, |state| {
                        ParserRecognizer { state }.pop_bytes(1)
                    });
                }
                branch.active = previous;
            }
            self.depth -= 1;
            self.revert_changes_after(self.depth);
        }
    }

    fn collapse(&mut self) {}

    fn trie_started(&mut self, label: &str) {
        self.for_each_branch(|state| state.trie_started_inner(label));
    }

    fn trie_finished(&mut self) {
        self.revert_all_changes();
        self.for_each_branch(ParserState::trie_finished_inner);
        for branch in &mut self.branches {
            branch.active = true;
            branch.active_history.clear();
            branch.pending_step = None;
        }
        self.depth = 0;
    }

    fn save_stats(&mut self, nodes_walked: usize) {
        self.owner.stats.trie_nodes_walked += nodes_walked;
    }

    fn try_push_byte(&mut self, byte: u8) -> bool {
        self.prepare_steps(byte);
        self.add_hidden_fallback(byte);

        let mut any = false;
        for idx in 0..self.branches.len() {
            let was_active = self.branches[idx].active;
            self.branches[idx].active_history.push(was_active);
            if was_active {
                let step = self.branches[idx].pending_step.take().unwrap();
                let active = self.apply_step(idx, byte, step);
                self.branches[idx].active = active;
                any |= active;
            }
        }

        if any {
            self.depth += 1;
            self.prune_superseded();
        } else {
            for branch in &mut self.branches {
                branch.active = branch.active_history.pop().unwrap();
            }
            self.revert_changes_after(self.depth);
        }
        any
    }
}

#[inline(never)]
pub(super) fn compute_bias(
    state: &mut ParserState,
    computer: &dyn BiasComputer,
    start: &[u8],
) -> SimpleVob {
    let mut set = computer.trie().alloc_token_set();
    computer
        .trie()
        .add_bias(&mut FallbackRecognizer::new(state), &mut set, start);
    set
}

#[inline(never)]
pub(super) fn forced_byte(state: &mut ParserState) -> Option<u8> {
    if state.is_accepting() {
        return None;
    }
    let mut recognizer = FallbackRecognizer::new(state);
    recognizer.trie_started("forced_byte");
    let forced = {
        let mut allowed = (u8::MIN..=u8::MAX).filter(|&byte| recognizer.byte_allowed(byte));
        allowed.next().filter(|_| allowed.next().is_none())
    };
    recognizer.trie_finished();
    forced
}

#[inline(never)]
pub(super) fn chop_tokens(
    state: &mut ParserState,
    trie: &TokTrie,
    tokens: &[TokenId],
) -> (usize, usize) {
    trie.chop_tokens(&mut FallbackRecognizer::new(state), tokens)
}

#[inline(never)]
pub(super) fn validate_tokens(parser: &mut Parser, tokens: &[TokenId]) -> usize {
    let mut copy = parser.clone();
    let before = copy.state.stats.clone();
    let tok_env = copy.state.tok_env.clone();
    let trie = tok_env.tok_trie();
    let mut accepted = 0;

    for &token in tokens {
        if trie.eos_tokens().contains(&token) {
            if copy.is_accepting() {
                accepted += 1;
            }
            break;
        }
        let bytes = trie.decode_raw(&[token]);
        if copy.apply_token(&bytes, token).is_err() {
            break;
        }
        accepted += 1;
    }

    merge_work_stats(&mut parser.state, &before, &copy.state);
    accepted
}

#[inline(never)]
pub(super) fn recover(state: &mut ParserState, byte: Option<u8>, flush_end: bool) -> (bool, usize) {
    with_fallback(state, |fallback| fallback.recover(byte, flush_end))
}

#[inline(never)]
pub(super) fn forced_byte_committed(state: &mut ParserState, byte: u8) {
    with_fallback(state, |fallback| fallback.forced_byte_committed(byte));
}

#[inline(never)]
pub(super) fn token_committed(state: &mut ParserState, bytes: &[u8], token: TokenId) {
    with_fallback(state, |fallback| fallback.token_committed(bytes, token));
}

#[inline(never)]
pub(super) fn discard_fallback(state: &mut ParserState) {
    with_fallback(state, |fallback| fallback.discard_fallback());
}

#[inline(never)]
pub(super) fn accepting_allows_eos(state: &mut ParserState) -> bool {
    state.shared_box.lexer_backtracking.is_some()
        && with_fallback(state, |fallback| fallback.accepting_allows_eos())
}

#[inline(never)]
pub(super) fn refresh_fallback(state: &mut ParserState) {
    with_fallback(state, |fallback| fallback.refresh_fallback());
}

#[inline(never)]
pub(super) fn prepare_rollback(state: &mut ParserState, target: usize) {
    if state.shared_box.lexer_backtracking.is_some() {
        with_fallback(state, |fallback| fallback.prepare_rollback(target));
    }
}

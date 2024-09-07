use anyhow::{ensure, Result};
use toktrie::{Splice, StepArg, StepResult, TokenId};

use crate::{
    output::{ParserOutput, Reporter},
    TokenParser,
};

#[derive(Clone)]
pub struct Constraint {
    pub parser: TokenParser,
    pub log_json_progress: bool,
    pub temperature: f32,
    reporter: Reporter,
    step_arg: Option<StepArg>,
    last_res: StepResult,
    delayed_stop: bool,
    started: bool,
}

impl Constraint {
    /// Construct a state machine for a sequence constraint.
    pub fn new(parser: TokenParser) -> Self {
        assert!(parser.is_fresh(), "Parser was already used");
        Self {
            reporter: Reporter::new(&parser),
            parser,
            step_arg: Some(StepArg::empty()),
            last_res: StepResult::noop(),
            delayed_stop: false,
            started: false,
            log_json_progress: false,
            temperature: 0.0,
        }
    }

    fn save_progress_and_result(&mut self, res: &StepResult) {
        if let Some(temp) = res.temperature {
            self.temperature = temp;
        }
        self.last_res = res.clone();
        if self.log_json_progress {
            for p in self.reporter.get_progress(&mut self.parser, &self.last_res) {
                self.parser.logger.write_buffer("JSON-OUT: ");
                self.parser
                    .logger
                    .write_buffer(&serde_json::to_string(&p).unwrap());
                self.parser.logger.write_buffer("\n");
            }
        }
    }

    /// You can call this first with the prompt from the user, when not
    /// running in chat mode.
    /// This will return a new prompt, possibly with some tokens added as per
    /// the grammar (and possibly with some tokens removed, for token healing).
    pub fn process_prompt(&mut self, prompt: Vec<TokenId>) -> Vec<TokenId> {
        assert!(!self.started);
        self.started = true;
        self.parser.process_prompt(prompt)
    }

    /// This computes token sampling mask.
    /// It typically takes up to a millisecond for a 100k tokenizer.
    /// It will return an error when the order of calls is violated.
    /// The result will be either:
    /// - a mask of allowed tokens to sample, or
    /// - an unconditional splice result, indicating that the parser wants to append tokens, or
    /// - a stop result, indicating that the parser is done
    /// The splice is never returned when ff_tokens are disabled in InferenceCapabilities.
    /// After this returns, commit_token() must be called with the sampled token if any.
    pub fn compute_mask(&mut self) -> Result<StepResult> {
        if !self.started {
            self.started = true;
            self.parser.start_without_prompt();
        }

        if self.delayed_stop {
            self.delayed_stop = false;
            let stop = StepResult::stop();
            self.save_progress_and_result(&stop);
            return Ok(stop);
        }

        ensure!(!self.last_res.is_stop(), "compute_bias() called after stop");
        ensure!(
            self.step_arg.is_some(),
            "commit_token() not called before compute_bias()"
        );
        let step_arg = self.step_arg.take().unwrap();
        let res = self.parser.mid_process(step_arg);
        self.save_progress_and_result(&res);
        Ok(res)
    }

    /// This commits the sampled token (if any), and sees if this forces any more tokens
    /// on the output (if ff_tokens are enabled in InferenceCapabilities).
    pub fn commit_token(&mut self, sampled_token: Option<TokenId>) -> Result<StepResult> {
        ensure!(
            self.step_arg.is_none(),
            "commit_token() called twice or without compute_bias()"
        );

        // if last result was to stop or to unconditionally splice, we're done already
        if self.last_res.is_stop() {
            return Ok(self.last_res.clone());
        }

        if let Some(splice) = self.last_res.unconditional_splice() {
            // if ff_tokens are not supported, there should always be sampling mask
            // and not an unconditional splice
            assert!(self.parser.inference_caps.ff_tokens);

            // prepare argument for the next step
            self.step_arg = Some(StepArg::from_splice(splice, sampled_token));
            return Ok(self.last_res.clone());
        }

        // otherwise, append the sampled token and see if more tokens can be forced
        ensure!(
            self.last_res.sample_mask.is_some(),
            "internal error: invalid mid_process() result"
        );
        let mask = self.last_res.sample_mask.as_ref().unwrap();

        ensure!(
            sampled_token.is_some(),
            "sampled_token is required when mask was present"
        );
        let sampled_token = sampled_token.unwrap();

        // check if token is allowed
        ensure!(
            mask.is_allowed(sampled_token),
            "sampled_token was not allowed by the mask"
        );

        // if ff_tokens are not supported, just commit the sampled token
        if !self.parser.inference_caps.ff_tokens {
            self.step_arg = Some(StepArg::from_sampled_token(sampled_token));
            self.last_res = StepResult::splice(0, vec![sampled_token]);
            return Ok(self.last_res.clone());
        }

        // now, advance the parser with the sampled token - this should be very quick
        let pres = self.parser.advance_parser(StepArg {
            backtrack: 0,
            tokens: vec![sampled_token],
            sampled: Some(sampled_token),
        });

        // save any logs
        self.save_progress_and_result(&pres);

        // even if the result here is to stop, we still need to return an
        // unconditional splice with the sampled token, since the caller
        // needs to add it to their output
        let mut splice = if pres.is_stop() {
            self.delayed_stop = true;
            Splice::noop()
        } else {
            let splice = pres.unconditional_splice().unwrap().clone();
            // arg for the next step is just this splice, since the sampled token is already consumed
            self.step_arg = Some(StepArg::from_splice(&splice, None));
            splice
        };

        // however, we need to adjust the splice to account for the sampled token
        // when returning to the caller - they are going to update their state
        // based only on this result
        if splice.backtrack > 0 {
            splice.backtrack -= 1; // the sampled token was ignored
        } else {
            splice.ff_tokens.insert(0, sampled_token);
        }

        self.last_res = StepResult::splice(splice.backtrack, splice.ff_tokens.clone());
        Ok(self.last_res.clone())
    }

    /// This returns parser outputs to be passed back to the user.
    /// You can use that for structured output, or set log_json_progress to true
    /// and then use flush_logs() to get a string, from which the user
    /// can extract the JSON of the outputs.
    pub fn flush_progress(&mut self) -> Vec<ParserOutput> {
        self.reporter.get_progress(&mut self.parser, &self.last_res)
    }

    /// Logs to be sent to the user.
    pub fn flush_logs(&mut self) -> String {
        self.parser.logger.get_and_clear_logs()
    }
}
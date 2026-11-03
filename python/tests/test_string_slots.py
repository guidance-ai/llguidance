import json
import unittest
from concurrent.futures import ThreadPoolExecutor

from llguidance import LLGrammarTemplate, LLMatcher, LLTokenizer


class StringSlotsTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = LLTokenizer("byte")
        self.bundle = {
            "grammars": [
                {
                    "name": "main",
                    "lark_grammar": 'start: "[" @slot "," ESCAPED_STRING "]"\n%import common.ESCAPED_STRING',
                },
                {
                    "name": "slot",
                    "string_slot": "targets",
                    "lark_grammar": "start: ESCAPED_STRING\n%import common.ESCAPED_STRING",
                },
            ]
        }
        self.template = LLGrammarTemplate(self.tokenizer, self.bundle)

    def test_full_compilation_equivalence(self):
        for value in ["", "Kitchen", 'a"b\\c\n', "Étage 東京"]:
            literal = json.dumps(json.dumps(value, ensure_ascii=False), ensure_ascii=False)
            full = LLMatcher(
                self.tokenizer,
                'start: "[" VALUE "," ESCAPED_STRING "]"\nVALUE: ' + literal + "\n%import common.ESCAPED_STRING",
                log_level=0,
            )
            native = self.template.new_matcher({"targets": [value, value]})
            for token in json.dumps([value, "unrestricted"], ensure_ascii=False, separators=(",", ":")).encode():
                self.assertEqual(native.compute_bitmask(), full.compute_bitmask())
                self.assertTrue(native.consume_token(token))
                self.assertTrue(full.consume_token(token))
            self.assertTrue(native.is_accepting())

    def test_validation(self):
        self.assertEqual(self.template.slot_names, ["targets"])
        for value in ({"unknown": ["x"]}, {"targets": []}, {"targets": "bad"}, {"targets": [1]}):
            with self.assertRaises((ValueError, TypeError)):
                self.template.new_matcher(value)
        bounded = LLGrammarTemplate(self.tokenizer, self.bundle, max_binding_bytes=2, max_binding_values=1)
        for value in (["abc"], ["a", "a"]):
            with self.assertRaises(ValueError):
                bounded.new_matcher({"targets": value})
        self.assertFalse(bounded.new_matcher({"targets": ["a"]}).is_error())

    def test_concurrent_fallback_and_bindings(self):
        def run(i):
            matcher = self.template.new_matcher({"targets": [str(i)]} if i % 2 else {})
            for token in json.dumps([str(i), "free"], separators=(",", ":")).encode():
                matcher.compute_bitmask()
                self.assertTrue(matcher.consume_token(token))
            self.assertTrue(matcher.is_accepting())

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(run, range(32)))

    def test_binding_replaces_fallback_and_uses_canonical_json(self):
        template = LLGrammarTemplate(
            self.tokenizer, {"grammars": [{"string_slot": "names", "lark_grammar": 'start: /"[A-Z]+"/'}]}
        )

        def accepts(matcher, text):
            tokens = list(text.encode("utf-8"))
            return matcher.validate_tokens(tokens) == len(tokens)

        self.assertFalse(accepts(template.new_matcher(), '"Kitchen"'))
        self.assertTrue(accepts(template.new_matcher({"names": ["Kitchen"]}), '"Kitchen"'))
        self.assertFalse(accepts(template.new_matcher({"names": ["Kitchen"]}), '"OTHER"'))
        self.assertTrue(accepts(template.new_matcher({"names": ["é"]}), '"é"'))
        self.assertFalse(accepts(template.new_matcher({"names": ["é"]}), r'"\u00e9"'))

    def test_token_limit_matches_full_compilation(self):
        bundle = {**self.bundle, "max_tokens": 2}
        template = LLGrammarTemplate(self.tokenizer, bundle)
        for bindings in [None, {"targets": ["Kitchen"]}]:
            full_bundle = json.loads(json.dumps(bundle))
            if bindings:
                full_bundle["grammars"][1]["lark_grammar"] = 'start: "\\"Kitchen\\""'
                del full_bundle["grammars"][1]["string_slot"]
            bound = template.new_matcher(bindings)
            full = LLMatcher(self.tokenizer, json.dumps(full_bundle), log_level=0)
            for token in b'["Kitchen"':
                self.assertEqual(bound.compute_bitmask(), full.compute_bitmask())
                self.assertEqual(bound.consume_token(token), full.consume_token(token))
                self.assertEqual(bound.stop_reason(), full.stop_reason())
                if bound.is_stopped():
                    break

    def test_lexer_budget_failure_keeps_template_usable(self):
        from llguidance import LLParserLimits

        template = LLGrammarTemplate(
            self.tokenizer, {"grammars": [self.bundle["grammars"][1]]}, limits=LLParserLimits(initial_lexer_fuel=2000)
        )
        with self.assertRaises(ValueError):
            template.new_matcher({"targets": ["x" * 50000]})
        self.assertFalse(template.new_matcher({"targets": ["Kitchen"]}).is_error())


if __name__ == "__main__":
    unittest.main()

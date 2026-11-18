import base64
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request

from serve.grammar import MAX_BYTES, engine_key, normalize, validate_request
from serve.gbnf import SchemaUnsupported, for_response_format, schema_to_gbnf
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve
from serve.frontend import ChatTemplate


class GrammarApi(unittest.TestCase):
    def test_round_trip(self):
        text = 'root ::= "yes" | "no"'
        key = engine_key(text)
        self.assertTrue(key.startswith(" grammar=b64:"))
        self.assertEqual(base64.urlsafe_b64decode(key[len(" grammar=b64:"):]).decode(), text)
        self.assertEqual(engine_key(None), "")
        self.assertEqual(engine_key(""), "")
        self.assertEqual(StrataEngine.sampling_keys({"grammar": text}), key)

    def test_invalid(self):
        for value in [True, 1, {}, [], "", "   ", "root\x00 ::= \"a\""]:
            with self.subTest(value=repr(value)[:40]), self.assertRaises(ValueError):
                normalize(value)
        with self.assertRaises(ValueError):
            normalize("a" * (MAX_BYTES + 1))

    def test_capability(self):
        req = {"grammar": 'root ::= "a"'}
        with self.assertRaises(ValueError):
            validate_request(req, SimpleNamespace(info={}))
        with self.assertRaises(ValueError):        # the split engine cannot carry it
            validate_request(req, SimpleNamespace(can_grammar=True, split_engine=True))
        self.assertIsNotNone(validate_request(req, SimpleNamespace(can_grammar=True)))
        self.assertIsNone(validate_request({}, SimpleNamespace()))


class Gbnf(unittest.TestCase):
    def test_object(self):
        g = schema_to_gbnf({"type": "object", "properties": {"a": {"type": "string"},
                                                             "b": {"type": "integer"}},
                            "required": ["a"], "additionalProperties": False})
        self.assertIn('root ::= r0', g)
        self.assertIn('\\"a\\"', g)
        self.assertIn('r1', g)
        self.assertIn('"}"', g)

    def test_enum_and_const(self):
        g = schema_to_gbnf({"type": "object", "properties": {"k": {"enum": ["x", "y"]}},
                            "required": ["k"]})
        self.assertIn('\\"x\\"', g)
        self.assertIn('\\"y\\"', g)
        g = schema_to_gbnf({"type": "object", "properties": {"k": {"const": 3}}, "required": ["k"]})
        self.assertIn('"3"', g)

    def test_optional_subsets(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"},
                                                   "c": {"type": "string"}}, "required": ["a"]}
        g = schema_to_gbnf(schema)
        self.assertIn("|", g)              # 2**2 subsets of b and c

    def test_too_many_optionals(self):
        props = {chr(ord("a") + i): {"type": "string"} for i in range(6)}
        with self.assertRaises(SchemaUnsupported):
            schema_to_gbnf({"type": "object", "properties": props})

    def test_arrays(self):
        g = schema_to_gbnf({"type": "array", "items": {"type": "integer"}})
        self.assertIn("[", g)
        g = schema_to_gbnf({"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}]})
        self.assertIn("r1", g)

    def test_unions_and_refs(self):
        g = schema_to_gbnf({"type": "object", "properties": {"x": {"anyOf": [{"type": "string"},
                                                                           {"type": "null"}]}},
                            "required": ["x"]})
        self.assertIn("|", g)
        recursive = {"type": "object", "properties": {"name": {"type": "string"},
                                                      "child": {"$ref": "#"}},
                     "required": ["name"]}
        g = schema_to_gbnf(recursive)          # the reference cycle must not loop the converter
        self.assertIn("r0", g)

    def test_rejected_features(self):
        for node in [{"type": "string", "pattern": "^a"},
                     {"type": "string", "format": "date-time"},
                     {"type": "integer", "minimum": 0},
                     {"type": "array", "items": {"type": "string"}, "minItems": 1},
                     {"type": "string", "not": {}},
                     {"$ref": "https://example.com/s.json"},
                     {"type": "object", "additionalProperties": {"type": "string"}}]:
            with self.subTest(node=str(node)[:50]), self.assertRaises(SchemaUnsupported):
                schema_to_gbnf(node)

    def test_response_format(self):
        rf = {"type": "json_schema", "json_schema": {"name": "x",
                                                     "schema": {"type": "object", "properties": {},
                                                                "additionalProperties": False}}}
        self.assertIsNotNone(for_response_format(rf))
        self.assertIsNone(for_response_format({"type": "json_object"}))
        self.assertIsNone(for_response_format({"type": "text"}))
        bad = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "string",
                                                                              "pattern": "a"}}}
        self.assertIsNone(for_response_format(bad))    # keeps prompt-then-validate


class GrammarHttp(unittest.TestCase):
    def test_rejected_before_stream(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "ok", max_context=4096)      # no can_grammar: the key must be refused
        service = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(service, port=0)
        try:
            for body in [{"grammar": 'root ::= "a"'},
                         {"grammar": "", "messages": [{"role": "user", "content": "hi"}]},
                         {"grammar": "root ::= \"a\"", "tools": [{"type": "function",
                                                                  "function": {"name": "f", "parameters": {}}}]}]:
                req = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "stream": True, **body}
                data = json.dumps(req).encode()
                request = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                                                 data=data, headers={"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request, timeout=10)
                with caught.exception as response:
                    self.assertEqual(response.code, 400)
                    self.assertIn("grammar", json.loads(response.read())["error"]["message"])
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()

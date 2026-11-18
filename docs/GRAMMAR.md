# Grammar-constrained decoding

A grammar tells the engine what the answer may look like *while it is being written*. At every step the engine masks
every token the grammar would not accept, so the answer cannot leave the grammar — not at the first token, not in the
middle. Asking for JSON with a grammar is different from asking for JSON in the prompt: the model can answer badly,
but it cannot answer in the wrong shape.

This is the same GBNF format llama.cpp uses. The algorithm is llama.cpp's `src/llama-grammar.cpp` (MIT licence, kept
in `src/grammar/`, unchanged apart from the small vocabulary interface it is handed).

## Asking for one

Send a GBNF grammar as `grammar` on a chat completion (or a Responses request):

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Answer yes or no: is the sky blue?"}],
  "max_tokens": 8,
  "chat_template_kwargs": {"enable_thinking": false},
  "grammar": "root ::= \"yes\" | \"no\""
}'
```

The answer is exactly `yes` or `no`, whatever the temperature. A grammar is also what `response_format: json_schema`
uses under the hood when the engine can serve one, so most people never write one by hand (see
[DETAILS.md](DETAILS.md), "JSON output").

Thinking: a grammar constrains the **first generated token**, so a request that would open a thinking block gets none.
Turn thinking off for grammar requests (`enable_thinking: false` as above) or the constrained text lands in the
reasoning channel.

## What works and what does not

- The engine must report `grammar=1` in its `INFO` line and `grammar` in `READY`. That is the CUDA/HIP engine with a
  native pack (the model file carries its own tokenizer). The split engine (all cards) and the Qwen3.6 engine do not
  serve grammars; a request that asks for one is refused with 400 rather than quietly ignoring it.
- A grammar with tools or MCP is refused: an unconstrained tool call and a constrained answer cannot be the same
  request.
- Grammar requests decode **one token per round**: the mask is exact for the next token only, and the engine's
  speculative windows (MTP / DFlash2) would run ahead of it. Expect a lower tokens/s than the same prompt without a
  grammar; the engine still reads the prompt the same way.
- Limits, all enforced in the engine: 64 KiB of grammar text, 262,144 grammar elements, nesting depth 128, 4,096 parse
  states, parse-stack depth 1,024. Past a limit the request fails with 400 and a message naming it; nothing hangs.
- Only the first grammar body is supported: `lazy` grammars and trigger words are not exposed.
- On an engine running with `--batch`/`--parallel`, a grammar request runs on the plain single-request path (other
  requests wait their turn); it is never moved into a batch slot, whose windows run ahead of the mask.

## JSON Schema conversion

`response_format: {"type": "json_schema", ...}` is converted to GBNF when every feature of the schema is supported:
`type` (object, array, string, number, integer, boolean, null, or a union of these), `properties`, `required`,
`additionalProperties: false`, `items`, `prefixItems`, `enum`, `const`, `anyOf`, `oneOf`, `allOf` over objects, local
`#` references (including recursion), and objects with at most four optional properties. Keys are emitted in
declaration order and never repeated, because the validator refuses duplicate keys.

Anything else — `pattern`, `format`, the numeric/string/array bounds, `uniqueItems`, `not`, `if`/`then`/`else`,
`additionalProperties` schemas, remote references — is **not** converted. The request still works exactly as it did:
prompted for the shape, validated afterwards. The converter never emits a grammar that is looser than the validator;
it either matches it or keeps the old path.

## What it costs

Measured on one machine, one run each, so treat them as a direction rather than a benchmark: RTX 3090 (24 GB),
Windows 11, the Swift Qwen3.8-Flash-Next GSQ-RCO IQ2_XS pack (two shards), CUDA build, `--spec 4`, greedy answers of
57–78 tokens:

| request | tokens/s |
| --- | --- |
| no grammar (speculative decoding on) | 78.5 |
| a free-text grammar | 59.2 |
| a generated JSON-schema grammar | 34.7 |

Two things account for it. First, a grammar request decodes one token per round (speculation off), which is most of the
free-text gap. Second, building a mask for a grammar state that has not been seen before costs about 30–100 ms over this
pack's 248,320 tokens (a full vocabulary scan); a state that repeats — every comma, every colon, every character inside
a JSON string — costs about 13 µs, because the allowed set is cached per state. The vocabulary is decoded into token
code points once per engine process (about 30 ms), not once per request. A cold-miss number is static analysis of the
mask path, measured with a benchmark that walks a fixed JSON text; the tokens/s figures are end-to-end requests.

Correctness on the same run: 8 of 8 requests returned exactly `yes` or `no` under `root ::= "yes" | "no"` at
temperature 1, and 10 of 10 requests returned JSON that passed `jsonschema` against the nested schema above.

## The engine protocol

The server sends `grammar=b64:<base64url>` (and optionally `grammar_root=<name>`) on the `GEN` line. The engine reads
the model's tokenizer from the GGUF metadata shard the first time a request carries a grammar (a few MB of pieces,
once), then applies a full-vocabulary logit bias per step: `-inf` for every token the grammar refuses. The request's
own `logit_bias` is kept on the tokens the grammar allows.

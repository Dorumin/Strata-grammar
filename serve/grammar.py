"""Request-scoped GBNF grammars for constrained decoding.

A grammar makes the engine mask every token the grammar does not accept, so a structured answer is valid by
construction rather than by prompting and checking afterwards. The engine takes it on the GEN line as
`grammar=b64:<base64url>` (a GEN key's value cannot hold a space), which is what `engine_key` returns.

Normalize before HTTP streaming starts: a bad grammar is a 400, never a reply that fails later.
"""
import base64
import threading
from collections import OrderedDict

#: The engine's own limit (src/grammar/grammar.h, kMaxGrammarBytes). Checked here so an oversized grammar is a 400
#: before anything is sent.
MAX_BYTES = 64 * 1024

#: "root" is the rule name GBNF grammars use; the engine's default matches.
DEFAULT_ROOT = "root"


def normalize(value):
    """The GBNF text, or None when the request carries no grammar. Raises ValueError for anything unusable."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("grammar: expected a GBNF grammar string")
    if not value.strip():
        raise ValueError("grammar: the GBNF string is empty")
    if "\x00" in value:
        raise ValueError("grammar: the GBNF string contains a NUL byte")
    data = value.encode("utf-8")
    if len(data) > MAX_BYTES:
        raise ValueError(f"grammar: {len(data)} bytes, the limit is {MAX_BYTES}")
    return value


# A grammar is sent on every request of an agent loop and is usually the same text each time; cache the encoded key
# so a long one is not base64-encoded again per request. Keyed by the text itself (exact), bounded in size.
_CACHE_MAX = 64
_cache = OrderedDict()   # text -> " grammar=b64:..."
_lock = threading.Lock()


def engine_key(value):
    """The GEN-line fragment for `value`, or "" when there is no grammar (a blank string is refused by
    `validate_request`, so treating it as absent here only keeps a bad value from raising in a sampling call)."""
    if value is None or value == "":
        return ""
    text = normalize(value)
    with _lock:
        hit = _cache.get(text)
        if hit is not None:
            _cache.move_to_end(text)
            return hit
    key = " grammar=b64:" + base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")
    with _lock:
        _cache[text] = key
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return key


def available(engine):
    """Whether `engine` can constrain decoding. The READY line says so only when the model's own tokenizer can
    serve it; an older engine says nothing and is refused rather than sent a key it would ignore."""
    return getattr(engine, "can_grammar", False) is True


def validate_request(req, engine):
    """Normalize the request's grammar and refuse it when the engine cannot serve one. Returns the text or None."""
    text = normalize(req.get("grammar"))
    if text is None:
        return None
    if getattr(engine, "split_engine", False):
        raise ValueError("grammar: not supported by the all-cards split engine; start with --split-engine off")
    if not available(engine):
        raise ValueError("grammar: this engine does not support constrained decoding; it needs a CUDA/HIP build "
                         "with a native pack (an engine older than the feature refuses the key)")
    return text

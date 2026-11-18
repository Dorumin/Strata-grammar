"""JSON Schema -> GBNF, for engines that constrain decoding.

The converter is deliberately narrow, and it fails closed: a schema feature it does not understand is refused, so the
caller keeps the old prompt-then-validate path rather than sending the engine a grammar that is looser than the
validator (which is how the 502s would come back). It never errors for a schema the server would otherwise accept.

Supported: type (object/array/string/number/integer/boolean/null, or a list), properties, required,
additionalProperties false, items, prefixItems, enum, const, anyOf/oneOf, allOf over objects, local $ref (recursion
included), description/title/default/$schema/$id (ignored).

Refused with SchemaUnsupported: pattern, format, the numeric/string/array bounds, uniqueItems, not, if/then/else,
dependentSchemas, propertyNames, remote or $dynamicRef references, and objects with more than MAX_OPTIONAL optional
properties (each subset of optional keys is one branch, so the grammar can never repeat or reorder a key).

Keys are emitted in declaration order and never repeated: the server rejects duplicate JSON keys
(serve/structured.py, validated_json), so a grammar that could repeat one would produce output the validator refuses.
"""
import json

#: How many optional properties an object may have. Every subset is one grammar branch (2**n), and the branch's keys
#: are emitted in declaration order, which is what keeps a key from appearing twice.
MAX_OPTIONAL = 4

_COMMON = r"""
ws ::= [ \t\r\n]*
string ::= "\"" schar* "\""
schar ::= [^"\\\x00-\x1f\x7f] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] )
number ::= "-"? int frac? exp?
int ::= "0" | [1-9] [0-9]*
frac ::= "." [0-9]+
exp ::= [eE] [+-]? [0-9]+
integer ::= "-"? int
boolean ::= "true" | "false"
null ::= "null"
""".strip()

_JSON_TYPES = ("object", "array", "string", "number", "integer", "boolean", "null")


class SchemaUnsupported(ValueError):
    """The schema uses something the converter cannot express; the caller keeps prompt-then-validate."""


def _lit(s):
    """A GBNF string literal matching `s` exactly (code points, not UTF-8 bytes: the grammar compares code points)."""
    out = []
    for ch in s:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif o < 0x20 or o == 0x7F:
            out.append("\\x%02x" % o)
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class _Gen:
    def __init__(self, root):
        self.root = root
        self.names = {}      # id(node) -> rule name
        self.pending = []    # (name, node)
        self.rules = []      # (name, body)

    def name_for(self, node):
        if not isinstance(node, dict):
            raise SchemaUnsupported("a schema must be an object")
        nid = id(node)
        name = self.names.get(nid)
        if name is None:
            name = "r%d" % len(self.names)
            self.names[nid] = name
            self.pending.append((name, node))
        return name

    def resolve(self, ref):
        if not isinstance(ref, str) or not ref.startswith("#"):
            raise SchemaUnsupported("only local $ref references are supported")
        node = self.root
        for part in ref[1:].split("/")[1:]:
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, dict) or part not in node:
                raise SchemaUnsupported("unresolvable $ref %r" % ref)
            node = node[part]
        return node

    def run(self, root_name):
        self.name_for(self.root)
        root_rule = self.names[id(self.root)]
        while self.pending:
            name, node = self.pending.pop(0)
            self.rules.append((name, self.body(node)))
        return "\n".join([("root ::= " + root_rule)] + ["%s ::= %s" % (n, b) for n, b in self.rules]) + "\n" + _COMMON

    # ---- one schema node -> a GBNF expression
    def body(self, node):
        if not isinstance(node, dict):
            raise SchemaUnsupported("a schema must be an object")
        for key in ("pattern", "patternProperties", "format", "not", "if", "then", "else", "dependentSchemas",
                    "propertyNames", "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum",
                    "exclusiveMaximum", "multipleOf", "minItems", "maxItems", "uniqueItems", "minProperties",
                    "maxProperties", "contains", "$dynamicRef", "$recursiveRef"):
            if key in node:
                raise SchemaUnsupported("%s is not supported" % key)
        if "$ref" in node:
            return self.name_for(self.resolve(node["$ref"]))
        if "const" in node:
            return _lit(_json_text(node["const"]))
        if isinstance(node.get("enum"), list) and node["enum"]:
            return " | ".join(_lit(_json_text(v)) for v in node["enum"])
        if "allOf" in node:
            return self.name_for(self._merge_all_of(node))
        if isinstance(node.get("anyOf"), list) and node["anyOf"]:
            return " | ".join(self.name_for(self._branch(v)) for v in node["anyOf"])
        if isinstance(node.get("oneOf"), list) and node["oneOf"]:
            # oneOf also demands exactly one match; jsonschema still enforces that (validate after the fact).
            return " | ".join(self.name_for(self._branch(v)) for v in node["oneOf"])
        types = node.get("type")
        if types is None:
            if "properties" in node or "required" in node:
                types = "object"
            elif "items" in node or "prefixItems" in node:
                types = "array"
            else:
                raise SchemaUnsupported("a schema with no type")
        if isinstance(types, str):
            types = [types]
        if not isinstance(types, list) or not types:
            raise SchemaUnsupported("type must be a string or a list of strings")
        bodies = [self._typed(node, t) for t in types]
        return bodies[0] if len(bodies) == 1 else "(" + " | ".join(bodies) + ")"

    def _branch(self, node):
        return self.resolve(node["$ref"]) if isinstance(node, dict) and "$ref" in node else node

    def _typed(self, node, t):
        if t == "object":
            return self._object(node)
        if t == "array":
            return self._array(node)
        if t == "string":
            return "string"
        if t == "number":
            return "number"
        if t == "integer":
            return "integer"
        if t == "boolean":
            return "boolean"
        if t == "null":
            return "null"
        raise SchemaUnsupported("unknown type %r" % (t,))

    def _merge_all_of(self, node):
        """allOf over object schemas: one object whose properties and required lists are the union."""
        merged = {"type": "object", "properties": {}, "required": []}
        branches = [self._branch(b) for b in node["allOf"]]
        if "properties" in node or "required" in node:
            branches.append({k: v for k, v in node.items() if k != "allOf"})
        for b in branches:
            if not isinstance(b, dict) or b.get("type") != "object":
                raise SchemaUnsupported("allOf is only supported over object schemas")
            merged["properties"].update(b.get("properties") or {})
            for r in b.get("required") or []:
                if r not in merged["required"]:
                    merged["required"].append(r)
            if b.get("additionalProperties") not in (None, False):
                raise SchemaUnsupported("additionalProperties schemas are not supported")
        return merged

    def _object(self, node):
        props = node.get("properties")
        required = node.get("required") or []
        if not isinstance(props, dict) or not props:
            if required:
                raise SchemaUnsupported("required properties need properties")
            if node.get("additionalProperties") not in (None, False):
                raise SchemaUnsupported("additionalProperties schemas are not supported")
            return 'ws "{" ws "}"'
        if node.get("additionalProperties") not in (None, False):
            raise SchemaUnsupported("additionalProperties schemas are not supported")
        for r in required:
            if r not in props:
                raise SchemaUnsupported("required property %r has no schema" % r)
        optional = [k for k in props if k not in required]
        if len(optional) > MAX_OPTIONAL:
            raise SchemaUnsupported("an object with %d optional properties (at most %d)" % (len(optional), MAX_OPTIONAL))
        refs = {k: self.name_for(v) for k, v in props.items()}
        branches = []
        for mask in range(1 << len(optional)):
            chosen = {optional[i] for i in range(len(optional)) if (mask >> i) & 1}
            keys = [k for k in props if k in required or k in chosen]
            if not keys:
                branches.append('"{" ws "}"')
                continue
            chunks = ['"{"']
            for i, k in enumerate(keys):
                if i:
                    chunks.append(' ws ","')
                chunks.append(' ws %s ws ":" ws %s' % (_lit(_json_text(k)), refs[k]))
            chunks.append(' ws "}"')
            branches.append("".join(chunks))
        return branches[0] if len(branches) == 1 else "(" + " | ".join(branches) + ")"

    def _array(self, node):
        items = node.get("items")
        prefix = node.get("prefixItems")
        if prefix is not None:
            if not isinstance(prefix, list):
                raise SchemaUnsupported("prefixItems must be a list")
            parts = [self.name_for(v) for v in prefix]
            chunks = ['"["']
            for i, ref in enumerate(parts):
                if i:
                    chunks.append(' ws ","')
                chunks.append(" ws %s" % ref)
            if items is not None and items is not False:
                # the prefix, then any number of the remaining items
                rest = self.name_for(items)
                chunks.append(' ( ws "," ws %s )*' % rest)
            chunks.append(' ws "]"')
            return "".join(chunks)
        if items is None or items is False:
            return '"[" ws "]"'
        ref = self.name_for(items)
        return '"[" ws ( %s ( ws "," ws %s )* )? ws "]"' % (ref, ref)


def schema_to_gbnf(schema):
    """The GBNF text for `schema`. Raises SchemaUnsupported when it cannot be expressed exactly."""
    if not isinstance(schema, dict):
        raise SchemaUnsupported("the schema must be an object")
    return _Gen(schema).run("root")


def for_response_format(response_format):
    """The grammar for an OpenAI `response_format`, or None when the old prompt-then-validate path should keep it.

    json_object is left unconverted on purpose: "any JSON object" needs arbitrary keys, and a grammar that allows any
    key can repeat one, which the validator refuses. json_schema is converted when every feature it uses is
    supported; the validator still runs either way.
    """
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") != "json_schema":
        return None
    spec = response_format.get("json_schema")
    schema = spec.get("schema") if isinstance(spec, dict) else None
    if not isinstance(schema, dict):
        return None
    try:
        return schema_to_gbnf(schema)
    except SchemaUnsupported:
        return None

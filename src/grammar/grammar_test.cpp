// Unit tests for the grammar core.  No model, no GPU, no GGUF: a vocabulary is built in memory so each test says
// exactly which pieces the grammar should accept.  Built by the CMake target `strata_grammar_test`.
#include "grammar.h"
#include "grammar_vocab.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace strata::grammar;

namespace {

int g_failures = 0;

void check(bool ok, const char* what, int line) {
    if (!ok) {
        std::printf("FAIL line %d: %s\n", line, what);
        ++g_failures;
    }
}

#define CHECK(x) check((x), #x, __LINE__)

// A vocabulary in id order.  Id 0 is the end of generation and a control token, as in a real model.
GrammarVocab make_vocab(std::vector<std::string> pieces, std::vector<int32_t> eog_ids = {0}) {
    GrammarVocab v;
    v.n_vocab = static_cast<int32_t>(pieces.size());
    v.pieces = std::move(pieces);
    const size_t n = v.pieces.size();   // pieces is moved from: never ask it for its size again
    v.eog.assign(n, 0);
    v.special.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        v.special[i] = 1;
        v.by_piece.emplace(v.pieces[i], static_cast<int32_t>(i));
    }
    for (size_t i = 1; i < n; ++i) v.special[i] = 0;
    for (const int32_t id : eog_ids) {
        if (id >= 0 && static_cast<size_t>(id) < n) v.eog[static_cast<size_t>(id)] = 1;
    }
    v.pieces[0].clear();  // a control token has an empty piece, exactly like the loader's
    return v;
}

bool allowed_at(const std::vector<uint8_t>& a, int32_t id) {
    return id >= 0 && static_cast<size_t>(id) < a.size() && a[static_cast<size_t>(id)] != 0;
}

void test_two_words() {
    // 0 = <eog>, 1 = "yes", 2 = "no"
    const GrammarVocab v = make_vocab({"", "yes", "no"});
    std::string err;
    auto g = Grammar::parse("root ::= \"yes\" | \"no\"", "root", v, err);
    CHECK(g != nullptr);
    if (!g) {
        std::printf("  parse error: %s\n", err.c_str());
        return;
    }
    CHECK(!g->complete());

    std::vector<uint8_t> a;
    CHECK(g->fill_allowed(a, err));
    CHECK(a.size() == 3);
    CHECK(!allowed_at(a, 0));  // EOG not allowed while the grammar is unfinished
    CHECK(allowed_at(a, 1));
    CHECK(allowed_at(a, 2));

    CHECK(g->accept(1, err));
    CHECK(g->complete());
    CHECK(g->fill_allowed(a, err));
    CHECK(allowed_at(a, 0));  // complete: end of generation allowed
    CHECK(!allowed_at(a, 1));  // and nothing else
    CHECK(!allowed_at(a, 2));
}

void test_character_by_character() {
    // A grammar over single-character pieces, the way a byte-level vocabulary splits a word.
    const GrammarVocab v = make_vocab({"", "y", "e", "s", "n", "o"});
    std::string err;
    auto g = Grammar::parse("root ::= \"yes\"", "root", v, err);
    CHECK(g != nullptr);
    if (!g) return;

    std::vector<uint8_t> a;
    CHECK(g->fill_allowed(a, err));
    CHECK(allowed_at(a, 1));  // "y"
    CHECK(!allowed_at(a, 2));

    CHECK(g->accept(1, err));
    CHECK(g->fill_allowed(a, err));
    CHECK(allowed_at(a, 2));  // "e"
    CHECK(!allowed_at(a, 3));

    CHECK(g->accept(2, err));
    CHECK(g->accept(3, err));
    CHECK(g->complete());
    CHECK(g->fill_allowed(a, err));
    CHECK(allowed_at(a, 0));
    CHECK(!allowed_at(a, 1));

    // An exhausted grammar does not accept another character.
    CHECK(!g->accept(4, err));
}

void test_json_like() {
    // 0 = <eog>, 1 "{", 2 "a", 3 ":", 4 "0", 5 "1", 6 "}", 7 "x"
    const GrammarVocab v = make_vocab({"", "{", "a", ":", "0", "1", "}"});
    std::string err;
    auto g = Grammar::parse("root ::= \"{\" \"a\" \":\" num \"}\"\nnum ::= \"0\" | \"1\"", "root", v, err);
    CHECK(g != nullptr);
    if (!g) {
        std::printf("  parse error: %s\n", err.c_str());
        return;
    }
    std::vector<uint8_t> a;
    CHECK(g->fill_allowed(a, err));
    // Only "{" may start; the other pieces are not in the first set.
    for (int32_t id = 0; id < 7; ++id) CHECK(allowed_at(a, id) == (id == 1));
    CHECK(g->accept(1, err));
    CHECK(g->fill_allowed(a, err));
    CHECK(allowed_at(a, 2) && !allowed_at(a, 3));
    CHECK(g->accept(2, err));
    CHECK(g->accept(3, err));
    CHECK(g->fill_allowed(a, err));
    CHECK(allowed_at(a, 4) && allowed_at(a, 5) && !allowed_at(a, 6));
    CHECK(g->accept(5, err));
    CHECK(g->fill_allowed(a, err));
    CHECK(allowed_at(a, 6));
    CHECK(g->accept(6, err));
    CHECK(g->complete());
}

void test_token_terminals() {
    // <[id]> names a token by number.  <token> names one by its exact vocabulary spelling, including the angle
    // brackets -- which in practice means a special token, spelled as the vocabulary spells it.
    const GrammarVocab v = make_vocab({"", "yes", "no", "<|im_end|>"});
    std::string err;
    auto g1 = Grammar::parse("root ::= <[2]>", "root", v, err);
    CHECK(g1 != nullptr);
    if (g1) {
        std::vector<uint8_t> a;
        CHECK(g1->fill_allowed(a, err));
        CHECK(allowed_at(a, 2) && !allowed_at(a, 1));
    }
    auto g2 = Grammar::parse("root ::= " + v.pieces[3], "root", v, err);
    CHECK(g2 != nullptr);
    if (g2) {
        std::vector<uint8_t> a;
        CHECK(g2->fill_allowed(a, err));
        CHECK(allowed_at(a, 3) && !allowed_at(a, 1) && !allowed_at(a, 2));
    }
}

void test_rejections() {
    const GrammarVocab v = make_vocab({"", "a"});
    std::string err;

    // Left recursion is refused, as upstream does.
    CHECK(Grammar::parse("root ::= root | \"a\"", "root", v, err) == nullptr);
    CHECK(!err.empty());

    // A missing root rule and a syntax error are refused.
    CHECK(Grammar::parse("root ::= \"a\"", "other", v, err) == nullptr);
    CHECK(Grammar::parse("root ::= \"unterminated", "root", v, err) == nullptr);

    // Too big.
    const std::string big(Grammar::kMaxGrammarBytes + 1, 'a');
    CHECK(Grammar::parse(big, "root", v, err) == nullptr);
    CHECK(err.find("limit") != std::string::npos);

    // Nesting deep enough to overflow the parser's recursion is refused before it runs.
    std::string nested = "root ::= ";
    nested.append(Grammar::kMaxNesting + 8, '(');
    nested += "\"a\"";
    nested.append(Grammar::kMaxNesting + 8, ')');
    CHECK(Grammar::parse(nested, "root", v, err) == nullptr);
    CHECK(err.find("nested") != std::string::npos);
}

void test_stuck_grammar() {
    // The vocabulary has no token that can start the rule: nothing is allowed and the caller must fail the request.
    const GrammarVocab v = make_vocab({"", "z"});
    std::string err;
    auto g = Grammar::parse("root ::= \"abc\"", "root", v, err);
    CHECK(g != nullptr);
    if (!g) return;
    std::vector<uint8_t> a;
    CHECK(!g->fill_allowed(a, err));
    CHECK(!err.empty());
}

void test_multibyte_piece() {
    // A single piece holding several code points, the way a BPE vocabulary has " hello" and "é".
    const GrammarVocab v = make_vocab({"", "\xC3\xA9", "hi"});
    std::string err;
    auto g = Grammar::parse("root ::= \"\xC3\xA9\" | \"hi\"", "root", v, err);
    CHECK(g != nullptr);
    if (!g) return;
    std::vector<uint8_t> a;
    CHECK(g->fill_allowed(a, err));
    CHECK(allowed_at(a, 1));  // é, one piece of two bytes
    CHECK(allowed_at(a, 2));
    CHECK(g->accept(1, err));
    CHECK(g->complete());
}

void test_eog_only_when_complete() {
    // An empty rule makes the grammar complete immediately: only EOG is allowed.
    const GrammarVocab v = make_vocab({"", "a", "b"});
    std::string err;
    auto g = Grammar::parse("root ::= [a-b]?", "root", v, err);
    CHECK(g != nullptr);
    if (!g) return;
    CHECK(g->complete());
    std::vector<uint8_t> a;
    CHECK(g->fill_allowed(a, err));
    CHECK(allowed_at(a, 0));
    CHECK(allowed_at(a, 1));  // and a/b are still available: the rule is optional, not finished
    CHECK(allowed_at(a, 2));
}

void test_lazy_trigger() {
    // The think-then-answer shape: the grammar waits for a trigger token (the chat's think-end, say), the thinking
    // runs free, and the mask starts at the token AFTER the trigger -- which the grammar itself never sees.
    // 0 = <eog>, 1 = "<end>" (the trigger), 2 "{", 3 "a", 4 "}", 5 "x".
    const GrammarVocab v = make_vocab({"", "<end>", "{", "a", "}", "x"});
    std::string err;
    auto g = Grammar::parse("root ::= \"{\" \"a\" \"}\"", "root", v, err, {1});
    CHECK(g != nullptr);
    if (!g) return;
    CHECK(!g->complete());

    std::vector<uint8_t> a;
    CHECK(g->fill_allowed(a, err));
    // While waiting, every ordinary piece is allowed -- the thinking runs free -- and the trigger token itself too.
    CHECK(allowed_at(a, 1));
    CHECK(allowed_at(a, 2));
    CHECK(allowed_at(a, 5));
    CHECK(!allowed_at(a, 0));  // EOG still waits for the grammar

    // Ordinary tokens pass through without touching the grammar.
    CHECK(g->accept(5, err));
    CHECK(g->fill_allowed(a, err));
    CHECK(allowed_at(a, 2));
    CHECK(allowed_at(a, 1));

    // The trigger fires the grammar without being consumed: the root's "{" must start right after it.
    CHECK(g->accept(1, err));
    CHECK(!g->complete());
    CHECK(g->fill_allowed(a, err));
    CHECK(allowed_at(a, 2));
    CHECK(!allowed_at(a, 5));  // the mask now holds
    CHECK(!allowed_at(a, 1));  // the trigger was a boundary, not part of the answer

    CHECK(g->accept(2, err));
    CHECK(g->accept(3, err));
    CHECK(g->accept(4, err));
    CHECK(g->complete());
    CHECK(g->fill_allowed(a, err));
    CHECK(allowed_at(a, 0));
    CHECK(!allowed_at(a, 2));
}

// The real model's metadata shard, when one is given: the pieces, the byte decoder, the EOG ids and a JSON grammar
// over the full vocabulary.  This is the part a synthetic vocabulary cannot check -- a GPT-2 piece table decoded
// wrongly still has the right ids, so only the real file shows it.
void test_real_vocab(const char* shard) {
    GrammarVocab v;
    std::string err;
    if (!load_vocab_from_gguf(shard, {248044, 248046}, v, err)) {
        std::printf("FAIL loading %s: %s\n", shard, err.c_str());
        ++g_failures;
        return;
    }
    std::printf("real vocab: %d pieces\n", v.n_vocab);
    CHECK(v.n_vocab > 100000);
    CHECK(v.pieces.size() == static_cast<size_t>(v.n_vocab));

    // A control token keeps its literal spelling; both the model's eos and the engine's stops end generation.
    CHECK(v.token_to_piece(248046) == "<|im_end|>");
    CHECK(v.is_eog(248046));
    CHECK(v.is_eog(248044));

    // Byte-level decoding: a space is a space, never the U+0120 ("\xC4\xA0") a raw GPT-2 piece table would leave
    // behind.  Count both across the vocabulary.
    size_t with_space = 0, with_encoded_space = 0, empty = 0;
    for (const std::string& p : v.pieces) {
        if (p.empty()) { ++empty; continue; }
        if (p.find(' ') != std::string::npos) ++with_space;
        if (p.find("\xC4\xA0") != std::string::npos) ++with_encoded_space;
    }
    std::printf("real vocab: %zu pieces hold a space, %zu still hold U+0120, %zu empty\n", with_space,
                with_encoded_space, empty);
    CHECK(with_encoded_space == 0);
    CHECK(with_space > 1000);

    // A JSON grammar over the real pieces: "{" must be allowed to start it, and after accepting it a quote may follow.
    std::string gerr;
    auto g = Grammar::parse("root ::= \"{\" ws key\nws ::= [ ]?\nkey ::= \"\\\"\"", "root", v, gerr);
    CHECK(g != nullptr);
    if (!g) {
        std::printf("FAIL real grammar: %s\n", gerr.c_str());
        return;
    }
    std::vector<uint8_t> allowed;
    CHECK(g->fill_allowed(allowed, gerr));
    int32_t brace = -1;
    for (int32_t id = 0; id < v.n_vocab; ++id)
        if (v.token_to_piece(id) == "{") { brace = id; break; }
    CHECK(brace >= 0);
    if (brace >= 0) {
        std::printf("real vocab: '{%s' is token %d\n", "", brace);
        CHECK(allowed_at(allowed, brace));
        CHECK(g->accept(brace, gerr));
        CHECK(g->fill_allowed(allowed, gerr));
        int32_t quote = -1;
        for (int32_t id = 0; id < v.n_vocab; ++id)
            if (v.token_to_piece(id) == "\"") { quote = id; break; }
        CHECK(quote >= 0);
        if (quote >= 0) CHECK(allowed_at(allowed, quote));
    }
    // A banned character and an unfinished grammar: EOG must not be allowed yet.
    CHECK(!allowed_at(allowed, 248046));
}

}  // namespace

int main(int argc, char** argv) {
    test_two_words();
    test_character_by_character();
    test_json_like();
    test_token_terminals();
    test_rejections();
    test_stuck_grammar();
    test_multibyte_piece();
    test_eog_only_when_complete();
    test_lazy_trigger();
    if (argc > 1) test_real_vocab(argv[1]);

    if (g_failures == 0) {
        std::printf("grammar_test: all tests passed\n");
        return 0;
    }
    std::printf("grammar_test: %d failure(s)\n", g_failures);
    return 1;
}

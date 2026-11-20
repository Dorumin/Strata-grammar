#pragma once

// A parsed GBNF grammar and the operations a decode loop needs from it: which token ids may be sampled now, advancing
// over the token that was sampled, and whether the grammar is complete.  The grammar algorithm is llama.cpp's (see
// llama_grammar.h); this type adds the limits that turn an untrusted grammar into a client error rather than a hung or
// oversized engine.
#include "grammar_vocab.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::grammar {

class Grammar {
public:
    // Parses `gbnf` with `root` as the start rule.  nullptr with `err` set when the text does not parse, is
    // left-recursive, or is over one of the limits below.  `vocab` must outlive the Grammar.
    //
    // `trigger_tokens` makes the grammar lazy: until one of these token ids is generated the mask constrains
    // nothing (the thinking part of a reply runs free), and the grammar takes over from the token AFTER the
    // trigger.  The trigger itself is a protocol boundary the grammar never sees.  The trigger token ids stay
    // samplable the whole time, or the grammar could never fire.
    static std::unique_ptr<Grammar> parse(const std::string& gbnf, const std::string& root, const GrammarVocab& vocab,
                                          std::string& err, const std::vector<int32_t>& trigger_tokens = {});
    ~Grammar();

    Grammar(const Grammar&) = delete;
    Grammar& operator=(const Grammar&) = delete;

    // Fills a 1/0 table indexed by token id: 1 = the grammar accepts that token in its current state.  EOG is allowed
    // only once the grammar is complete.  Returns false, with `err` set, when nothing at all is allowed (the grammar
    // is stuck) or the mask would cost more than `kMaskWorkBudget` -- the request must fail, not sample blindly.
    bool fill_allowed(std::vector<uint8_t>& allowed, std::string& err);

    // Advances over a generated token.  Returns false, with `err` set, when the grammar does not accept it (only
    // reachable in speculative rollback mode, where a token may have been produced under an earlier mask).
    bool accept(int32_t token, std::string& err);

    // True once some parse position has consumed the whole root rule: end of generation is allowed from here.
    bool complete() const;

    // Limits.  All are far above what a real grammar uses and low enough that a 64 KiB request cannot stall a serve.
    static constexpr size_t kMaxGrammarBytes = 64 * 1024;
    static constexpr size_t kMaxRuleElements = 262144;
    static constexpr size_t kMaxNesting = 128;
    static constexpr size_t kMaxStacks = 4096;
    static constexpr size_t kMaxStackDepth = 1024;
    static constexpr size_t kMaskWorkBudget = 200u * 1000u * 1000u;  // candidate x stack evaluations per mask

private:
    struct Impl;
    explicit Grammar(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::grammar

#pragma once

// The small slice of a tokenizer the GBNF grammar needs: each token's piece (raw bytes), whether it ends the
// generation, and whether it is a special token that the grammar must never sample.  Filled from the model's GGUF
// metadata shard (tokenizer.ggml.tokens / token_type / eos_token_id), which the engine already opens for geometry.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::grammar {

struct GrammarVocab {
    int32_t n_vocab = 0;
    std::vector<std::string> pieces;   // raw bytes per token; empty = not a grammar candidate
    std::vector<uint8_t> eog;          // 1 = end of generation (allowed when the grammar is complete)
    std::vector<uint8_t> special;      // 1 = control/unused token: never a grammar candidate (EOG handled first)

    // Exact-piece lookup for GBNF's textual <token> terminals (rare; numeric <[id]> needs no table).
    std::unordered_map<std::string, int32_t> by_piece;

    const std::string& token_to_piece(int32_t id) const;
    bool is_eog(int32_t id) const;
    bool is_special(int32_t id) const;

    // llama_vocab::tokenize's shape, limited to "the whole text is exactly one token".  Returns 1 and the id, else 0.
    int32_t tokenize(const char* text, int32_t len, int32_t* out, int32_t n_out, bool add_special,
                     bool special) const;
};

// Reads tokenizer.ggml.tokens / token_type / eos_token_id from the model's metadata shard.  `extra_eog` adds the
// engine's own stop ids (o.eos_ids).  False with `err` when the file has no tokenizer arrays; the engine then
// advertises no grammar support and refuses grammar requests rather than guessing.
bool load_vocab_from_gguf(const std::string& metadata_shard, const std::vector<int64_t>& extra_eog,
                          GrammarVocab& out, std::string& err);

}  // namespace strata::grammar

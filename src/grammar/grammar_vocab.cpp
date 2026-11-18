// The vocabulary slice the GBNF grammar needs, read from the model's own metadata shard.
//
// Two things here are easy to get wrong and impossible to see in the token ids:
//
//  * A GPT-2 style vocabulary stores each piece in the bytes-to-unicode encoding, so a space is U+0120 and a newline
//    is U+010A.  The grammar compares UTF-8 code points, so every normal piece has to be decoded back to raw bytes.
//    Getting this wrong does not change a single id, but every piece then looks like "Ġhello" and the grammar
//    rejects most tokens.
//  * A padded vocabulary has ids past the end of tokenizer.ggml.tokens.  Those get an empty piece and are always
//    masked; they must never be left selectable.
#include "grammar_vocab.h"

#include "strata/artifact/gguf_reader.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace strata::grammar {
namespace {

// GPT-2's bytes_to_unicode, inverted: code point -> the byte it stands for.  Built once.
const std::unordered_map<uint32_t, uint8_t>& byte_decoder() {
    static const std::unordered_map<uint32_t, uint8_t> table = [] {
        std::unordered_map<uint32_t, uint8_t> t;
        for (int b = '!'; b <= '~'; ++b) t[static_cast<uint32_t>(b)] = static_cast<uint8_t>(b);
        for (int b = 0xA1; b <= 0xAC; ++b) t[static_cast<uint32_t>(b)] = static_cast<uint8_t>(b);
        for (int b = 0xAE; b <= 0xFF; ++b) t[static_cast<uint32_t>(b)] = static_cast<uint8_t>(b);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (t.count(static_cast<uint32_t>(b))) continue;
            t[static_cast<uint32_t>(256 + n)] = static_cast<uint8_t>(b);
            ++n;
        }
        return t;
    }();
    return table;
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// One code point of a well-formed UTF-8 string; `len` is set to how many bytes it took.  A malformed sequence is
// taken as a single byte so that one bad piece can never desynchronise the rest of the vocabulary.
uint32_t next_code_point(const std::string& s, size_t& i, size_t& len) {
    const uint8_t b = static_cast<uint8_t>(s[i]);
    if (b < 0x80) {
        len = 1;
        return b;
    }
    int n = 0;
    uint32_t cp = 0;
    if ((b & 0xE0) == 0xC0) {
        n = 1;
        cp = b & 0x1F;
    } else if ((b & 0xF0) == 0xE0) {
        n = 2;
        cp = b & 0x0F;
    } else if ((b & 0xF8) == 0xF0) {
        n = 3;
        cp = b & 0x07;
    } else {
        len = 1;
        return b;
    }
    if (i + static_cast<size_t>(n) >= s.size()) {
        len = 1;
        return b;
    }
    for (int k = 1; k <= n; ++k) {
        const uint8_t c = static_cast<uint8_t>(s[i + static_cast<size_t>(k)]);
        if ((c & 0xC0) != 0x80) {
            len = 1;
            return b;
        }
        cp = (cp << 6) | (c & 0x3F);
    }
    len = static_cast<size_t>(n) + 1;
    return cp;
}

// The bytes-to-unicode encoding back to raw bytes.
std::string decode_bpe_piece(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    const auto& dec = byte_decoder();
    for (size_t i = 0; i < raw.size();) {
        size_t len = 0;
        const uint32_t cp = next_code_point(raw, i, len);
        const auto it = dec.find(cp);
        if (it != dec.end()) {
            out.push_back(static_cast<char>(it->second));
        } else {
            // Not in the table: a code point the encoder never produces, so it is already the literal character.
            append_utf8(out, cp);
        }
        i += len;
    }
    return out;
}

// SentencePiece keeps byte-fallback tokens under a "<0xXX>" name instead of encoding them.
bool spm_byte_piece(const std::string& raw, std::string& out) {
    if (raw.size() != 6 || raw[0] != '<' || raw[1] != '0' || raw[2] != 'x' || raw[5] != '>') return false;
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    const int hi = hex(raw[3]);
    const int lo = hex(raw[4]);
    if (hi < 0 || lo < 0) return false;
    out.assign(1, static_cast<char>((hi << 4) | lo));
    return true;
}

// gguf token_type values (llama.cpp's llama_token_type)
constexpr int kTokenTypeNormal = 1;
constexpr int kTokenTypeControl = 3;
constexpr int kTokenTypeUserDefined = 4;
constexpr int kTokenTypeUnused = 5;
constexpr int kTokenTypeByte = 6;

}  // namespace

const std::string& GrammarVocab::token_to_piece(int32_t id) const {
    static const std::string empty;
    if (id < 0 || static_cast<size_t>(id) >= pieces.size()) return empty;
    return pieces[static_cast<size_t>(id)];
}

bool GrammarVocab::is_eog(int32_t id) const {
    return id >= 0 && static_cast<size_t>(id) < eog.size() && eog[static_cast<size_t>(id)] != 0;
}

bool GrammarVocab::is_special(int32_t id) const {
    return id >= 0 && static_cast<size_t>(id) < special.size() && special[static_cast<size_t>(id)] != 0;
}

int32_t GrammarVocab::tokenize(const char* text, int32_t len, int32_t* out, int32_t n_out, bool /*add_special*/,
                               bool /*special*/) const {
    // GBNF's textual <token> terminal names a token by its exact vocabulary spelling; numeric <[id]> needs no lookup.
    if (n_out < 1 || text == nullptr || len < 0) return 0;
    const auto it = by_piece.find(std::string(text, static_cast<size_t>(len)));
    if (it == by_piece.end()) return 0;
    out[0] = it->second;
    return 1;
}

bool load_vocab_from_gguf(const std::string& metadata_shard, const std::vector<int64_t>& extra_eog, GrammarVocab& out,
                          std::string& err) {
    out = GrammarVocab{};
    err.clear();

    std::unique_ptr<strata::GgufFile> file;
    try {
        file = std::make_unique<strata::GgufFile>(metadata_shard);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }

    strata::MetaValue tokens;
    if (std::string e = file->full_array("tokenizer.ggml.tokens", tokens); !e.empty()) {
        err = e;
        return false;
    }
    if (tokens.type != strata::MetaType::ARRAY || tokens.elem != strata::MetaType::STRING || tokens.count == 0) {
        err = "tokenizer.ggml.tokens is not a list of strings";
        return false;
    }

    // The vocabulary model decides how a piece is spelled: GPT-2 stores bytes-to-unicode, SentencePiece stores
    // text with <0xXX> byte fallbacks.
    bool bpe = true;
    if (const strata::MetaValue* model = file->get("tokenizer.ggml.model")) {
        bpe = model->s != "llama" && model->s != "spm";
    }

    strata::MetaValue types;
    bool have_types = file->full_array("tokenizer.ggml.token_type", types).empty() &&
                      types.type == strata::MetaType::ARRAY &&
                      (types.elem == strata::MetaType::I32 || types.elem == strata::MetaType::U32) &&
                      types.items.size() == tokens.count;

    const int32_t n = static_cast<int32_t>(tokens.count);
    out.n_vocab = n;
    out.pieces.resize(static_cast<size_t>(n));
    out.eog.assign(static_cast<size_t>(n), 0);
    out.special.assign(static_cast<size_t>(n), 0);
    out.by_piece.reserve(static_cast<size_t>(n));

    for (int32_t i = 0; i < n; ++i) {
        const std::string& raw = tokens.items[static_cast<size_t>(i)].s;
        const int type = have_types ? static_cast<int>(static_cast<int32_t>(types.items[static_cast<size_t>(i)].u))
                                    : kTokenTypeNormal;

        if (type == kTokenTypeControl || type == kTokenTypeUserDefined || type == kTokenTypeUnused) {
            // Special tokens are spelled literally and are never grammar candidates (EOG is handled before this test).
            out.pieces[static_cast<size_t>(i)] = raw;
            out.special[static_cast<size_t>(i)] = 1;
        } else if (type == kTokenTypeByte) {
            std::string byte;
            out.pieces[static_cast<size_t>(i)] = spm_byte_piece(raw, byte) ? byte : decode_bpe_piece(raw);
        } else {
            std::string byte;
            // A SentencePiece vocabulary's byte fallback keeps its <0xXX> name even though its type is NORMAL.
            out.pieces[static_cast<size_t>(i)] = (!bpe && spm_byte_piece(raw, byte)) ? byte : decode_bpe_piece(raw);
        }

        // The textual <token> terminal is spelled exactly as the vocabulary spells it.
        if (!raw.empty() && raw[0] != 0) out.by_piece.emplace(raw, i);
    }

    // End of generation: the model's own id (a number, or a list of them in some files), plus whatever the engine is
    // already told to stop on.  A key of another shape is ignored: the engine's own stop ids still apply.
    if (const strata::MetaValue* eos = file->get("tokenizer.ggml.eos_token_id")) {
        if (eos->type == strata::MetaType::ARRAY) {
            for (const strata::MetaValue& e : eos->items) {
                const int64_t id = static_cast<int64_t>(e.u);
                if (id >= 0 && id < n) out.eog[static_cast<size_t>(id)] = 1;
            }
        } else if (eos->is_num()) {
            const int64_t id = static_cast<int64_t>(eos->u);
            if (id >= 0 && id < n) out.eog[static_cast<size_t>(id)] = 1;
        }
    }
    for (const int64_t id : extra_eog) {
        if (id >= 0 && id < n) out.eog[static_cast<size_t>(id)] = 1;
    }

    return true;
}

}  // namespace strata::grammar

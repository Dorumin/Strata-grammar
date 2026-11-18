#pragma once

// Adapted from llama.cpp's src/llama-grammar.h at pin 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT, see
// third_party/strata-llama/LICENSE).  Changes: the llama.h/llama_vocab dependency is replaced by a small
// GrammarVocab interface, the sampling glue (llama_grammar_apply_impl) is replaced by
// llama_grammar_fill_allowed, and everything lives in strata::grammar.  The grammar algorithm itself is unchanged.
#include "llama_compat.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::grammar {

using llama_token = int32_t;

struct GrammarVocab;

// grammar element type
enum llama_gretype {
    // end of rule definition
    LLAMA_GRETYPE_END            = 0,

    // start of alternate definition for rule
    LLAMA_GRETYPE_ALT            = 1,

    // non-terminal element: reference to rule
    LLAMA_GRETYPE_RULE_REF       = 2,

    // terminal element: character (code point)
    LLAMA_GRETYPE_CHAR           = 3,

    // inverse char(s) ([^a], [^a-b] [^abc])
    LLAMA_GRETYPE_CHAR_NOT       = 4,

    // modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to
    // be an inclusive range ([a-z])
    LLAMA_GRETYPE_CHAR_RNG_UPPER = 5,

    // modifies a preceding LLAMA_GRETYPE_CHAR or
    // LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
    LLAMA_GRETYPE_CHAR_ALT       = 6,

    // any character (.)
    LLAMA_GRETYPE_CHAR_ANY       = 7,

    // terminal element: token (<[token-id]>)
    LLAMA_GRETYPE_TOKEN          = 8,

    // inverse token (!<[token-id]>)
    LLAMA_GRETYPE_TOKEN_NOT      = 9,
};

typedef struct llama_grammar_element {
    enum llama_gretype type;
    uint32_t           value; // Unicode code point, rule ID, or token ID
} llama_grammar_element;

struct llama_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct llama_grammar_candidate {
    size_t               index;
    const uint32_t     * code_points;
    llama_partial_utf8   partial_utf8;
    llama_token          id;
};

using llama_grammar_rule  = std::vector<      llama_grammar_element>;
using llama_grammar_stack = std::vector<const llama_grammar_element *>;

using llama_grammar_rules      = std::vector<llama_grammar_rule>;
using llama_grammar_stacks     = std::vector<llama_grammar_stack>;
using llama_grammar_candidates = std::vector<llama_grammar_candidate>;

// TODO: remove, needed for tests atm
const llama_grammar_rules  & llama_grammar_get_rules (const struct llama_grammar * grammar);
      llama_grammar_stacks & llama_grammar_get_stacks(      struct llama_grammar * grammar);

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `llama_grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
void llama_grammar_accept(struct llama_grammar * grammar, uint32_t chr);

std::vector<llama_grammar_candidate> llama_grammar_reject_candidates_for_stack(
        const llama_grammar_rules      & rules,
        const llama_grammar_stack      & stack,
        const llama_grammar_candidates & candidates);

struct llama_grammar_parser {
    const GrammarVocab * vocab;
    std::map<std::string, uint32_t> symbol_ids;

    llama_grammar_rules rules;

    llama_grammar_parser(const struct GrammarVocab * vocab = nullptr) : vocab(vocab) {}

    llama_grammar_stack c_rules() const;

    uint32_t get_symbol_id(const char * src, size_t len);
    uint32_t generate_symbol_id(const std::string & base_name);

    void add_rule(uint32_t rule_id, const llama_grammar_rule & rule);

    const char * parse_alternates(
            const char        * src,
            const std::string & rule_name,
            uint32_t            rule_id,
            bool                is_nested);

    const char * parse_sequence(
            const char         * src,
            const std::string  & rule_name,
            llama_grammar_rule & rule,
            bool               is_nested);

    const char * parse_rule(const char * src);

    bool parse(const char * src);
    void print(FILE * file);
};

struct llama_grammar_trigger_pattern {
    std::string pattern;
    std::regex  regex;

    size_t find(const std::string & input) const;
};

struct llama_grammar {
    // maintain a list of llama_tokens and their positions in the trigger_buffer
    using token_pos = std::pair<llama_token, std::pair<size_t, size_t>>;

    // note: allow null vocab for testing (not great)
    const GrammarVocab * vocab;

    const llama_grammar_rules  rules;  // TODO: shared ptr
          llama_grammar_stacks stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    llama_partial_utf8 partial_utf8;

    // lazy grammars wait for trigger words or tokens before constraining the sampling.
    // we still have trigger_tokens for non-lazy grammars to force printing of special trigger tokens.
    // (useful e.g. for tool_choice=required)
    bool                     lazy             = false;
    bool                     awaiting_trigger = false; // Initialized to true for lazy grammars only
    std::string              trigger_buffer;           // Output buffered by lazy grammar. Will be cleared once trigger is found.
    std::vector<token_pos>   trigger_buffer_positions; // Tokens buffered by lazy grammar. Used to replay when a trigger is found.
    std::vector<llama_token> trigger_tokens;           // Tokens that trigger a lazy grammar, or tokens to force printing of (even if special).
    std::vector<llama_grammar_trigger_pattern>
                             trigger_patterns;         // Regular expressions that trigger a lazy grammar. Must be a full match of the entire generated
                                                       // string, and the grammar will be given the string from the first match group onwards.

};

//
// internal API
//

// note: needed for tests (not great)
struct llama_grammar * llama_grammar_init_impl(
        const struct GrammarVocab * vocab,
        const llama_grammar_element ** rules,
        size_t n_rules,
        size_t start_rule_index);

struct llama_grammar * llama_grammar_init_impl(
        const struct GrammarVocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens);

void llama_grammar_free_impl(struct llama_grammar * grammar);

struct llama_grammar * llama_grammar_clone_impl(const struct llama_grammar & grammar);

void llama_grammar_accept_impl(
              struct llama_grammar & grammar,
                       llama_token   token);

void llama_grammar_accept_str(
              struct llama_grammar & grammar,
                 const std::string & piece);

void llama_grammar_accept_token(
              struct llama_grammar & grammar,
                       llama_token   token,
                 const std::string & piece);

// The sampling seam, replacing upstream's llama_grammar_apply_impl.  Writes a 1/0 table sized to the vocabulary:
// 1 = the grammar accepts that token in the current state, 0 = masked.  EOG is allowed only once a stack is empty,
// special tokens and tokens with no piece are never allowed.  Returns false, with `err` set, when nothing at all is
// allowed -- the grammar is stuck and the request must fail rather than sample an undefined row.
//
// The vocabulary, decoded and indexed once: every token's code points, the ids a mask may ever allow (not EOG, not a
// special token, not an empty piece), and the EOG ids.  Built by llama_grammar_decode_pieces, shared by every Grammar
// over one vocabulary.  Iterating `ids` instead of the whole vocabulary keeps the mask off the piece strings, whose
// random access was most of its cost.
struct llama_grammar_decoded {
    // Code points in ONE buffer, with an offset per token id, 0-terminated.  A vector per token put every scan on
    // 248k separate heap blocks (random access, about 25 ms a mask); this is two sequential arrays.
    std::vector<uint32_t>           flat;
    std::vector<uint32_t>            off;       // n_vocab + 1 entries: token id -> flat index
    std::vector<llama_partial_utf8> partial;   // the UTF-8 sequence a piece ends in, if any (a byte token)
    std::vector<int32_t>               ids;     // the candidate tokens
    std::vector<int32_t>            eog_ids;
    // Candidate tokens grouped by their first code point, as a compressed sparse row: bucket b owns
    // by_first_ids[by_first[b] .. by_first[b+1]).  Bucket 0x110000 holds the tokens that begin with no complete code
    // point (byte tokens).  A mask then tests each distinct first code point once and visits only the tokens that can
    // start on an accepted one, instead of walking 248k tokens per state.
    std::vector<uint32_t>        by_first;      // 0x110002 entries: prefix sums, one bucket per code point plus one
    std::vector<int32_t>     by_first_ids;
    std::vector<uint32_t>       first_cps;      // the distinct first code points present, ascending

    const uint32_t * cps(int32_t id) const { return flat.data() + off[static_cast<size_t>(id)]; }
};

bool llama_grammar_fill_allowed(
        const struct llama_grammar & grammar,
              const llama_grammar_decoded * decoded,
                   std::vector<uint8_t> & allowed,
                           std::string & err);

// Decodes every token's piece into `out`.  Do this once per vocabulary, not once per token.
void llama_grammar_decode_pieces(
        const GrammarVocab & vocab,
        llama_grammar_decoded & out);

}  // namespace strata::grammar

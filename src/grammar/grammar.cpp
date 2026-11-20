#include "grammar.h"

#include "llama_grammar.h"

#include <deque>
#include <exception>
#include <mutex>
#include <unordered_map>

namespace strata::grammar {
namespace {

// The upstream parser recurses through nested groups and alternates, so a 64 KiB string of '(' would overflow the
// stack before any limit below could be checked.  This scans the text the way the parser reads it closely enough to
// measure depth: string literals and character classes are skipped, a backslash escapes the next byte in both.
bool check_nesting(const std::string& gbnf, size_t limit, std::string& err) {
    size_t depth = 0;
    bool in_string = false;
    bool in_class = false;
    for (size_t i = 0; i < gbnf.size(); ++i) {
        const char c = gbnf[i];
        if (c == '\\' && (in_string || in_class)) {
            ++i;  // the escaped byte cannot open or close anything
            continue;
        }
        if (in_string) {
            if (c == '"') in_string = false;
            continue;
        }
        if (in_class) {
            if (c == ']') in_class = false;
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '[') {
            in_class = true;
        } else if (c == '(') {
            if (++depth > limit) {
                err = "grammar is nested deeper than " + std::to_string(limit) + " levels";
                return false;
            }
        } else if (c == ')' && depth > 0) {
            --depth;
        }
    }
    return true;
}

}  // namespace

struct Grammar::Impl {
    const GrammarVocab* vocab = nullptr;
    llama_grammar* g = nullptr;
    // Every token's code points, decoded once and shared by every Grammar over the same vocabulary: the decode is
    // ~30 ms for 248k pieces, and paying it once per request would be a visible part of a short answer.
    std::shared_ptr<const llama_grammar_decoded> decoded;

    // Allowed sets by grammar state.  A mask is a 248k-entry scan, and a JSON grammar returns to the same state at
    // every comma and colon: a small cache turns most steps into a memset and a list of ids.  Keyed by the exact
    // state (the stacks' element pointers and the partial UTF-8), so a hit can never be a different state.
    struct Cached {
        std::vector<int32_t> ids;      // the allowed token ids, sorted: far smaller than the full table
        int32_t n_vocab = 0;
    };
    std::unordered_map<std::string, Cached> cache;
    std::deque<std::string> cache_order;
    static constexpr size_t kCacheMax = 32;

    void remember(const std::string& key, const std::vector<uint8_t>& allowed) {
        Cached c;
        c.n_vocab = static_cast<int32_t>(allowed.size());
        for (size_t i = 0; i < allowed.size(); ++i)
            if (allowed[i]) c.ids.push_back(static_cast<int32_t>(i));
        if (cache.size() >= kCacheMax) {
            cache.erase(cache_order.front());
            cache_order.pop_front();
        }
        cache.emplace(key, std::move(c));
        cache_order.push_back(key);
    }
};

// One decoded vocabulary per process: the engine has one model, and every Grammar over it shares the table.
static std::shared_ptr<const llama_grammar_decoded> shared_decode_table(const GrammarVocab& vocab) {
    static std::mutex mu;
    static const GrammarVocab* owner = nullptr;
    static std::weak_ptr<const llama_grammar_decoded> table;
    std::lock_guard<std::mutex> lock(mu);
    std::shared_ptr<const llama_grammar_decoded> live = table.lock();
    if (live && owner == &vocab) return live;
    auto made = std::make_shared<llama_grammar_decoded>();
    llama_grammar_decode_pieces(vocab, *made);
    owner = &vocab;
    table = made;
    return made;
}

// The grammar state a mask depends on: every stack's element pointers (stable for the life of the grammar, they
// point into its rules) and the carried-over partial UTF-8.
static std::string state_key(const llama_grammar& g) {
    std::string key;
    // The lazy state is part of the state: a grammar waiting for its trigger and a grammar just triggered over the
    // same token have the same stacks (the trigger is not consumed), but the first allows everything and the second
    // must mask from the root's start.
    key.push_back(g.awaiting_trigger ? '\x02' : '\x01');
    for (const auto& stack : g.stacks) {
        key.reserve(key.size() + stack.size() * sizeof(const llama_grammar_element*) + 8);
        for (const auto* e : stack) {
            const uint64_t v = reinterpret_cast<uint64_t>(e);
            key.append(reinterpret_cast<const char*>(&v), sizeof v);
        }
        key.push_back('\x01');
    }
    // `value` only means anything while a UTF-8 sequence is in progress: with n_remain == 0 it holds the last
    // character decoded, which differs for every token even though the grammar state is the same (this alone turned
    // every character inside a JSON string into a cache miss).
    key.push_back(static_cast<char>(g.partial_utf8.n_remain == 0 ? 0 : 1));
    if (g.partial_utf8.n_remain != 0) {
        const unsigned char tail[8] = {static_cast<unsigned char>(g.partial_utf8.value),
                                       static_cast<unsigned char>(g.partial_utf8.value >> 8),
                                       static_cast<unsigned char>(g.partial_utf8.value >> 16),
                                       static_cast<unsigned char>(g.partial_utf8.value >> 24),
                                       static_cast<unsigned char>(g.partial_utf8.n_remain & 0xFF),
                                       static_cast<unsigned char>((g.partial_utf8.n_remain >> 8) & 0xFF),
                                       static_cast<unsigned char>((g.partial_utf8.n_remain >> 16) & 0xFF),
                                       static_cast<unsigned char>((g.partial_utf8.n_remain >> 24) & 0xFF)};
        key.append(reinterpret_cast<const char*>(tail), sizeof tail);
    }
    return key;
}

Grammar::Grammar(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Grammar::~Grammar() {
    if (impl_ && impl_->g) llama_grammar_free_impl(impl_->g);
}

std::unique_ptr<Grammar> Grammar::parse(const std::string& gbnf, const std::string& root, const GrammarVocab& vocab,
                                        std::string& err, const std::vector<int32_t>& trigger_tokens) {
    err.clear();
    if (gbnf.empty()) {
        err = "grammar is empty";
        return nullptr;
    }
    if (gbnf.size() > kMaxGrammarBytes) {
        err = "grammar is " + std::to_string(gbnf.size()) + " bytes, the limit is " + std::to_string(kMaxGrammarBytes);
        return nullptr;
    }
    if (root.empty()) {
        err = "grammar root rule is empty";
        return nullptr;
    }
    if (!check_nesting(gbnf, kMaxNesting, err)) return nullptr;

    // init_impl logs its own reason (parse error, missing root, left recursion); the request gets one clear message.
    const std::vector<llama_token> triggers(trigger_tokens.begin(), trigger_tokens.end());
    llama_grammar* g = llama_grammar_init_impl(&vocab, gbnf.c_str(), root.c_str(),
                                               /*lazy=*/!triggers.empty(),
                                               /*trigger_patterns=*/nullptr, /*num_trigger_patterns=*/0,
                                               /*trigger_tokens=*/triggers.empty() ? nullptr : triggers.data(),
                                               /*num_trigger_tokens=*/triggers.size());
    if (g == nullptr) {
        err = "grammar does not parse, has no '" + root + "' rule, or is left-recursive";
        return nullptr;
    }

    size_t elements = 0;
    for (const auto& rule : g->rules) elements += rule.size();
    if (elements > kMaxRuleElements) {
        llama_grammar_free_impl(g);
        err = "grammar expands to " + std::to_string(elements) + " elements, the limit is " +
              std::to_string(kMaxRuleElements);
        return nullptr;
    }

    auto impl = std::make_unique<Impl>();
    impl->vocab = &vocab;
    impl->g = g;
    return std::unique_ptr<Grammar>(new Grammar(std::move(impl)));
}

bool Grammar::fill_allowed(std::vector<uint8_t>& allowed, std::string& err) {
    err.clear();
    llama_grammar* g = impl_->g;
    if (g->stacks.empty()) {
        err = "the grammar has no viable parse position";
        return false;
    }
    const size_t work = g->stacks.size() * static_cast<size_t>(impl_->vocab->n_vocab);
    if (work > kMaskWorkBudget) {
        err = "the grammar needs " + std::to_string(work) +
              " token checks for one step (the limit is " + std::to_string(kMaskWorkBudget) + ")";
        return false;
    }
    const std::string key = state_key(*g);
    const auto hit = impl_->cache.find(key);
    if (hit != impl_->cache.end() && hit->second.n_vocab == impl_->vocab->n_vocab) {
        allowed.assign(static_cast<size_t>(hit->second.n_vocab), 0);
        for (const int32_t id : hit->second.ids) allowed[static_cast<size_t>(id)] = 1;
        return true;
    }
    if (!impl_->decoded) impl_->decoded = shared_decode_table(*impl_->vocab);
    if (!llama_grammar_fill_allowed(*g, impl_->decoded.get(), allowed, err)) return false;
    impl_->remember(key, allowed);
    return true;
}

bool Grammar::accept(int32_t token, std::string& err) {
    err.clear();
    try {
        llama_grammar_accept_impl(*impl_->g, token);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    if (impl_->g->stacks.size() > kMaxStacks) {
        err = "the grammar grew past " + std::to_string(kMaxStacks) + " parse positions";
        return false;
    }
    for (const auto& stack : impl_->g->stacks) {
        if (stack.size() > kMaxStackDepth) {
            err = "the grammar grew past " + std::to_string(kMaxStackDepth) + " levels of nesting at run time";
            return false;
        }
    }
    return true;
}

bool Grammar::complete() const {
    for (const auto& stack : impl_->g->stacks) {
        if (stack.empty()) return true;
    }
    return false;
}

}  // namespace strata::grammar

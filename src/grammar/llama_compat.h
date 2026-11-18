#pragma once

// Minimal stand-ins for the llama.cpp headers the extracted grammar code used to include.  The grammar needs no
// llama model, context, ggml or sampler machinery; these macros are the whole surface it touched.
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define GGML_ASSERT(cond)                                                                       \
    do {                                                                                        \
        if (!(cond)) {                                                                          \
            std::fprintf(stderr, "strata grammar: assertion failed: %s (%s:%d)\n", #cond,       \
                         __FILE__, __LINE__);                                                   \
            std::abort();                                                                        \
        }                                                                                       \
    } while (0)

#define GGML_ABORT(msg)                                                                         \
    do {                                                                                        \
        std::fprintf(stderr, "strata grammar: %s (%s:%d)\n", (msg), __FILE__, __LINE__);         \
        std::abort();                                                                            \
    } while (0)

#define LLAMA_LOG_ERROR(...) do { std::fprintf(stderr, __VA_ARGS__); } while (0)
#define LLAMA_LOG_DEBUG(...) do { } while (0)

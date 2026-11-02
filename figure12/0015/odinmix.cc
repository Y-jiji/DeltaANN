#include "pipemix.hpp"

#include <cstddef>

/** @brief Threads a run takes per core unless knob threads says otherwise: with PipeANN's polls
 *  waiting once idle, two a core overlap one's compute with the other's reads (GCP z3, 100K mix:
 *  60 s at two, 92 s at one spinning, 3:34 total at four against 3:04 at two) */
constexpr std::size_t THREADX = 2;

/**
 * @brief OdinANN's mix driver, pipemix.hpp's run over OdinANN's insert and merge
 * @param argc argument count
 * @param argv dataset, data, graph, queries, quant, index, log, then key=value knobs
 * @return 0 on success
 */
int main(int argc, char** argv);

int main(int argc, char** argv) {
    // [1]
    return dispatch(argc, argv, false, THREADX);
}

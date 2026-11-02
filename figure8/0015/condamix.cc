#include "pipemix.hpp"

#include <cstddef>

/** @brief Threads a run takes per core unless knob threads says otherwise, two as OdinANN's: 16 threads
 *  on the z3's 8 cores, CONDA's best of the bgsweep runs */
constexpr std::size_t THREADX = 2;

/**
 * @brief CONDA's mix driver, pipemix.hpp's run over CONDA's insert and delete repair
 * @param argc argument count
 * @param argv dataset, data, graph, queries, quant, index, log, then key=value knobs
 * @return 0 on success
 */
int main(int argc, char** argv);

int main(int argc, char** argv) {
    // [1]
    return dispatch(argc, argv, true, THREADX);
}

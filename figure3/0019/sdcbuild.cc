#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "parlay/parallel.h"
#include "utils/euclidian_point.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/stats.h"
#include "utils/types.h"

#include "pqcode.hpp"
#include "sdcindex.h"

/** @brief Degree bound of the graph, 32 for the 10M slices */
constexpr long BUILD_R = 32;
/** @brief Beam width of the build's insert search */
constexpr long BUILD_L = 128;
/** @brief Slack of the last pass, ParlayANN's alpha on squared distances */
constexpr double BUILD_ALPHA = 1.2;
/** @brief Insert passes, alpha 1 on every pass before the last */
constexpr long BUILD_PASSES = 2;

/**
 * @brief Build a ParlayANN graph over the base, the insert search by ADC to the codebook's codes,
 * every prune distance SDC of the codes or exact
 * @tparam POINT ParlayANN point type of the base
 * @param base base file, ParlayANN bin format
 * @param book raw f32 codebook from `pqtrain`, BAND 8 and BITS 8
 * @param graph ParlayANN graph file written
 * @param exact prune by exact distance instead of SDC
 */
template <typename POINT>
static void build(char* base, char* book, char* graph, bool exact);

/**
 * @brief sdcbuild <int8|float> <base> <codebook> <graph> <sdc|exact>
 * @param argc argument count, 6
 * @param argv element type, the three file paths, then the prune distance
 * @return 0 on success
 */
int main(int argc, char** argv);

template <typename POINT>
static void build(char* base, char* book, char* graph, bool exact) {
    using T = std::remove_cvref_t<decltype(std::declval<POINT>()[0])>;
    // [1]
    parlayANN::PointRange<POINT> points(base);
    PQCode quant(book, points.dimension());
    std::vector<uint8_t> codes(points.size() * quant.nband);
    parlay::parallel_for(0, points.size(), [&] (long i) {
        quant.encode((const T*) points.location(i), codes.data() + i * quant.nband);
    });
    std::cout << "sdcbuild encoded points=" << points.size() << " bands=" << quant.nband
              << std::endl;
    // [2]
    parlayANN::BuildParams params(BUILD_R, BUILD_L, BUILD_ALPHA, BUILD_PASSES, 0, 0, 0, 0,
                                  false, 0, false, 0, 0, 0.0, 10, .125);
    parlayANN::Graph<unsigned int> edges(params.max_degree(), points.size());
    parlayANN::stats<unsigned int> counts(edges.size());
    parlayANN::SdcIndex<parlayANN::PointRange<POINT>, parlayANN::PointRange<POINT>, unsigned int>
        index(params, quant, codes.data(), exact);
    index.build_index(edges, points, points, counts);
    // [3]
    edges.save(graph);
    std::cout << "sdcbuild saved " << graph << std::endl;
}

int main(int argc, char** argv) {
    // [1]
    if (argc != 6 || (std::strcmp(argv[5], "sdc") != 0 && std::strcmp(argv[5], "exact") != 0)) {
        std::cerr << "sdcbuild <int8|float> <base> <codebook> <graph> <sdc|exact>" << std::endl;
        return 1;
    }
    // [2]
    bool exact = std::strcmp(argv[5], "exact") == 0;
    if (std::strcmp(argv[1], "int8") == 0) {
        build<parlayANN::Euclidian_Point<int8_t>>(argv[2], argv[3], argv[4], exact);
    } else if (std::strcmp(argv[1], "float") == 0) {
        build<parlayANN::Euclidian_Point<float>>(argv[2], argv[3], argv[4], exact);
    } else {
        std::cerr << "unknown element type " << argv[1] << std::endl;
        return 1;
    }
    return 0;
}

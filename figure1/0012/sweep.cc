#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "utils/check_nn_recall.h"
#include "utils/euclidian_point.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/types.h"

/** @brief Neighbors each query returns and is graded on */
constexpr long TOPK = 10;
/** @brief Searches per beam width; the analysis keeps the median */
constexpr long REPEAT = 3;
/** @brief ParlayANN batch factor; zero merges candidates after every expansion */
constexpr double BATCH = 0.0;
/** @brief Beam widths swept, each with no visit limit */
constexpr long BEAMS[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32,
    34, 36, 38, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 120, 140, 160,
    180, 200, 225, 250, 275, 300, 375, 500, 750, 1000};

/**
 * @brief Search one graph at every beam width, printing one line per search
 * @tparam POINT ParlayANN point type of the base and query files
 * @param label dataset and graph joined by a dot, e.g. sift.orig
 * @param base base file, ParlayANN bin format
 * @param query query file, ParlayANN bin format
 * @param truth ground truth file with TOPK ids per query
 * @param graph ParlayANN graph file
 */
template <typename POINT>
static void sweep(char* label, char* base, char* query, char* truth, char* graph);

/**
 * @brief sweep <label> <uint8|int8|float> <base> <query> <truth> <graph>
 * @param argc argument count, 7
 * @param argv label, element type, then the four file paths
 * @return 0 on success
 */
int main(int argc, char** argv);

template <typename POINT>
static void sweep(char* label, char* base, char* query, char* truth, char* graph) {
    // [1]
    std::string name(label);
    auto dot = name.find('.');
    if (dot == std::string::npos) {
        std::cerr << "label must be <ds>.<graph>: " << name << std::endl;
        std::abort();
    }
    std::cout << "ds=" << name.substr(0, dot) << " graph=" << name.substr(dot + 1) << std::endl;
    // [2]
    parlayANN::PointRange<POINT> points(base);
    parlayANN::PointRange<POINT> queries(query);
    parlayANN::groundTruth<unsigned int> answer(truth);
    parlayANN::Graph<unsigned int> edges(graph);
    // [3]
    bool fits = answer.size() == queries.size() && answer.dimension() == TOPK;
    if (!fits || edges.size() != points.size()) {
        std::cerr << "mismatch: truth " << answer.size() << "x" << answer.dimension()
                  << ", queries " << queries.size() << ", graph " << edges.size()
                  << ", points " << points.size() << std::endl;
        std::abort();
    }
    // [4]
    for (long beam : BEAMS) {
        parlayANN::QueryParams params(TOPK, beam, (long) points.size(), edges.max_degree());
        params.batch_factor = BATCH;
        for (long run = 0; run < REPEAT; run++) {
            parlayANN::checkRecall(edges, points, queries, points, queries, points, queries,
                                   answer, false, 0, TOPK, params, true);
        }
    }
}

int main(int argc, char** argv) {
    // [1]
    if (argc != 7) {
        std::cerr << "sweep <label> <uint8|int8|float> <base> <query> <truth> <graph>" << std::endl;
        return 1;
    }
    // [2]
    if (std::strcmp(argv[2], "uint8") == 0) {
        sweep<parlayANN::Euclidian_Point<uint8_t>>(argv[1], argv[3], argv[4], argv[5], argv[6]);
    } else if (std::strcmp(argv[2], "int8") == 0) {
        sweep<parlayANN::Euclidian_Point<int8_t>>(argv[1], argv[3], argv[4], argv[5], argv[6]);
    } else if (std::strcmp(argv[2], "float") == 0) {
        sweep<parlayANN::Euclidian_Point<float>>(argv[1], argv[3], argv[4], argv[5], argv[6]);
    } else {
        std::cerr << "unknown element type " << argv[2] << std::endl;
        return 1;
    }
    return 0;
}

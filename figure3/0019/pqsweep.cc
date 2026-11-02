#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "parlay/parallel.h"
#include "utils/graph.h"

#include "pqcode.hpp"

/** @brief Neighbors each query returns and is graded on */
constexpr long TOPK = 10;
/** @brief Searches per beam width; the analysis keeps the median QPS */
constexpr long REPEAT = 3;
/** @brief Beam widths swept by default */
constexpr long BEAMS[] = {10, 16, 24, 32, 50, 70, 100, 140, 200, 300};

/**
 * @brief Beam widths to sweep: env BEAMS as comma-separated widths, else BEAMS
 * @return the beam widths in sweep order
 */
static std::vector<long> beams();

/**
 * @brief Read a ParlayANN bin file, `u32 rows, u32 dims` then the rows, contiguous
 * @tparam T element type
 * @param path file path
 * @param dims dimension read from the header
 * @return rows times dims elements
 */
template <typename T> static std::vector<T> load(const char* path, long& dims);

/**
 * @brief One query's beam search from start node 0, by ADC over the codes or by exact distance,
 * then every expanded node reranked by exact squared L2 to its TOPK nearest
 * @tparam T element type of base and query
 * @param graph ParlayANN graph
 * @param quant codebook the codes were taken with
 * @param codes `nband` bytes per base point
 * @param base contiguous base rows
 * @param query one query row
 * @param dims dimension
 * @param beam beam width
 * @param exact traverse by exact distance instead of ADC
 * @return the TOPK ids, nearest first
 * SHAME(WIDEFUNC) SHAME(MANYARG)
 */
template <typename T>
static std::vector<uint32_t> search(const parlayANN::Graph<uint32_t>& graph, const PQCode& quant, const uint8_t* codes,
                                    const T* base, const T* query, long dims, long beam, bool exact);

/**
 * @brief Search every query at every beam width REPEAT times, printing one line per search
 * @tparam T element type of base and query
 * @param label dataset and graph joined by a dot, e.g. spacev.AB
 * @param base base file, ParlayANN bin format
 * @param query query file, ParlayANN bin format
 * @param truth ground truth file, ParlayANN format from gtmerge
 * @param book raw f32 codebook from `pqtrain`, BAND 8 and BITS 8
 * @param graph ParlayANN graph file
 * @param exact traverse by exact distance instead of ADC
 * SHAME(MANYARG)
 */
template <typename T>
static void sweep(char* label, char* base, char* query, char* truth, char* book, char* graph,
                  bool exact);

/**
 * @brief pqsweep <label> <int8|float> <pq|exact> <base> <query> <truth> <codebook> <graph>
 * @param argc argument count, 9
 * @param argv label, element type, traversal mode, then the five file paths
 * @return 0 on success
 */
int main(int argc, char** argv);

static std::vector<long> beams() {
    // [1]
    const char* text = std::getenv("BEAMS");
    if (text == nullptr) return std::vector<long>(std::begin(BEAMS), std::end(BEAMS));
    // [2]
    std::vector<long> widths;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) widths.push_back(std::stol(item));
    return widths;
}

template <typename T> static std::vector<T> load(const char* path, long& dims) {
    // [1]
    FILE* file = std::fopen(path, "rb");
    uint32_t head[2];
    if (file == nullptr || std::fread(head, sizeof(uint32_t), 2, file) != 2) {
        std::fprintf(stderr, "%s: cannot read header\n", path);
        std::abort();
    }
    // [2]
    dims = head[1];
    std::vector<T> data((size_t) head[0] * head[1]);
    if (std::fread(data.data(), sizeof(T), data.size(), file) != data.size()) {
        std::fprintf(stderr, "%s: short read\n", path);
        std::abort();
    }
    std::fclose(file);
    return data;
}

template <typename T>
static std::vector<uint32_t> search(const parlayANN::Graph<uint32_t>& graph, const PQCode& quant, const uint8_t* codes,
                                    const T* base, const T* query, long dims, long beam, bool exact) {
    // [1]
    std::vector<float> cast(dims);
    for (long d = 0; d < dims; d++) cast[d] = (float) query[d];
    std::vector<float> lut(quant.nband * PQ_KS);
    quant.lookup(cast.data(), lut.data());
    auto full = [&] (uint32_t id) {
        float sum = 0;
        for (long d = 0; d < dims; d++) {
            float diff = cast[d] - (float) base[(size_t) id * dims + d];
            sum += diff * diff;
        }
        return sum;
    };
    auto measure = [&] (uint32_t id) {
        return exact ? full(id) : quant.adc(lut.data(), codes + (size_t) id * quant.nband);
    };
    // [2]
    std::vector<std::pair<float, uint32_t>> pool = {{measure(0), 0}};
    std::vector<bool> done = {false};
    std::unordered_set<uint32_t> seen = {0};
    std::vector<uint32_t> expanded;
    while (true) {
        long at = 0;
        while (at < (long) pool.size() && done[at]) at++;
        if (at == (long) pool.size()) break;
        done[at] = true;
        expanded.push_back(pool[at].second);
        auto edges = graph[pool[at].second];
        for (size_t j = 0; j < edges.size(); j++) {
            uint32_t next = edges[j];
            if (!seen.insert(next).second) continue;
            float dist = measure(next);
            if ((long) pool.size() == beam && dist >= pool.back().first) continue;
            auto spot = std::lower_bound(pool.begin(), pool.end(), std::make_pair(dist, next));
            long pos = spot - pool.begin();
            pool.insert(spot, {dist, next});
            done.insert(done.begin() + pos, false);
            if ((long) pool.size() > beam) {
                pool.pop_back();
                done.pop_back();
            }
        }
    }
    // [3]
    std::vector<std::pair<float, uint32_t>> ranked;
    for (uint32_t id : expanded) ranked.push_back({full(id), id});
    std::sort(ranked.begin(), ranked.end());
    std::vector<uint32_t> top;
    for (long k = 0; k < TOPK && k < (long) ranked.size(); k++) top.push_back(ranked[k].second);
    return top;
}

template <typename T>
static void sweep(char* label, char* base, char* query, char* truth, char* book, char* graph,
                  bool exact) {
    // [1]
    std::string name(label);
    auto dot = name.find('.');
    std::cout << "ds=" << name.substr(0, dot) << " graph=" << name.substr(dot + 1)
              << " mode=" << (exact ? "exact" : "pq") << " book=" << book << std::endl;
    long dims = 0, qdims = 0, depth = 0;
    std::vector<T> points = load<T>(base, dims);
    std::vector<T> queries = load<T>(query, qdims);
    std::vector<uint32_t> answer = load<uint32_t>(truth, depth);
    parlayANN::Graph<uint32_t> edges(graph);
    long n = points.size() / dims, q = queries.size() / qdims;
    if (qdims != dims || depth < TOPK || (long) edges.size() != n) {
        std::cerr << "mismatch: dims " << dims << " query dims " << qdims << " truth depth "
                  << depth << " graph " << edges.size() << " points " << n << std::endl;
        std::abort();
    }
    // [2]
    PQCode quant(book, dims);
    std::vector<uint8_t> codes(n * quant.nband);
    parlay::parallel_for(0, n, [&] (long i) {
        quant.encode(points.data() + (size_t) i * dims, codes.data() + (size_t) i * quant.nband);
    });
    // [3]
    for (long beam : beams()) {
        for (long run = 0; run < REPEAT; run++) {
            std::vector<std::vector<uint32_t>> found(q);
            auto start = std::chrono::steady_clock::now();
            parlay::parallel_for(0, q, [&] (long i) {
                const T* row = queries.data() + (size_t) i * dims;
                found[i] = search(edges, quant, codes.data(), points.data(), row, dims, beam,
                                  exact);
            }, 1);
            auto stop = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(stop - start).count();
            long hits = 0;
            for (long i = 0; i < q; i++) {
                for (uint32_t id : found[i]) {
                    const uint32_t* row = answer.data() + (size_t) i * depth;
                    hits += std::find(row, row + TOPK, id) != row + TOPK;
                }
            }
            std::cout << "search: beam=" << beam << " recall=" << (double) hits / (q * TOPK)
                      << " qps=" << q / secs << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    // [1]
    if (argc != 9 || (std::strcmp(argv[3], "pq") != 0 && std::strcmp(argv[3], "exact") != 0)) {
        std::cerr << "pqsweep <label> <int8|float> <pq|exact> <base> <query> <truth> <codebook> "
                  << "<graph>" << std::endl;
        return 1;
    }
    // [2]
    bool exact = std::strcmp(argv[3], "exact") == 0;
    if (std::strcmp(argv[2], "int8") == 0) {
        sweep<int8_t>(argv[1], argv[4], argv[5], argv[6], argv[7], argv[8], exact);
    } else if (std::strcmp(argv[2], "float") == 0) {
        sweep<float>(argv[1], argv[4], argv[5], argv[6], argv[7], argv[8], exact);
    } else {
        std::cerr << "unknown element type " << argv[2] << std::endl;
        return 1;
    }
    return 0;
}

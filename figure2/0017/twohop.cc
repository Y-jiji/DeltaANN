#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "utils/beamSearch.h"
#include "utils/euclidian_point.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/stats.h"
#include "utils/types.h"
#include "vamana/index.h"

/** @brief Beam width of each search, ParlayANN's build L; its expanded set is pruned */
constexpr long BEAM = 128;
/** @brief Most nodes one search may expand, the tally's rank capacity */
constexpr long VISIT = 1024;
/** @brief SaturationPrune's alpha on L2, the graphs' build alpha */
constexpr double ALPHA = 1.2;

/** @brief A node's fate in one point's prune: kept, pruned, or outside the beam */
enum Fate { KEPT, DROP, AWAY };

/**
 * @brief Where an edge v -> u into the beam has company: another kept beam node ranked before v
 * also links to u, else one ranked after v does, else no other kept beam node links to u
 */
enum Cover { EARLY, LATER, ALONE };

/**
 * @brief Which other beam nodes link to a beam node u: some kept node, else only pruned nodes, else
 * none
 */
enum Reach { KEPT_IN, DROP_IN, NONE_IN };

/**
 * @brief Counts summed over the points searched
 * nodes: [rank * 2 + fate of v], points whose rank-th beam node v has that fate
 * edges: [(rank * 2 + fate of v) * 3 + fate of u], graph edges v -> u out of those nodes
 * cover: [(rank * 2 + fate of v) * 3 + cover], those edges whose u is kept, by their cover
 * reach: [(rank * 2 + fate of u) * 3 + reach], points whose rank-th beam node u has that reach
 */
struct Tally {
    std::vector<long> nodes;
    std::vector<long> edges;
    std::vector<long> cover;
    std::vector<long> reach;
};

/**
 * @brief Cut every out-list longer than degree to its degree nearest, near to far, ties by id
 * @tparam POINT ParlayANN point type of the base
 * @param points the base the graph is built on
 * @param graph ParlayANN graph over the base, trimmed in place
 * @param degree out-neighbors a node keeps
 */
template <typename POINT>
static void trim(parlayANN::PointRange<POINT>& points, parlayANN::Graph<unsigned>& graph,
                 long degree);

/**
 * @brief The paper's SaturationPrune: passes at alpha' = 1, ALPHA, ALPHA^2, ... over the candidates
 * near to far on L2, then a saturating pass, until degree are kept
 * @tparam POINT ParlayANN point type of the base and the points
 * @param points the base the graph is built on
 * @param query the point pruned for
 * @param visit the candidates, node and squared distance to the point, near to far
 * @param degree neighbors kept, R
 * @return the kept nodes
 */
template <typename POINT>
static std::vector<unsigned> saturate(parlayANN::PointRange<POINT>& points, const POINT& query,
                                      const std::vector<std::pair<unsigned, float>>& visit,
                                      long degree);

/**
 * @brief Search the graph for one point, prune its expanded set by SaturationPrune, tally every
 * edge
 * @tparam POINT ParlayANN point type of the base and the points
 * @param tally counts the point adds to
 * @param points the base the graph is built on
 * @param graph ParlayANN graph over the base, trimmed to degree
 * @param query a point outside the base
 * @param degree out-neighbors the search reads per node and the prune keeps at most
 * SHAME(TALLFUNC)
 */
template <typename POINT>
static void count(Tally& tally, parlayANN::PointRange<POINT>& points,
                  parlayANN::Graph<unsigned>& graph, const POINT& query, long degree);

/**
 * @brief Add the counts of one tally into another
 * @param into the tally added to
 * @param from the tally added
 */
static void merge(Tally& into, const Tally& from);

/**
 * @brief Tally the edges of every point's beam over one graph trimmed to degree, the points spread
 * over all workers
 * @tparam POINT ParlayANN point type of the base and the points
 * @param base base file, ParlayANN bin format
 * @param query points outside the base, ParlayANN bin format
 * @param graph ParlayANN graph file over the base
 * @param degree out-neighbors a node keeps after the trim, at most the graph's max degree
 * @return the counts summed over all points
 */
template <typename POINT>
static Tally run(char* base, char* query, char* graph, long degree);

/**
 * @brief twohop <uint8|int8|float> <base> <query> <graph> <degree>, the csv to stdout
 * @param argc argument count, 6
 * @param argv element type, the three file paths, then the degree
 * @return 0 on success
 */
int main(int argc, char** argv);

template <typename POINT>
static void trim(parlayANN::PointRange<POINT>& points, parlayANN::Graph<unsigned>& graph,
                 long degree) {
    // [1]
    parlay::parallel_for(0, graph.size(), [&](size_t node) {
        auto out = graph[node];
        if ((long) out.size() <= degree) {
            return;
        }
        // [2]
        std::vector<std::pair<float, unsigned>> ranked;
        for (size_t i = 0; i < out.size(); i++) {
            ranked.push_back({points[node].distance(points[out[i]]), out[i]});
        }
        std::sort(ranked.begin(), ranked.end());
        // [3]
        parlay::sequence<unsigned> keep(degree);
        for (long i = 0; i < degree; i++) {
            keep[i] = ranked[i].second;
        }
        graph[node].update_neighbors(keep);
    });
}

template <typename POINT>
static std::vector<unsigned> saturate(parlayANN::PointRange<POINT>& points, const POINT& query,
                                      const std::vector<std::pair<unsigned, float>>& visit,
                                      long degree) {
    // [1]
    constexpr double NEVER = std::numeric_limits<double>::infinity();
    long size = (long) visit.size();
    std::vector<double> far(size), need(size, 0.0);
    std::vector<char> taken(size, 0);
    for (long i = 0; i < size; i++) {
        far[i] = std::sqrt((double) query.distance(points[visit[i].first]));
    }
    std::vector<unsigned> kept;
    auto take = [&](long i) {
        taken[i] = 1;
        kept.push_back(visit[i].first);
        auto node = points[visit[i].first];
        for (long j = 0; j < size; j++) {
            if (taken[j]) {
                continue;
            }
            double near = std::sqrt((double) node.distance(points[visit[j].first]));
            need[j] = std::max(need[j], near > 0 ? far[j] / near : NEVER);
        }
    };
    // [2]
    for (double alpha = 1.0; (long) kept.size() < degree; alpha *= ALPHA) {
        for (long i = 0; i < size && (long) kept.size() < degree; i++) {
            if (!taken[i] && need[i] < alpha) {
                take(i);
            }
        }
        bool open = false;
        for (long i = 0; i < size; i++) {
            open = open || (!taken[i] && need[i] < NEVER);
        }
        if (!open) {
            break;
        }
    }
    // [3]
    for (long i = 0; i < size && (long) kept.size() < degree; i++) {
        if (!taken[i]) {
            take(i);
        }
    }
    return kept;
}

template <typename POINT>
static void count(Tally& tally, parlayANN::PointRange<POINT>& points,
                  parlayANN::Graph<unsigned>& graph, const POINT& query, long degree) {
    // [1]
    parlayANN::QueryParams search(0, BEAM, (long) points.size(), degree);
    parlay::sequence<unsigned> start = {0};
    auto found = parlayANN::beam_search(query, graph, points, start, search);
    std::vector<std::pair<unsigned, float>> beam(found.first.second.begin(),
                                                 found.first.second.end());
    if ((long) beam.size() > VISIT) {
        std::cerr << "expanded " << beam.size() << " nodes, over " << VISIT << std::endl;
        std::abort();
    }
    // [2]
    std::vector<unsigned> kept = saturate(points, query, beam, degree);
    std::sort(kept.begin(), kept.end());
    // [3]
    long size = (long) beam.size();
    std::vector<int> fates(size);
    std::vector<std::pair<unsigned, long>> where;
    for (long rank = 0; rank < size; rank++) {
        bool keep = std::binary_search(kept.begin(), kept.end(), beam[rank].first);
        fates[rank] = keep ? KEPT : DROP;
        where.push_back({beam[rank].first, rank});
    }
    std::sort(where.begin(), where.end());
    auto slot = [&](unsigned id) {
        auto at = std::lower_bound(where.begin(), where.end(), std::pair<unsigned, long>{id, 0});
        return at != where.end() && at->first == id ? at->second : -1L;
    };
    // [4]
    std::vector<long> first(size, size), last(size, -1);
    for (long rank = 0; rank < size; rank++) {
        if (fates[rank] != KEPT) {
            continue;
        }
        auto out = graph[beam[rank].first];
        for (size_t i = 0; i < out.size(); i++) {
            long at = slot(out[i]);
            if (at >= 0) {
                first[at] = std::min(first[at], rank);
                last[at] = std::max(last[at], rank);
            }
        }
    }
    // [5]
    for (long rank = 0; rank < size; rank++) {
        long row = rank * 2 + fates[rank];
        tally.nodes[row]++;
        auto out = graph[beam[rank].first];
        for (size_t i = 0; i < out.size(); i++) {
            long at = slot(out[i]);
            tally.edges[row * 3 + (at < 0 ? AWAY : fates[at])]++;
            if (at >= 0 && fates[at] == KEPT) {
                int cover = first[at] < rank ? EARLY : last[at] > rank ? LATER : ALONE;
                tally.cover[row * 3 + cover]++;
            }
        }
    }
    // [6]
    std::vector<int> reach(size, NONE_IN);
    for (long rank = 0; rank < size; rank++) {
        auto out = graph[beam[rank].first];
        for (size_t i = 0; i < out.size(); i++) {
            long at = slot(out[i]);
            if (at < 0 || at == rank) {
                continue;
            }
            if (fates[rank] == KEPT) {
                reach[at] = KEPT_IN;
            } else if (reach[at] == NONE_IN) {
                reach[at] = DROP_IN;
            }
        }
    }
    for (long rank = 0; rank < size; rank++) {
        tally.reach[(rank * 2 + fates[rank]) * 3 + reach[rank]]++;
    }
}

static void merge(Tally& into, const Tally& from) {
    // [1]
    for (size_t i = 0; i < into.nodes.size(); i++) {
        into.nodes[i] += from.nodes[i];
    }
    // [2]
    for (size_t i = 0; i < into.edges.size(); i++) {
        into.edges[i] += from.edges[i];
    }
    // [3]
    for (size_t i = 0; i < into.cover.size(); i++) {
        into.cover[i] += from.cover[i];
    }
    // [4]
    for (size_t i = 0; i < into.reach.size(); i++) {
        into.reach[i] += from.reach[i];
    }
}

template <typename POINT>
static Tally run(char* base, char* query, char* graph, long degree) {
    // [1]
    auto clock = std::chrono::steady_clock::now();
    auto secs = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - clock).count();
    };
    auto* sink = std::cout.rdbuf(std::cerr.rdbuf());
    parlayANN::PointRange<POINT> points(base);
    parlayANN::PointRange<POINT> queries(query);
    parlayANN::Graph<unsigned> edges(graph);
    std::cout.rdbuf(sink);
    bool fits = degree >= 1 && degree <= edges.max_degree();
    if (!fits || edges.size() != points.size() || queries.dimension() != points.dimension()) {
        std::cerr << "mismatch: degree " << degree << " of " << edges.max_degree() << ", graph "
                  << edges.size() << ", points " << points.size() << "x"
                  << points.dimension() << ", queries " << queries.size() << "x"
                  << queries.dimension() << std::endl;
        std::abort();
    }
    std::cerr << "twohop loaded points=" << points.size() << " queries=" << queries.size()
              << " maxdeg=" << edges.max_degree() << " secs=" << secs() << std::endl;
    trim(points, edges, degree);
    std::cerr << "twohop trimmed degree=" << degree << " secs=" << secs() << std::endl;
    // [2]
    Tally zero{std::vector<long>(VISIT * 2, 0), std::vector<long>(VISIT * 2 * 3, 0),
               std::vector<long>(VISIT * 2 * 3, 0), std::vector<long>(VISIT * 2 * 3, 0)};
    std::vector<Tally> parts(parlay::num_workers(), zero);
    parlay::parallel_for(0, queries.size(), [&](size_t i) {
        count(parts[parlay::worker_id()], points, edges, queries[i], degree);
    }, 1);
    // [3]
    for (auto& part : parts) {
        merge(zero, part);
    }
    std::cerr << "twohop counted workers=" << parts.size() << " secs=" << secs() << std::endl;
    return zero;
}

int main(int argc, char** argv) {
    // [1]
    if (argc != 6) {
        std::cerr << "twohop <uint8|int8|float> <base> <query> <graph> <degree>" << std::endl;
        return 1;
    }
    // [2]
    Tally tally;
    long degree = std::stol(argv[5]);
    if (std::strcmp(argv[1], "uint8") == 0) {
        tally = run<parlayANN::Euclidian_Point<uint8_t>>(argv[2], argv[3], argv[4], degree);
    } else if (std::strcmp(argv[1], "int8") == 0) {
        tally = run<parlayANN::Euclidian_Point<int8_t>>(argv[2], argv[3], argv[4], degree);
    } else if (std::strcmp(argv[1], "float") == 0) {
        tally = run<parlayANN::Euclidian_Point<float>>(argv[2], argv[3], argv[4], degree);
    } else {
        std::cerr << "unknown element type " << argv[1] << std::endl;
        return 1;
    }
    // [3]
    const char* names[] = {"kept", "pruned"};
    std::cout << "rank,status,nodes,kept,pruned,outside,early,later,alone,fromkept,viapruned,nobeam"
              << std::endl;
    long ranks = 0;
    for (long rank = 0; rank < VISIT; rank++) {
        if (tally.nodes[rank * 2 + KEPT] + tally.nodes[rank * 2 + DROP] > 0) {
            ranks = rank + 1;
        }
    }
    for (long rank = 0; rank < ranks; rank++) {
        for (long fate = KEPT; fate <= DROP; fate++) {
            long row = rank * 2 + fate;
            std::cout << rank + 1 << "," << names[fate] << "," << tally.nodes[row] << ","
                      << tally.edges[row * 3 + KEPT] << "," << tally.edges[row * 3 + DROP] << ","
                      << tally.edges[row * 3 + AWAY] << "," << tally.cover[row * 3 + EARLY] << ","
                      << tally.cover[row * 3 + LATER] << "," << tally.cover[row * 3 + ALONE] << ","
                      << tally.reach[row * 3 + KEPT_IN] << ","
                      << tally.reach[row * 3 + DROP_IN] << ","
                      << tally.reach[row * 3 + NONE_IN] << std::endl;
        }
    }
    return 0;
}

#pragma once
// The body odinmix (OdinANN) and condamix (CONDA) share over PipeANN: the same load, workload,
// sweeps and logs, the engine one argument.

#include "fnctbench.hpp"
#include "pipeindex.hpp"

#include "nbr/pq_nbr.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <sys/prctl.h>
#include <utility>
#include <vector>

/** @brief Points replaced per batch, the oldest removed and as many new inserted */
constexpr std::size_t STEP = 1'000'000;
/** @brief Sweeps after the load's, evenly spaced over the batches that replace the base once */
constexpr std::size_t CHECKPOINTS = 10;
/** @brief Out-neighbor slots of a vertex, the parlay graph's R */
constexpr std::size_t DEGREE = 64;
/** @brief Candidate list of the insert search */
constexpr std::uint32_t BUILD_L = 128;
/** @brief Prune alpha of the new node and of each overflowing neighbor */
constexpr float ALPHA = 1.2f;
/** @brief Query search beam width W, the reads a round keeps in flight, upstream's
 *  DynamicIndex::search limit pipe_search widens to */
constexpr std::uint32_t SEARCHW = 32;
/** @brief Empty polls after which a PipeANN poll waits for a completion instead of spinning, set
 *  as PIPEANN_POLL_SPINS unless the environment sets it */
constexpr const char* POLLSPINS = "16";
/** @brief Neighbors a merge keeps per deleted node, PipeANN's default */
constexpr std::uint32_t SAMPLE = 20;
/** @brief CONDA's prune capacity C, its runner's value */
constexpr std::uint32_t CONDA_C = 750;
/** @brief One of every RATE op rows is kept */
constexpr std::size_t RATE = 50;
/** @brief Deletes a retirement waits for past a batch's, so retirements drift into later batches
 *  and overlap their searches */
constexpr std::size_t SKEW = 10'000;
/** @brief Candidate lists a sweep searches with, one QPS/recall point each */
constexpr std::array<std::size_t, 10> BEAMS = {10, 16, 24, 32, 50, 70, 100, 140, 200, 300};
/** @brief Beam widths mode=scan sweeps, inserts and deletes at, all over one load */
constexpr std::array<std::uint32_t, 8> WIDTHS = {2, 4, 6, 8, 12, 16, 24, 32};
/** @brief Beam widths mode=scan scans in place of WIDTHS under wide=1, past their widest */
constexpr std::array<std::uint32_t, 4> WIDE = {24, 32, 48, 64};
/** @brief Inserts, then removes of those rows, of each mode=scan width per pass */
constexpr std::size_t SCANMIX = 20'000;
/** @brief Passes of mode=scan's update stages over WIDTHS, in order then back */
constexpr std::size_t PASSES = 2;
/** @brief Passes a sweep makes at each candidate list */
constexpr std::size_t REPEAT = 3;
/** @brief Neighbors and candidate list of every search of a runmix */
constexpr std::size_t MIX_K = 50;
/** @brief Points per insert-only call of the fill stage */
constexpr std::size_t FILL = 100'000;
/** @brief Knobs both drivers take, the first six shared with freshdel; fill the rows inserted
 *  after the load, before the first sweep; grow the batches that under mode=churn only insert
 *  before the removals start */
constexpr std::array<const char*, 17> KEYS = {"batches", "every", "threads", "step",  "qlen",
                                              "mode",    "budget", "sample",  "wide",  "swidth",
                                              "iwidth",  "dwidth", "reuse",   "beam",  "repeat",
                                              "fill",    "grow"};
/** @brief The search entry point both drivers start from: vertex 0, as every system's */
constexpr std::size_t ENTRY = 0;
/** @brief Counter names of OdinProfile, in column order */
constexpr std::array<const char*, 9> NAMES = {"start", "laten", "hop", "dist", "estim",
                                              "cold",  "pass",  "sdc", "wrote"};

/** @brief One op's counters by NAMES: wall-clock start, latency laten in ns, then the index's */
struct OdinProfile {
    std::array<std::size_t, NAMES.size()> counts{};

    /**
     * @brief Stamp start, set laten at end, or add n to a counter
     * @param name start, end, or a name of NAMES
     * @param n units
     */
    void event(const char* name, std::size_t n);

    /**
     * @brief The counters by name
     * @return (name, count) in NAMES order
     */
    std::vector<std::pair<const char*, std::size_t>> iter() const;
};

/**
 * @brief Write the first base points of the raw corpus data as PipeANN's bin, npts and dim first
 * @tparam T element type of a vector
 * @param data the headerless raw corpus
 * @param path the bin to write
 * @param base points to copy
 * @param dim values per point
 */
template <class T>
static void basebin(const std::string& data, const std::string& path, std::size_t base,
                    std::size_t dim);

/**
 * @brief PQ chunks of a PipeANN pivots file, the code bytes per point
 * @param pivots PipeANN pivots file
 * @return the chunk count its chunk offsets hold
 */
static std::uint32_t chunks(const std::string& pivots);

/**
 * @brief Search every query repeat times at every (list, width) of plan, the search beam width set
 * to the width and to after at the end, tagged {label}_k{list}_r{r}
 * @tparam T element type of a vector
 * @param bench the bench over the index
 * @param index the index whose search beam width the sweep sets
 * @param query the queries
 * @param label tag prefix, sweep_b{batch} or scan_w{width}
 * @param plan (candidate list, beam width) pairs in order
 * @param after the search beam width left set, the runmix searches'
 * @param repeat passes over the queries at each pair
 */
template <class T>
static void sweep(fnctbench::Bench<fnctbench::PipeIndex<T>&, OdinProfile, T>& bench,
                  DynamicIndex<T>& index, std::span<const T> query, const std::string& label,
                  std::span<const std::pair<std::size_t, std::uint32_t>> plan,
                  std::uint32_t after, std::size_t repeat);

/**
 * @brief Delete the values of fresh from a CONDA index as a retirement does, each lazily, then
 * CONDA's delete step and batch reset with nthread threads, its writes left to the background writer
 * @tparam T element type of a vector
 * @param index the CONDA index, no op running on it
 * @param fresh the values to delete
 * @param nthread threads of the delete step
 * @return seconds the whole delete took
 */
template <class T>
static double retire(DynamicIndex<T>& index, const fnctbench::Range& fresh, std::uint32_t nthread);

/**
 * @brief Load the flat graph over the corpus's first rows into OdinANN, or CONDA when conda, run
 * batches runmix calls, batch b tagged mix_{b} removing rows (b - grow) * step to
 * (b - grow + 1) * step, or grow_{b} under mode=grow or before grow, inserting the next step rows
 * and searching as many, the index retiring its deletes every budget of them, sweeping the
 * queries after the load, every every batches and after the last, every search, insert and CONDA
 * delete repair at PipeANN's default beam widths, cores times threadx threads by default, logging
 * as [odin] or [conda]; or under mode=scan, per width
 * of WIDTHS (WIDE under wide=1) a sweep scan_w{width} over BEAMS on the loaded index, then PASSES
 * passes over those widths, in order then back, of a runmix scanins_w{width}_p{pass} of SCANMIX
 * inserts and scandel_w{width}_p{pass} removing them, under CONDA a timed retire() logged to
 * stderr in its place; SHAME(TALLFUNC)
 * @tparam T element type of a vector
 * @param dim values per point
 * @param qlen queries in the query file unless knob qlen says otherwise
 * @param cmd the command line
 * @param conda CONDA's insert and delete repair in place of OdinANN's
 * @param threadx threads per core unless knob threads says otherwise
 * @return 0 on success
 */
template <class T>
static int run(std::size_t dim, std::size_t qlen, const fnctbench::Command& cmd, bool conda,
               std::size_t threadx);

/**
 * @brief Dispatch on the dataset, sift as uint8 with 10K queries, msturing as float with 10K,
 * spacev as int8 with 29316, deep as float 96 with 10K, PIPEANN_POLL_SPINS set to POLLSPINS first
 * unless set, any process of the user allowed to ptrace it under FNCT_PTRACER
 * @param argc argument count
 * @param argv dataset, data, graph, queries, quant, index, log, then key=value knobs
 * @param conda CONDA's insert and delete repair in place of OdinANN's
 * @param threadx threads per core unless knob threads says otherwise
 * @return 0 on success
 */
static int dispatch(int argc, char** argv, bool conda, std::size_t threadx);

inline void OdinProfile::event(const char* name, std::size_t n) {
    // [1]
    const std::string key(name);
    const auto clock = [] {
        return static_cast<std::size_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::system_clock::now().time_since_epoch())
                                            .count());
    };
    if (key == "start") {
        counts[0] = clock();
        return;
    }
    if (key == "end") {
        counts[1] = clock() - counts[0];
        return;
    }
    // [2]
    for (std::size_t at = 2; at < NAMES.size(); at++) {
        if (key == NAMES[at]) {
            counts[at] += n;
        }
    }
}

inline std::vector<std::pair<const char*, std::size_t>> OdinProfile::iter() const {
    // [1]
    std::vector<std::pair<const char*, std::size_t>> out;
    for (std::size_t at = 0; at < NAMES.size(); at++) {
        out.emplace_back(NAMES[at], counts[at]);
    }
    return out;
}

template <class T>
static void basebin(const std::string& data, const std::string& path, std::size_t base,
                    std::size_t dim) {
    // [1]
    std::ifstream input(data, std::ios::binary);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!input || !output) {
        fnctbench::die("copy the base of " + data + " to " + path);
    }
    const std::uint32_t header[2] = {static_cast<std::uint32_t>(base),
                                     static_cast<std::uint32_t>(dim)};
    output.write(reinterpret_cast<const char*>(header), sizeof(header));
    // [2]
    std::vector<char> chunk(1 << 26);
    std::size_t left = base * dim * sizeof(T);
    while (left > 0) {
        const std::size_t size = std::min(left, chunk.size());
        if (!input.read(chunk.data(), static_cast<std::streamsize>(size)) ||
            !output.write(chunk.data(), static_cast<std::streamsize>(size))) {
            fnctbench::die("copy the base of " + data + " to " + path);
        }
        left -= size;
    }
}

static std::uint32_t chunks(const std::string& pivots) {
    // [1]
    std::vector<std::uint64_t> offsets;
    std::size_t rows = 0, cols = 0;
    pipeann::load_bin<std::uint64_t>(pivots, offsets, rows, cols);
    // [2]
    pipeann::get_bin_metadata(pivots, rows, cols, offsets[3]);
    return static_cast<std::uint32_t>(rows - 1);
}

template <class T>
static void sweep(fnctbench::Bench<fnctbench::PipeIndex<T>&, OdinProfile, T>& bench,
                  DynamicIndex<T>& index, std::span<const T> query, const std::string& label,
                  std::span<const std::pair<std::size_t, std::uint32_t>> plan,
                  std::uint32_t after, std::size_t repeat) {
    // [1]
    for (const auto& [k, width] : plan) {
        index.set_search_beam_width(width);
        for (std::size_t r = 0; r < repeat; r++) {
            bench.runverify(label + "_k" + std::to_string(k) + "_r" + std::to_string(r), query, k);
        }
    }
    // [2]
    index.set_search_beam_width(after);
}

template <class T>
static double retire(DynamicIndex<T>& index, const fnctbench::Range& fresh, std::uint32_t nthread) {
    // [1]
    const auto begin = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> tags;
    for (std::uint64_t value = fresh.first; value < fresh.second; value++) {
        index.lazy_delete(static_cast<std::uint32_t>(value));
        tags.push_back(static_cast<std::uint32_t>(value));
    }
    // [2]
    index.disk_index()->condadelete(tags, nthread);
    index.disk_index()->condabatch();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}

template <class T>
static int run(std::size_t dim, std::size_t qlen, const fnctbench::Command& cmd, bool conda,
               std::size_t threadx) {
    // [1]
    const char* who = conda ? "conda" : "odin";
    const std::size_t nthread = fnctbench::knob(cmd, "threads", fnctbench::cores() * threadx);
    const std::size_t step = fnctbench::knob(cmd, "step", STEP);
    const std::size_t nquery = fnctbench::knob(cmd, "qlen", qlen);
    const std::string mode = fnctbench::word(cmd, "mode", "churn");
    const std::size_t base =
        std::filesystem::file_size(cmd.graph) / (DEGREE * sizeof(std::uint32_t));
    const std::size_t rows = std::filesystem::file_size(cmd.point) / (dim * sizeof(T));
    const std::size_t spare = step > 0 && rows > base ? (rows - base) / step : 0;
    const std::size_t batches =
        fnctbench::knob(cmd, "batches", std::min(step > 0 ? base / step : 0, spare - (spare > 0)));
    const std::size_t every =
        fnctbench::knob(cmd, "every", std::max<std::size_t>(batches / CHECKPOINTS, 1));
    const std::size_t budget = fnctbench::knob(cmd, "budget", step + SKEW);
    const std::size_t sample = fnctbench::knob(cmd, "sample", SAMPLE);
    const bool reuse = fnctbench::knob(cmd, "reuse", 0) != 0;
    const std::size_t beam = fnctbench::knob(cmd, "beam", 0);
    const std::size_t repeat = fnctbench::knob(cmd, "repeat", REPEAT);
    const std::size_t fill = fnctbench::knob(cmd, "fill", 0);
    const std::size_t grow = fnctbench::knob(cmd, "grow", 0);
    if (mode != "churn" && mode != "grow" && mode != "scan") {
        fnctbench::die("mode=" + mode + ", want churn|grow|scan");
    }
    const bool scan = mode == "scan";
    if (nthread == 0 || step == 0 || every == 0 || (base + fill) % step != 0) {
        fnctbench::die("threads, step and every must be positive, base " + std::to_string(base) +
                       " plus fill " + std::to_string(fill) + " a multiple of step");
    }
    const std::span<const std::uint32_t> scanw =
        fnctbench::knob(cmd, "wide", 0) != 0 ? std::span<const std::uint32_t>(WIDE)
                                             : std::span<const std::uint32_t>(WIDTHS);
    const std::size_t scanned = PASSES * scanw.size() * SCANMIX;
    if (scan && base + scanned > rows) {
        fnctbench::die("the corpus holds " + std::to_string(rows) + " rows, a scan reads " +
                       std::to_string(base + scanned));
    }
    if (!scan && base + fill + (batches + 1) * step > rows) {
        fnctbench::die("the corpus holds " + std::to_string(rows) + " rows, a run of " +
                       std::to_string(batches) + " batches reads " +
                       std::to_string(base + fill + (batches + 1) * step));
    }
    std::fprintf(stderr, "[%s] mode=%s dim=%zu base=%zu fill=%zu grow=%zu step=%zu batches=%zu\n",
                 who, mode.c_str(), dim, base, fill, grow, step, batches);
    std::fprintf(stderr, "[%s] every=%zu\n", who, every);
    std::fprintf(stderr, "[%s] threads=%zu degree=%zu buildl=%u alpha=%.2f sample=%zu\n", who,
                 nthread, DEGREE, BUILD_L, ALPHA, sample);
    std::fprintf(stderr, "[%s] budget=%zu mixk=%zu rate=%zu repeat=%zu\n", who, budget, MIX_K, RATE,
                 repeat);
    // [2]
    const std::string bin = cmd.index + "_base.bin";
    const std::uint32_t bytes = chunks(cmd.quant);
    const bool kept = reuse && std::filesystem::exists(bin) &&
                      std::filesystem::exists(cmd.index + "_disk.index") &&
                      std::filesystem::exists(cmd.index + "_pq_compressed.bin");
    if (!kept) {
        basebin<T>(cmd.point, bin, base, dim);
        std::filesystem::copy_file(cmd.quant, cmd.index + "_pq_pivots.bin",
                                   std::filesystem::copy_options::overwrite_existing);
        pipeann::PQNeighbor<T>(pipeann::Metric::L2).encode(cmd.index, bin, bytes);
    }
    std::fprintf(stderr, "[%s] reuse=%d kept=%d beam=%zu repeat=%zu\n", who, (int) reuse,
                 (int) kept, beam, repeat);
    // [3]
    pipeann::IndexBuildParameters params;
    params.R = static_cast<std::uint32_t>(DEGREE);
    params.L = BUILD_L;
    params.alpha = ALPHA;
    params.num_threads = static_cast<std::uint32_t>(nthread);
    params.max_nthreads = static_cast<std::uint32_t>(nthread * 2);
    params.beam_width =
        static_cast<std::uint32_t>(fnctbench::knob(cmd, "iwidth", params.beam_width));
    params.delete_beam_width =
        static_cast<std::uint32_t>(fnctbench::knob(cmd, "dwidth", params.beam_width));
    params.sampled_nbrs_for_delete = static_cast<std::uint32_t>(sample);
    if (conda) {
        params.conda = 1;
        params.C = CONDA_C;
    }
    DynamicIndex<T> index(static_cast<std::uint32_t>(dim), pipeann::Metric::L2, &params);
    index.set_index_prefix(cmd.index);
    index.omp_set_num_threads(static_cast<std::uint32_t>(nthread));
    index.load_graph(cmd.index, bin, cmd.graph, static_cast<std::uint32_t>(DEGREE), bytes);
    const auto swidth = fnctbench::knob(cmd, "swidth", SEARCHW);
    index.set_search_beam_width(static_cast<std::uint32_t>(swidth));
    const std::uint32_t searchw = index.search_beam_width();
    std::fprintf(stderr, "[%s] default widths search=%u insert=%u delete=%u\n", who, searchw,
                 params.beam_width, params.delete_beam_width);
    auto* disk = index.disk_index();
    disk->meta_.entry_point = ENTRY;
    disk->meta_.entry_point_id = static_cast<decltype(disk->meta_.entry_point_id)>(ENTRY);
    std::fprintf(stderr, "[%s] entry point %zu\n", who, ENTRY);
    if (conda) {
        disk->condaload(bin, base + (scan ? scanned : fill + batches * step),
                        static_cast<std::uint32_t>(disk->meta_.entry_point));
    }
    std::fprintf(stderr, "[%s] load done, %zu points\n", who,
                 static_cast<std::size_t>(index.npoints()));
    // [4]
    const auto settle = conda ? fnctbench::Settle::CONDA : fnctbench::Settle::MERGE;
    fnctbench::PipeIndex<T> tags(index, settle, budget, static_cast<std::uint32_t>(nthread),
                                 static_cast<std::uint32_t>(nthread));
    fnctbench::DataStream<T> stream(cmd.point, dim, rows);
    stream.rewind(base);
    std::optional<std::vector<T>> loaded =
        fnctbench::DataStream<T>(cmd.query, dim, nquery).next(nquery, nquery);
    if (!loaded || loaded->size() != nquery * dim) {
        fnctbench::die("read " + std::to_string(nquery) + " queries from " + cmd.query);
    }
    const std::vector<T> query = std::move(*loaded);
    fnctbench::Bench<fnctbench::PipeIndex<T>&, OdinProfile, T> bench(tags, std::move(stream),
                                                                     {nthread, RATE}, cmd.trace);
    std::vector<std::pair<std::size_t, std::uint32_t>> plan;
    // [5]
    for (std::size_t i = 0; scan && i < scanw.size(); i++) {
        plan.clear();
        for (std::size_t k : BEAMS) {
            plan.emplace_back(k, scanw[i]);
        }
        sweep<T>(bench, index, query, "scan_w" + std::to_string(scanw[i]), plan, searchw, repeat);
    }
    for (std::size_t j = 0; scan && j < PASSES * scanw.size(); j++) {
        const std::size_t pass = j / scanw.size();
        const std::size_t at = j % scanw.size();
        const std::size_t i = pass % 2 == 0 ? at : scanw.size() - 1 - at;
        const std::string width = std::to_string(scanw[i]) + "_p" + std::to_string(pass);
        index.set_update_beam_widths(scanw[i], scanw[i]);
        const fnctbench::Range fresh = bench.runinsert("scanins_w" + width, SCANMIX);
        std::vector<std::uint64_t> remove;
        for (std::uint64_t value = fresh.first; !conda && value < fresh.second; value++) {
            remove.push_back(value);
        }
        if (conda) {
            const double spent = retire<T>(index, fresh, static_cast<std::uint32_t>(nthread));
            std::fprintf(stderr, "[%s] scan delete w%s rows %zu in %.3f s\n", who, width.c_str(),
                         static_cast<std::size_t>(fresh.second - fresh.first), spent);
        } else {
            bench.runmix("scandel_w" + width, 0, 0, 0, remove);
        }
        std::fprintf(stderr, "[%s] scan width %s done, rows %lu to %lu in and out\n", who,
                     width.c_str(), static_cast<unsigned long>(fresh.first),
                     static_cast<unsigned long>(fresh.second));
    }
    index.set_update_beam_widths(params.beam_width, params.delete_beam_width);
    if (scan) {
        bench.unwrap();
        return 0;
    }
    // [6]
    for (std::size_t i = 0; i < BEAMS.size(); i++) {
        if (beam == 0 || BEAMS[i] == beam) {
            plan.emplace_back(BEAMS[i], searchw);
        }
    }
    for (std::size_t k = 0; k * FILL < fill; k++) {
        const std::size_t n = std::min(FILL, fill - k * FILL);
        bench.runinsert("fill_" + std::to_string(k), n);
        std::fprintf(stderr, "[%s] fill %zu done, rows %zu to %zu in\n", who, k + 1,
                     base + k * FILL, base + k * FILL + n);
    }
    sweep<T>(bench, index, query, "sweep_b0", plan, searchw, repeat);
    for (std::size_t b = 0; b < batches; b++) {
        std::vector<std::uint64_t> remove;
        const std::size_t out = b < grow ? 0 : (b - grow) * step;
        for (std::size_t i = 0; mode == "churn" && b >= grow && i < step; i++) {
            remove.push_back(out + i);
        }
        bench.runmix((remove.empty() ? "grow_" : "mix_") + std::to_string(b), step, step, MIX_K,
                     remove);
        std::fprintf(stderr, "[%s] batch %zu done, %zu rows out from %zu, rows %zu to %zu in\n",
                     who, b + 1, remove.size(), out, base + fill + b * step,
                     base + fill + (b + 1) * step);
        if ((b + 1) % every == 0 || b + 1 == batches) {
            sweep<T>(bench, index, query, "sweep_b" + std::to_string(b + 1), plan, searchw, repeat);
        }
    }
    bench.unwrap();
    return 0;
}

static int dispatch(int argc, char** argv, bool conda, std::size_t threadx) {
    // [1]
    setenv("PIPEANN_POLL_SPINS", POLLSPINS, 0);
    if (std::getenv("FNCT_PTRACER") != nullptr) {
        prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
    }
    const fnctbench::Command cmd = fnctbench::command(argc, argv, KEYS);
    if (cmd.label == "sift") {
        return run<std::uint8_t>(128, 10'000, cmd, conda, threadx);
    }
    if (cmd.label == "msturing") {
        return run<float>(100, 10'000, cmd, conda, threadx);
    }
    if (cmd.label == "spacev") {
        return run<std::int8_t>(100, 29'316, cmd, conda, threadx);
    }
    if (cmd.label == "deep") {
        return run<float>(96, 10'000, cmd, conda, threadx);
    }
    fnctbench::die("dataset " + cmd.label + ", want sift|msturing|spacev|deep");
}

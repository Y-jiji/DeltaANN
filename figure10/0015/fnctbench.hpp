#pragma once

#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <sched.h>

namespace fnctbench {

/** @brief One (distance, value) hit of a search row, benchlog's LogHit */
struct LogHit {
    double dist;
    std::uint64_t item;
};

/** @brief One event-log row as benchlog takes it, rssmb null unless rssok, a cell unless known */
struct LogRow {
    const char* label;
    const char* opkey;
    std::uint64_t index;
    const LogHit* found;
    std::size_t nfind;
    bool valid;
    std::uint64_t rssmb;
    bool rssok;
    const std::uint64_t* cells;
    const std::uint8_t* known;
};

/** @brief benchlog's open event log, a zstd Arrow IPC stream */
struct LogFile;

extern "C" {
/**
 * @brief Bytes of benchlog's LogRow, to refuse a library built from another layout
 * @return sizeof its LogRow
 */
std::size_t benchrowsize();

/**
 * @brief Fill remap[..total] with 0..total in the order fnct-bench's runmix shuffles its jobs
 * @param total jobs
 * @param seed seed of the StdRng
 * @param remap total slots to fill
 */
void benchshuffle(std::size_t total, std::uint64_t seed, std::size_t* remap);

/**
 * @brief Create the event log at path with fnct-bench's schema, one u64 column per name
 * @param path log file
 * @param names event column names
 * @param nname count of names
 * @return the open log
 */
LogFile* benchopen(const char* path, const char* const* names, std::size_t nname);

/**
 * @brief Append rows as one record batch, none for no rows
 * @param log the open log
 * @param rows the rows
 * @param nrow count of rows
 */
void benchwrite(LogFile* log, const LogRow* rows, std::size_t nrow);

/**
 * @brief Finish the stream of log and free it
 * @param log the open log
 */
void benchclose(LogFile* log);
}

/** @brief Seed of runmix's job shuffle, fnct-bench's 43 */
constexpr std::uint64_t SHUFFLE_SEED = 43;
/** @brief Interval between two watchdog lines, fnct-bench's 10 s */
constexpr std::chrono::seconds WATCH_TICK{10};

/** @brief Values a run made alive, fnct-bench's Range<u64> as [first, second) */
using Range = std::pair<std::uint64_t, std::uint64_t>;

/** @brief Knobs every run shares: nthread threads a run spreads over, 1 of every lograte op rows
 * kept */
struct BenchConfig {
    std::size_t nthread;
    std::size_t lograte;
};

/** @brief One op's row until a flush, fnct-bench's EventRow, cells its profile's (name, count) */
struct EventRow {
    const char* opkey;
    const std::string* label;
    std::uint64_t index;
    std::optional<std::vector<LogHit>> found;
    std::optional<std::uint64_t> rssmb;
    std::vector<std::pair<const char*, std::size_t>> cells;
};

/** @brief A parked job, run again with false until it returns, or with true to wait it out */
using Retry = std::function<std::optional<std::optional<EventRow>>(bool)>;

/** @brief What a job hands parmap: its row if kept, or itself parked */
using Outcome = std::variant<std::optional<EventRow>, Retry>;

/**
 * @brief Print message to stderr and abort, a Rust panic's counterpart
 * @param message what went wrong
 */
[[noreturn]] static void die(const std::string& message);

/**
 * @brief CPUs the process may run on, Rust's available_parallelism, at least 1
 * @return the CPU count of the affinity mask
 */
static std::size_t cores();

/**
 * @brief The CPU a worker pins to: the (k mod count)-th CPU of the process's affinity mask, so a
 *        worker never leaves a taskset mask that is not 0..n-1
 * @param k the worker's index
 * @return the CPU id
 */
static int nthcpu(std::size_t k);

/**
 * @brief Raise peak to value if lower, Rust's AtomicUsize::fetch_max
 * @param peak the running maximum
 * @param value a sample
 */
static void fetchmax(std::atomic<std::size_t>& peak, std::size_t value);

/**
 * @brief A driver's command line, the same for every system: the dataset label, its raw corpus
 * point, flat graph, query file, quantizer file, index path and event log trace, then key=value
 * knobs
 */
struct Command {
    std::string label;
    std::string point;
    std::string graph;
    std::string query;
    std::string quant;
    std::string index;
    std::string trace;
    std::vector<std::pair<std::string, std::string>> knobs;
};

/**
 * @brief Parse <dataset> <data> <graph> <queries> <quant> <index> <log> [key=value ...], aborting
 * on a missing file argument, a knob without '=', or a key outside keys
 * @param argc argument count
 * @param argv arguments
 * @param keys the knobs the driver takes
 * @return the command
 */
static Command command(int argc, char** argv, std::span<const char* const> keys);

/**
 * @brief Knob name as a count, fallback when absent, aborting on a value that is not one
 * @param cmd the command
 * @param name the key
 * @param fallback the count without it
 * @return the count
 */
static std::size_t knob(const Command& cmd, const char* name, std::size_t fallback);

/**
 * @brief Knob name as text, fallback when absent
 * @param cmd the command
 * @param name the key
 * @param fallback the text without it
 * @return the text
 */
static std::string word(const Command& cmd, const char* name, const std::string& fallback);

/**
 * @brief fnct-bench's DataStream: a cursor over a headerless raw file of total points of ndims
 * values
 * @tparam T element type of a vector
 */
template <class T>
class DataStream {
  public:
    /**
     * @brief Open path, aborting when it cannot be
     * @param path raw point file
     * @param dim values per point
     * @param len points the stream holds
     */
    DataStream(const std::string& path, std::size_t dim, std::size_t len);

    /**
     * @brief Move the next read to point at
     * @param at the point the next read starts at
     */
    void rewind(std::size_t at);

    /**
     * @brief Point the next read starts at
     * @return the cursor
     */
    std::size_t position() const;

    /**
     * @brief Values per point
     * @return the dimension
     */
    std::size_t dim() const;

    /**
     * @brief Read up to width points from the cursor, advancing it by at most step
     * @param step points to consume
     * @param width points to read
     * @return the points read, none once the cursor is past the end
     */
    std::optional<std::vector<T>> next(std::size_t step, std::size_t width);

  private:
    std::size_t ndims;
    std::size_t total;
    std::size_t place;
    std::ifstream input;
};

/** @brief fnct-bench's WatchDog: prints a run's progress to stderr every interval, then an exit
 * line */
class WatchDog {
  public:
    /**
     * @brief Start the clock of a watchdog over total units
     * @param tag label of every line
     * @param total units to finish
     */
    WatchDog(std::string tag, std::size_t total);

    /** @brief One unit done; the last one stops the watch */
    void progress();

    /** @brief Stop the watch even if units remain */
    void shutdown();

    /**
     * @brief Print units done and their rate every interval until stopped, then the exit line
     * @param interval sleep between two lines
     * @param attrs renders the tail of every line
     */
    void watch(std::chrono::seconds interval, std::function<std::string()> attrs);

  private:
    /**
     * @brief Wait for interval or a stop, whichever is first
     * @param interval the longest wait
     * @return whether the watch is stopped
     */
    bool waitstop(std::chrono::seconds interval);

    std::string label;
    std::atomic<std::size_t> count;
    std::size_t total;
    std::mutex guard;
    std::condition_variable alarm;
    bool ended;
    std::chrono::steady_clock::time_point epoch;
};

/**
 * @brief fnct-bench's Bench over DB in TagIndex's place, by value or reference: bool insert(tag,
 * point, profile), bool remove(tag), size_t search(point, knn, profile) from many threads; if any,
 * tryinsert, none while DB holds inserts off, parks the insert on its worker, detach() runs at a
 * worker's exit, graphdump(each) serves vectorgraphdump; P has event(name, n) and iter()
 * @tparam DB the tagged index
 * @tparam P profile of one op
 * @tparam T element type of a vector
 */
template <class DB, class P, class T>
class Bench {
  public:
    using Dist = typename std::remove_reference_t<DB>::Dist;

    /**
     * @brief fnct-bench's wrap: bench db with stream, positioned at the next value to hand out
     * @param db the tagged index
     * @param data the base points
     * @param knobs the knobs every run shares
     * @param path the event log, an Arrow IPC stream
     */
    Bench(DB db, DataStream<T> data, BenchConfig knobs, const std::string& path);

    /** @brief Finish the event log unless unwrap already did */
    ~Bench();

    Bench(const Bench&) = delete;
    Bench& operator=(const Bench&) = delete;

    /**
     * @brief Finish the event log and hand the tagged index back
     * @return the tagged index
     */
    DB unwrap();

    /** @brief Print the index to stderr by its operator<< */
    void print() const;

    /**
     * @brief Read the next insert + search points, consuming the first insert: insert i goes in
     * under its stream position right after removing remove[i] if any, the rest are searched for k
     *        neighbors, and remove[insert..] go alone, all shuffled; SHAME(TALLFUNC)
     * @param tag tag of every row
     * @param insert points to insert
     * @param search points to search
     * @param k neighbors per search
     * @param remove values to remove
     * @return the values made alive
     */
    Range runmix(const std::string& tag, std::size_t insert, std::size_t search, std::size_t k,
                 std::span<const std::uint64_t> remove);

    /**
     * @brief runmix with insert points and nothing else
     * @param tag tag of every row
     * @param insert points to insert
     * @return the values made alive
     */
    Range runinsert(const std::string& tag, std::size_t insert);

    /**
     * @brief runmix searching the next search points for k neighbors
     * @param tag tag of every row
     * @param search points to search
     * @param k neighbors per search
     * @return an empty range
     */
    Range runsearch(const std::string& tag, std::size_t search, std::size_t k);

    /**
     * @brief runmix replacing every value of remove by a new one
     * @param tag tag of every row
     * @param remove values to remove
     * @return the values made alive
     */
    Range runupdate(const std::string& tag, std::span<const std::uint64_t> remove);

    /**
     * @brief Search every point of point for k neighbors, logging every row
     * @param tag tag of every row
     * @param point the points, dim values each
     * @param k neighbors per search
     */
    void runverify(const std::string& tag, std::span<const T> point, std::size_t k);

    /**
     * @brief Finish the event log and write one graphdump pass, row i for id i: graph its
     * out-neighbors as u32 padded with UINT32_MAX to degree, vector its raw vector
     * @param degree width of a graph row
     * @param graph sink of the graph rows
     * @param vector sink of the vectors
     * @return false, writing nothing, when an id names more than degree neighbors, or on a write
     * error
     */
    bool vectorgraphdump(std::size_t degree, std::ostream& graph, std::ostream& vector);

  private:
    /**
     * @brief Watch hook rendering the resident set in MB, folding it into peak
     * @param peak the running maximum
     * @return the hook
     */
    static std::function<std::string()> rsswatch(std::atomic<std::size_t>& peak);

    /**
     * @brief fnct-io's parmap: nthread pinned threads pull 0..total off one cursor, rows kept in
     * thread order; a worker retries its parked jobs before each next one and waits them out last
     * @tparam MAP callable from an index to an Outcome
     * @param total jobs
     * @param map the job
     * @return the kept rows
     */
    template <class MAP>
    std::vector<EventRow> parmap(std::size_t total, const MAP& map);

    /**
     * @brief Append rows to the event log as one batch, nothing for no rows
     * @param rows rows of one run
     */
    void logflush(const std::vector<EventRow>& rows);

    DB engine;
    LogFile* writer;
    DataStream<T> stream;
    BenchConfig config;
    std::vector<std::string> events;
};

[[noreturn]] static void die(const std::string& message) {
    // [1]
    std::fprintf(stderr, "%s\n", message.c_str());
    std::abort();
}

static std::size_t cores() {
    // [1]
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0 && CPU_COUNT(&mask) > 0) {
        return static_cast<std::size_t>(CPU_COUNT(&mask));
    }
    return 1;
}

static int nthcpu(std::size_t k) {
    // [1]
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0 || CPU_COUNT(&mask) == 0) {
        return static_cast<int>(k);
    }
    // [2]
    std::size_t want = k % static_cast<std::size_t>(CPU_COUNT(&mask));
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (CPU_ISSET(cpu, &mask) && want-- == 0) {
            return cpu;
        }
    }
    return static_cast<int>(k);
}

static void fetchmax(std::atomic<std::size_t>& peak, std::size_t value) {
    // [1]
    std::size_t seen = peak.load(std::memory_order_relaxed);
    while (seen < value && !peak.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
    }
}

static Command command(int argc, char** argv, std::span<const char* const> keys) {
    // [1]
    if (argc < 8) {
        die(std::string("usage: ") + argv[0] +
            " <dataset> <data> <graph> <queries> <quant> <index> <log> [key=value ...]");
    }
    Command cmd{argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], {}};
    // [2]
    for (int at = 8; at < argc; at++) {
        const std::string text = argv[at];
        const std::size_t cut = text.find('=');
        if (cut == std::string::npos) {
            die("knob " + text + " has no '='");
        }
        const std::string key = text.substr(0, cut);
        bool known = false;
        for (const char* allowed : keys) {
            known = known || key == allowed;
        }
        if (!known) {
            die("unknown knob " + key);
        }
        cmd.knobs.emplace_back(key, text.substr(cut + 1));
    }
    return cmd;
}

static std::size_t knob(const Command& cmd, const char* name, std::size_t fallback) {
    // [1]
    for (const auto& [key, value] : cmd.knobs) {
        if (key != name) {
            continue;
        }
        std::size_t count = 0;
        const char* tail = value.data() + value.size();
        if (value.empty() || std::from_chars(value.data(), tail, count).ptr != tail) {
            die("knob " + key + "=" + value + " is not a count");
        }
        return count;
    }
    return fallback;
}

static std::string word(const Command& cmd, const char* name, const std::string& fallback) {
    // [1]
    for (const auto& [key, value] : cmd.knobs) {
        if (key == name) {
            return value;
        }
    }
    return fallback;
}

template <class T>
DataStream<T>::DataStream(const std::string& path, std::size_t dim, std::size_t len)
    : ndims(dim), total(len), place(0), input(path, std::ios::binary) {
    // [1]
    if (!input) {
        die("open dataset file " + path);
    }
}

template <class T>
void DataStream<T>::rewind(std::size_t at) {
    place = at;
}

template <class T>
std::size_t DataStream<T>::position() const {
    return place;
}

template <class T>
std::size_t DataStream<T>::dim() const {
    return ndims;
}

template <class T>
std::optional<std::vector<T>> DataStream<T>::next(std::size_t step, std::size_t width) {
    // [1]
    if (place >= total) {
        return std::nullopt;
    }
    const std::size_t count = std::min(width, total - place);
    input.seekg(static_cast<std::streamoff>(place * ndims * sizeof(T)));
    if (!input) {
        die("seek dataset");
    }
    // [2]
    std::vector<T> out(count * ndims);
    input.read(reinterpret_cast<char*>(out.data()),
               static_cast<std::streamsize>(out.size() * sizeof(T)));
    if (!input) {
        die("read batch");
    }
    place += std::min(step, count);
    return out;
}

inline WatchDog::WatchDog(std::string tag, std::size_t total)
    : label(std::move(tag)), count(0), total(total), ended(false),
      epoch(std::chrono::steady_clock::now()) {}

inline void WatchDog::progress() {
    // [1]
    const std::size_t curr = count.fetch_add(1, std::memory_order_relaxed) + 1;
    if (curr >= total) {
        {
            std::lock_guard<std::mutex> lock(guard);
            ended = true;
        }
        alarm.notify_one();
    }
}

inline void WatchDog::shutdown() {
    // [1]
    {
        std::lock_guard<std::mutex> lock(guard);
        ended = true;
    }
    alarm.notify_one();
}

inline void WatchDog::watch(std::chrono::seconds interval, std::function<std::string()> attrs) {
    // [1]
    std::size_t prev = 0;
    const double seconds = std::chrono::duration<double>(interval).count();
    while (!waitstop(interval)) {
        const std::size_t curr = count.load(std::memory_order_relaxed);
        const double throughput = static_cast<double>(curr - prev) / seconds;
        const std::string tail = attrs();
        std::fprintf(stderr, "[%s] %zu/%zu %.0f/s %s\n", label.c_str(), curr, total, throughput,
                     tail.c_str());
        prev = curr;
    }
    // [2]
    const std::size_t done = count.load(std::memory_order_relaxed);
    const double span =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count();
    const double rate = span > 0.0 ? static_cast<double>(done) / span : 0.0;
    const std::string tail = attrs();
    std::fprintf(stderr, "[%s] exit %zu/%zu in %.1f s %.0f/s %s\n", label.c_str(), done, total,
                 span, rate, tail.c_str());
}

inline bool WatchDog::waitstop(std::chrono::seconds interval) {
    // [1]
    if (total <= count.load(std::memory_order_relaxed)) {
        return true;
    }
    std::unique_lock<std::mutex> lock(guard);
    if (ended) {
        return true;
    }
    alarm.wait_for(lock, interval);
    return ended;
}

template <class DB, class P, class T>
Bench<DB, P, T>::Bench(DB db, DataStream<T> data, BenchConfig knobs, const std::string& path)
    : engine(static_cast<DB&&>(db)), writer(nullptr), stream(std::move(data)), config(knobs) {
    // [1]
    if (benchrowsize() != sizeof(LogRow)) {
        die("benchlog's LogRow has " + std::to_string(benchrowsize()) + " bytes, this header's " +
            std::to_string(sizeof(LogRow)) + ": rebuild 0015/benchlog");
    }
    for (const auto& [name, value] : P().iter()) {
        events.emplace_back(name);
    }
    std::vector<const char*> names;
    for (const std::string& name : events) {
        names.push_back(name.c_str());
    }
    writer = benchopen(path.c_str(), names.data(), names.size());
}

template <class DB, class P, class T>
Bench<DB, P, T>::~Bench() {
    // [1]
    if (writer != nullptr) {
        benchclose(writer);
    }
}

template <class DB, class P, class T>
DB Bench<DB, P, T>::unwrap() {
    // [1]
    if (writer != nullptr) {
        benchclose(writer);
        writer = nullptr;
    }
    return static_cast<DB&&>(engine);
}

template <class DB, class P, class T>
void Bench<DB, P, T>::print() const {
    std::cerr << engine << '\n';
}

template <class DB, class P, class T>
Range Bench<DB, P, T>::runmix(const std::string& tag, std::size_t insert, std::size_t search,
                              std::size_t k, std::span<const std::uint64_t> remove) {
    // [1]
    const std::uint64_t base = stream.position();
    const std::size_t width = insert + search;
    const std::size_t extra = remove.size() > insert ? remove.size() - insert : 0;
    const std::size_t total = width + extra;
    if (total == 0) {
        return {base, base};
    }
    const std::size_t dim = stream.dim();
    std::vector<T> batch;
    if (width > 0) {
        std::optional<std::vector<T>> read = stream.next(insert, width);
        if (!read || read->size() != width * dim) {
            die("the stream ends before this run");
        }
        batch = std::move(*read);
    }
    // [2]
    WatchDog writedog(tag + " (write)", insert + extra);
    WatchDog searchdog(tag + " (search)", search);
    std::vector<std::size_t> remap(total);
    benchshuffle(total, SHUFFLE_SEED, remap.data());
    const auto keep = [&](std::size_t i, EventRow row) -> std::optional<EventRow> {
        if (i % config.lograte == 0) {
            return row;
        }
        return std::nullopt;
    };
    const auto job = [&](std::size_t i) -> Outcome {
        const std::size_t index = remap[i];
        P profile;
        if (index < insert) {
            if (index < remove.size() && !engine.remove(remove[index])) {
                die("a run removes value " + std::to_string(remove[index]) + ", which is not live");
            }
            const std::uint64_t value = base + index;
            const T* point = batch.data() + index * dim;
            const auto finish = [&, i, index, value](P& done, bool fresh) {
                done.event("end", 1);
                if (!fresh) {
                    die("a run inserts value " + std::to_string(value) +
                        ", which was inserted before");
                }
                writedog.progress();
                return keep(i, EventRow{"insert", &tag, index, std::nullopt, std::nullopt,
                                        done.iter()});
            };
            profile.event("start", 1);
            if constexpr (requires(std::remove_reference_t<DB>& db, std::uint64_t v, const T* x,
                                   P& f) { db.tryinsert(v, x, f); }) {
                const std::optional<bool> fresh = engine.tryinsert(value, point, profile);
                if (!fresh) {
                    return Retry([this, finish, value, point, profile](
                                     bool wait) mutable -> std::optional<std::optional<EventRow>> {
                        const std::optional<bool> again =
                            wait ? std::optional<bool>(engine.insert(value, point, profile))
                                 : engine.tryinsert(value, point, profile);
                        if (!again) {
                            return std::nullopt;
                        }
                        return finish(profile, *again);
                    });
                }
                return finish(profile, *fresh);
            } else {
                return finish(profile, engine.insert(value, point, profile));
            }
        }
        if (index < width) {
            std::vector<std::pair<Dist, std::uint64_t>> knn(k, {Dist{}, 0});
            profile.event("start", 1);
            const std::size_t n =
                engine.search(batch.data() + index * dim, std::span(knn), profile);
            std::vector<LogHit> hits;
            for (std::size_t at = 0; at < n; at++) {
                hits.push_back({static_cast<double>(knn[at].first), knn[at].second});
            }
            profile.event("end", 1);
            searchdog.progress();
            return keep(i, EventRow{"search", &tag, index, std::move(hits), std::nullopt,
                                    profile.iter()});
        }
        const std::size_t at = insert + (index - width);
        profile.event("start", 1);
        const bool gone = engine.remove(remove[at]);
        profile.event("end", 1);
        if (!gone) {
            die("a run removes value " + std::to_string(remove[at]) + ", which is not live");
        }
        writedog.progress();
        return keep(i, EventRow{"remove", &tag, at, std::nullopt, std::nullopt, profile.iter()});
    };
    // [3]
    P span;
    span.event("start", 1);
    std::atomic<std::size_t> peak{0};
    std::thread searchwatch([&] { searchdog.watch(WATCH_TICK, rsswatch(peak)); });
    std::thread writewatch([&] { writedog.watch(WATCH_TICK, rsswatch(peak)); });
    std::vector<EventRow> rows = parmap(total, job);
    searchwatch.join();
    writewatch.join();
    span.event("end", 1);
    // [4]
    rows.push_back(EventRow{"stage", &tag, 0, std::nullopt, peak.load(std::memory_order_relaxed),
                            span.iter()});
    searchdog.shutdown();
    writedog.shutdown();
    logflush(rows);
    return {base, base + insert};
}

template <class DB, class P, class T>
Range Bench<DB, P, T>::runinsert(const std::string& tag, std::size_t insert) {
    return runmix(tag, insert, 0, 0, {});
}

template <class DB, class P, class T>
Range Bench<DB, P, T>::runsearch(const std::string& tag, std::size_t search, std::size_t k) {
    return runmix(tag, 0, search, k, {});
}

template <class DB, class P, class T>
Range Bench<DB, P, T>::runupdate(const std::string& tag, std::span<const std::uint64_t> remove) {
    return runmix(tag, remove.size(), 0, 0, remove);
}

template <class DB, class P, class T>
void Bench<DB, P, T>::runverify(const std::string& tag, std::span<const T> point, std::size_t k) {
    // [1]
    const std::size_t dim = stream.dim();
    const std::size_t count = point.size() / dim;
    WatchDog dog(tag, count);
    const auto job = [&](std::size_t i) -> std::optional<EventRow> {
        std::vector<std::pair<Dist, std::uint64_t>> knn(k, {Dist{}, 0});
        P profile;
        profile.event("start", 1);
        const std::size_t n = engine.search(point.data() + i * dim, std::span(knn), profile);
        std::vector<LogHit> hits;
        for (std::size_t at = 0; at < n; at++) {
            hits.push_back({static_cast<double>(knn[at].first), knn[at].second});
        }
        profile.event("end", 1);
        dog.progress();
        return EventRow{"search", &tag, i, std::move(hits), std::nullopt, profile.iter()};
    };
    // [2]
    P span;
    span.event("start", 1);
    std::atomic<std::size_t> peak{0};
    std::thread watcher([&] { dog.watch(WATCH_TICK, rsswatch(peak)); });
    std::vector<EventRow> rows = parmap(count, [&](std::size_t i) -> Outcome { return job(i); });
    watcher.join();
    span.event("end", 1);
    // [3]
    rows.push_back(EventRow{"stage", &tag, 0, std::nullopt, peak.load(std::memory_order_relaxed),
                            span.iter()});
    logflush(rows);
    dog.shutdown();
}

template <class DB, class P, class T>
bool Bench<DB, P, T>::vectorgraphdump(std::size_t degree, std::ostream& graph,
                                      std::ostream& vector) {
    // [1]
    const std::size_t dim = stream.dim();
    DB db = unwrap();
    std::vector<std::uint32_t> rows;
    std::vector<T> points;
    bool over = false;
    db.graphdump(
        [&](const T* point, std::span<const std::uint32_t> near, std::optional<std::uint64_t>) {
            over |= near.size() > degree;
            for (std::size_t j = 0; j < degree; j++) {
                rows.push_back(j < near.size() ? near[j] : UINT32_MAX);
            }
            points.insert(points.end(), point, point + dim);
        });
    // [2]
    if (over) {
        return false;
    }
    if (!graph.write(reinterpret_cast<const char*>(rows.data()),
                     static_cast<std::streamsize>(rows.size() * sizeof(std::uint32_t)))) {
        return false;
    }
    return static_cast<bool>(vector.write(reinterpret_cast<const char*>(points.data()),
                                          static_cast<std::streamsize>(points.size() * sizeof(T))));
}

template <class DB, class P, class T>
std::function<std::string()> Bench<DB, P, T>::rsswatch(std::atomic<std::size_t>& peak) {
    return [&peak] {
        // [1]
        std::size_t rss = 0;
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.rfind("VmRSS:", 0) != 0) {
                continue;
            }
            const std::size_t first = line.find_first_not_of(" \t", 6);
            const std::size_t last = line.find_first_of(" \t", first);
            std::size_t kb = 0;
            if (first != std::string::npos) {
                const char* tail = line.data() + (last == std::string::npos ? line.size() : last);
                if (std::from_chars(line.data() + first, tail, kb).ptr != tail) {
                    kb = 0;
                }
            }
            rss = kb / 1024;
            break;
        }
        // [2]
        fetchmax(peak, rss);
        return "rss=" + std::to_string(rss) + "MB";
    };
}

template <class DB, class P, class T>
template <class MAP>
std::vector<EventRow> Bench<DB, P, T>::parmap(std::size_t total, const MAP& map) {
    // [1]
    std::atomic<std::size_t> cursor{0};
    std::vector<std::vector<EventRow>> kept(config.nthread);
    std::vector<std::thread> pool;
    for (std::size_t core = 0; core < config.nthread; core++) {
        pool.emplace_back([&, core] {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(nthcpu(core), &set);
            sched_setaffinity(0, sizeof(set), &set);
            std::deque<Retry> parked;
            const auto resume = [&](bool wait) {
                while (!parked.empty()) {
                    std::optional<std::optional<EventRow>> done = parked.front()(wait);
                    if (!done) {
                        return;
                    }
                    if (*done) {
                        kept[core].push_back(std::move(**done));
                    }
                    parked.pop_front();
                }
            };
            for (std::size_t at = cursor.fetch_add(1, std::memory_order_relaxed); at < total;
                 at = cursor.fetch_add(1, std::memory_order_relaxed)) {
                resume(false);
                Outcome out = map(at);
                if (std::optional<EventRow>* row = std::get_if<std::optional<EventRow>>(&out)) {
                    if (*row) {
                        kept[core].push_back(std::move(**row));
                    }
                } else {
                    parked.push_back(std::move(std::get<Retry>(out)));
                }
            }
            resume(true);
            if constexpr (requires(std::remove_reference_t<DB>& db) { db.detach(); }) {
                engine.detach();
            }
        });
    }
    // [2]
    for (std::thread& worker : pool) {
        worker.join();
    }
    std::vector<EventRow> rows;
    for (std::vector<EventRow>& part : kept) {
        for (EventRow& row : part) {
            rows.push_back(std::move(row));
        }
    }
    return rows;
}

template <class DB, class P, class T>
void Bench<DB, P, T>::logflush(const std::vector<EventRow>& rows) {
    // [1]
    if (rows.empty()) {
        return;
    }
    if (writer == nullptr) {
        die("empty log writer");
    }
    const std::size_t width = events.size();
    std::vector<std::uint64_t> cells(rows.size() * width, 0);
    std::vector<std::uint8_t> known(rows.size() * width, 0);
    std::vector<LogRow> flat(rows.size());
    // [2]
    for (std::size_t r = 0; r < rows.size(); r++) {
        const EventRow& row = rows[r];
        for (std::size_t at = 0; at < width; at++) {
            for (const auto& [name, value] : row.cells) {
                if (events[at] == name) {
                    cells[r * width + at] = value;
                    known[r * width + at] = 1;
                    break;
                }
            }
        }
        flat[r] = LogRow{row.label->c_str(),
                         row.opkey,
                         row.index,
                         row.found ? row.found->data() : nullptr,
                         row.found ? row.found->size() : 0,
                         row.found.has_value(),
                         row.rssmb.value_or(0),
                         row.rssmb.has_value(),
                         cells.data() + r * width,
                         known.data() + r * width};
    }
    benchwrite(writer, flat.data(), flat.size());
}

} // namespace fnctbench

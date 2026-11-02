#pragma once

#include "fnctbench.hpp"

#include "dynamic_index.h"
#include "utils/iocount.h"
#include "utils/percentile_stats.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <omp.h>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fnctbench {

/** @brief How an index retires deleted vectors: OdinANN's merge, or CONDA's delete at once */
enum class Settle : std::uint8_t { MERGE, CONDA };

/**
 * @brief OdinANN's DynamicIndex in TagIndex's place, inserting, deleting and searching by tag.
 * Under MERGE, deletes are lazy; once budget of them are pending, its maintenance thread drains
 * running inserts, holds new ones off and merges them as PipeANN would, searches going on
 * throughout, and every insert held off during one merge is let in before the next one starts.
 * Under CONDA, each delete runs CONDA's Delete at once, beside inserts and searches
 * @tparam T element type of a vector
 */
template <class T>
class PipeIndex {
  public:
    using Dist = float;

    /**
     * @brief Serve index, its background writer running as PipeANN starts it at load; aborts unless
     * the index's max_nthreads covers callers and nthread, so every buffer an op waits on exists,
     * its conda flag matches mode, limit is positive and the index has no navigation graph
     * @param index the loaded index, condaload already called under CONDA
     * @param mode how pending deletes are retired
     * @param limit pending deletes that start a retirement
     * @param nthread threads of one retirement
     * @param callers the most threads that call the index at once
     */
    PipeIndex(DynamicIndex<T>& index, Settle mode, std::size_t limit, std::uint32_t nthread,
              std::uint32_t callers);

    /** @brief Join the maintenance thread, then stop the background writer */
    ~PipeIndex();

    PipeIndex(const PipeIndex&) = delete;
    PipeIndex& operator=(const PipeIndex&) = delete;

    /**
     * @brief Insert point under tag, waiting only while a retirement holds inserts off
     * @tparam P profile of the op
     * @param tag the vector's tag
     * @param point its values
     * @param profile gets hop, dist, estim, cold, pass, sdc and wrote
     * @return true, as PipeANN keeps no record of tags ever inserted
     */
    template <class P>
    bool insert(std::uint64_t tag, const T* point, P& profile);

    /**
     * @brief Insert point under tag unless a retirement holds inserts off, without waiting
     * @tparam P profile of the op
     * @param tag the vector's tag
     * @param point its values
     * @param profile gets insert's counters if the insert ran
     * @return true if it ran, none if held off, the tag then let in before the next retirement
     */
    template <class P>
    std::optional<bool> tryinsert(std::uint64_t tag, const T* point, P& profile);

    /**
     * @brief Delete tag: under MERGE lazily, counting it toward the next merge; under CONDA at once
     * @param tag the vector's tag
     * @return true, as PipeANN keeps no record of live tags, or under CONDA whether tag was live
     */
    bool remove(std::uint64_t tag);

    /**
     * @brief Find the knn.size() nearest live tags of point with a candidate list as wide
     * @tparam P profile of the op
     * @param point the query
     * @param knn (distance, tag) slots
     * @param profile gets hop, dist, estim and cold
     * @return how many slots lead knn
     */
    template <class P>
    std::size_t search(const T* point, std::span<std::pair<float, std::uint64_t>> knn,
                       P& profile);

    /** @brief Release the calling thread's io_uring context, at a bench worker's exit */
    void detach();

  private:
    /**
     * @brief Insert point under tag into the engine, the gate entered, then leave the gate
     * @tparam P profile of the op
     * @param tag the vector's tag
     * @param point its values
     * @param profile gets hop, dist, estim, cold, pass, sdc and wrote
     */
    template <class P>
    void write(std::uint64_t tag, const T* point, P& profile);

    /** @brief Leave the gate, waking the maintenance thread if it waits on the last insert */
    void leave();

    /**
     * @brief Body of the maintenance thread, idle under CONDA: one merge each time the limit is
     * reached, a delete is still pending and every insert the last one held off has been let in; at
     * its exit, every thread of its OpenMP pool releases the io_uring context a merge gave it
     */
    void maintain();

    DynamicIndex<T>* engine;
    Settle policy;
    std::size_t budget;
    std::uint32_t fanout;
    std::atomic<std::size_t> queued;
    std::mutex locker;
    std::condition_variable signal;
    std::atomic<bool> closed;
    std::atomic<std::size_t> active;
    std::atomic<bool> wanted;
    std::unordered_set<std::uint32_t> parked;
    std::atomic<std::size_t> holdup;
    bool ending;
    std::array<std::string, 2> prefix;
    std::uint64_t rounds;
    std::thread keeper;
};

template <class T>
PipeIndex<T>::PipeIndex(DynamicIndex<T>& index, Settle mode, std::size_t limit,
                        std::uint32_t nthread, std::uint32_t callers)
    : engine(&index), policy(mode), budget(limit), fanout(nthread), queued(0), closed(false),
      active(0), wanted(false), holdup(0), ending(false),
      prefix{index.index_prefix(), index.index_prefix() + "_merge"}, rounds(0) {
    // [1]
    const std::uint64_t threads = index.build_params().max_nthreads;
    if (threads < static_cast<std::uint64_t>(callers) + nthread) {
        die("max_nthreads " + std::to_string(threads) + " gives fewer query buffers than " +
            std::to_string(callers) + " callers and " + std::to_string(nthread) +
            " retirement threads may hold; raise it to their sum");
    }
    if (index.disk_index()->mem_index_) {
        die("the index has a navigation graph, whose ids no retirement remaps");
    }
    if ((index.build_params().conda != 0) != (mode == Settle::CONDA) || limit == 0) {
        die("the index's conda flag must match the CONDA mode, and the limit be positive");
    }
    // [2]
    engine->start_bg_io();
    if (policy == Settle::CONDA) {
        engine->disk_index()->condabatch();
    }
    keeper = std::thread([this] { maintain(); });
}

template <class T>
PipeIndex<T>::~PipeIndex() {
    // [1]
    {
        std::lock_guard<std::mutex> lock(locker);
        ending = true;
    }
    signal.notify_all();
    keeper.join();
    engine->stop_bg_io();
}

template <class T>
template <class P>
bool PipeIndex<T>::insert(std::uint64_t tag, const T* point, P& profile) {
    // [1]
    for (;;) {
        if (std::optional<bool> done = tryinsert(tag, point, profile)) {
            return *done;
        }
        std::unique_lock<std::mutex> lock(locker);
        signal.wait(lock, [this] { return !closed.load(); });
    }
}

template <class T>
template <class P>
std::optional<bool> PipeIndex<T>::tryinsert(std::uint64_t tag, const T* point, P& profile) {
    // [1]
    if (tag > UINT32_MAX) {
        die("tag " + std::to_string(tag) + " exceeds PipeANN's 32-bit tags");
    }
    const auto key = static_cast<std::uint32_t>(tag);
    active.fetch_add(1);
    if (closed.load()) {
        std::unique_lock<std::mutex> lock(locker);
        if (closed.load()) {
            parked.insert(key);
            holdup.store(parked.size());
            lock.unlock();
            leave();
            return std::nullopt;
        }
    }
    // [2]
    if (holdup.load() > 0) {
        std::lock_guard<std::mutex> lock(locker);
        if (parked.erase(key) > 0) {
            holdup.store(parked.size());
            if (parked.empty()) {
                signal.notify_all();
            }
        }
    }
    // [3]
    write(tag, point, profile);
    return true;
}

template <class T>
template <class P>
void PipeIndex<T>::write(std::uint64_t tag, const T* point, P& profile) {
    // [1]
    pipeann::QueryStats native;
    const std::uint64_t seen = pipeann::dev_reads();
    engine->insert(point, static_cast<std::uint32_t>(tag), nullptr, &native);
    const std::uint64_t devrd = pipeann::dev_reads() - seen;
    leave();
    // [2]
    profile.event("hop", static_cast<std::size_t>(native.n_hops));
    profile.event("dist", static_cast<std::size_t>(native.n_exact));
    profile.event("estim", static_cast<std::size_t>(native.n_cmps));
    profile.event("cold", static_cast<std::size_t>(devrd));
    profile.event("pass", static_cast<std::size_t>(native.n_passes));
    profile.event("sdc", static_cast<std::size_t>(native.n_sdc));
    profile.event("wrote", static_cast<std::size_t>(native.n_pages_touched));
}

template <class T>
void PipeIndex<T>::leave() {
    // [1]
    if (active.fetch_sub(1) == 1 && closed.load()) {
        std::lock_guard<std::mutex> lock(locker);
        signal.notify_all();
    }
}

template <class T>
bool PipeIndex<T>::remove(std::uint64_t tag) {
    // [1]
    if (tag > UINT32_MAX) {
        die("tag " + std::to_string(tag) + " exceeds PipeANN's 32-bit tags");
    }
    engine->lazy_delete(static_cast<std::uint32_t>(tag));
    if (policy == Settle::CONDA) {
        return engine->disk_index()->condaremove(static_cast<std::uint32_t>(tag));
    }
    // [2]
    if (queued.fetch_add(1) + 1 == budget) {
        wanted.store(true);
        std::lock_guard<std::mutex> lock(locker);
        signal.notify_all();
    }
    return true;
}

template <class T>
template <class P>
std::size_t PipeIndex<T>::search(const T* point, std::span<std::pair<float, std::uint64_t>> knn,
                                 P& profile) {
    // [1]
    const auto width = static_cast<std::uint32_t>(knn.size());
    if (width == 0) {
        return 0;
    }
    std::vector<std::uint32_t> tags(width);
    std::vector<float> dists(width);
    pipeann::QueryStats native;
    const std::uint64_t seen = pipeann::dev_reads();
    const std::size_t n = engine->search(point, width, width, tags.data(), dists.data(), &native);
    profile.event("hop", static_cast<std::size_t>(native.n_hops));
    profile.event("dist", static_cast<std::size_t>(native.n_exact));
    profile.event("estim", static_cast<std::size_t>(native.n_cmps));
    profile.event("cold", static_cast<std::size_t>(pipeann::dev_reads() - seen));
    // [2]
    for (std::size_t at = 0; at < n; at++) {
        knn[at] = {dists[at], tags[at]};
    }
    return n;
}

template <class T>
void PipeIndex<T>::detach() {
    engine->deregister_thread();
}

template <class T>
void PipeIndex<T>::maintain() {
    // [1]
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(locker);
            signal.wait(lock, [this] { return (wanted.load() && parked.empty()) || ending; });
            if (ending) {
                break;
            }
            queued.store(0);
            wanted.store(false);
            if (engine->pending_deletes() == 0) {
                continue;
            }
            closed.store(true);
            signal.wait(lock, [this] { return active.load() == 0; });
        }
        // [2] a merge rewrites the index with no write in flight
        engine->stop_bg_io();
        rounds++;
        engine->save(prefix[rounds % 2], fanout);
        engine->start_bg_io();
        if (engine->disk_index()->mem_index_) {
            die("a retirement loaded a navigation graph, whose ids no retirement remaps");
        }
        // [3]
        {
            std::lock_guard<std::mutex> lock(locker);
            closed.store(false);
        }
        signal.notify_all();
    }
    // [4]
    const int team = std::max(static_cast<int>(fanout), omp_get_max_threads());
#pragma omp parallel num_threads(team)
    engine->deregister_thread();
}

}  // namespace fnctbench

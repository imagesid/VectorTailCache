// =============================================================================
// workload_logger.cpp
// Per-query workload logger for VectorTailCache analysis.
//
// Add this instrumentation to cached_beam_search in pq_flash_index.cpp.
// It logs per-query: latency, nodes visited, cache hits/misses, I/O times.
//
// Output: CSV file with one row per query — used for offline analysis
// to derive the optimal PCS formula from real data.
//
// HOW TO USE:
// 1. Add WorkloadLogger to PQFlashIndex as a member
// 2. Call logger.begin_query(query_id) at start of cached_beam_search
// 3. Call logger.record_cache_hit(node_id) / logger.record_miss(node_id, io_us)
//    at the relevant points
// 4. Call logger.end_query(total_us) at the end
// 5. Call logger.save(path) after all queries complete
// =============================================================================

#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <mutex>
#include <atomic>
#include <chrono>

namespace diskann {

struct QueryLog {
    uint64_t query_id;
    float    total_us;       // end-to-end latency
    float    io_us;          // total I/O wait time
    uint32_t nodes_visited;  // total nodes expanded
    uint32_t cache_hits;     // nodes found in DiskANN cache
    uint32_t cache_misses;   // nodes fetched from SSD
    uint32_t n_ios;          // number of SSD reads issued
    // Top-3 most expensive misses (node_id, io_us)
    uint32_t expensive_miss_node[3] = {0,0,0};
    float    expensive_miss_us[3]   = {0,0,0};
};

class WorkloadLogger {
public:
    WorkloadLogger() : _enabled(false), _query_count(0) {}

    void enable(size_t num_queries) {
        _logs.resize(num_queries);
        _enabled = true;
        _query_count = 0;
        std::cout << "[WorkloadLogger] Enabled for " << num_queries << " queries." << std::endl;
    }

    void disable() { _enabled = false; }
    bool is_enabled() const { return _enabled; }

    // Call at start of each query
    void begin_query(uint64_t qid) {
        if (!_enabled) return;
        auto &log = _logs[qid % _logs.size()];
        log.query_id     = qid;
        log.total_us     = 0;
        log.io_us        = 0;
        log.nodes_visited = 0;
        log.cache_hits   = 0;
        log.cache_misses = 0;
        log.n_ios        = 0;
        for (int i=0;i<3;i++) { log.expensive_miss_node[i]=0; log.expensive_miss_us[i]=0; }
    }

    // Call for each cache hit
    void record_cache_hit(uint64_t qid, uint32_t node_id) {
        if (!_enabled) return;
        auto &log = _logs[qid % _logs.size()];
        log.cache_hits++;
        log.nodes_visited++;
    }

    // Call for each SSD read (cache miss)
    void record_miss(uint64_t qid, uint32_t node_id, float io_us, uint32_t n_nodes_in_batch) {
        if (!_enabled) return;
        auto &log = _logs[qid % _logs.size()];
        log.cache_misses += n_nodes_in_batch;
        log.nodes_visited += n_nodes_in_batch;
        log.n_ios++;
        log.io_us += io_us;

        // Track most expensive individual misses
        float per_node = io_us / n_nodes_in_batch;
        for (int i=0; i<3; i++) {
            if (per_node > log.expensive_miss_us[i]) {
                // Shift down
                for (int j=2; j>i; j--) {
                    log.expensive_miss_us[j]  = log.expensive_miss_us[j-1];
                    log.expensive_miss_node[j] = log.expensive_miss_node[j-1];
                }
                log.expensive_miss_us[i]   = per_node;
                log.expensive_miss_node[i] = node_id;
                break;
            }
        }
    }

    // Call at end of each query
    void end_query(uint64_t qid, float total_us) {
        if (!_enabled) return;
        _logs[qid % _logs.size()].total_us = total_us;
        _query_count.fetch_add(1, std::memory_order_relaxed);
    }

    // Save to CSV
    void save(const std::string &path) {
        std::ofstream f(path);
        f << "query_id,total_us,io_us,nodes_visited,cache_hits,cache_misses,"
          << "n_ios,miss_node_0,miss_us_0,miss_node_1,miss_us_1,miss_node_2,miss_us_2\n";
        for (size_t i = 0; i < _logs.size(); i++) {
            auto &log = _logs[i];
            if (log.total_us == 0) continue;
            f << log.query_id << ","
              << log.total_us << "," << log.io_us << ","
              << log.nodes_visited << "," << log.cache_hits << "," << log.cache_misses << ","
              << log.n_ios << ","
              << log.expensive_miss_node[0] << "," << log.expensive_miss_us[0] << ","
              << log.expensive_miss_node[1] << "," << log.expensive_miss_us[1] << ","
              << log.expensive_miss_node[2] << "," << log.expensive_miss_us[2] << "\n";
        }
        std::cout << "[WorkloadLogger] Saved " << _query_count.load()
                  << " query logs to: " << path << std::endl;
    }

private:
    bool _enabled;
    std::atomic<uint32_t> _query_count;
    std::vector<QueryLog> _logs;  // pre-allocated, indexed by query_id
};

} // namespace diskann

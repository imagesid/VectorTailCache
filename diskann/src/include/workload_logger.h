// Copyright (c) VectorTailCache. All rights reserved.
// workload_logger.h — header-only per-query workload logger

#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <atomic>
#include <iostream>

namespace diskann
{

struct QueryLog
{
    uint64_t query_id      = 0;
    float    total_us      = 0;
    float    io_us         = 0;
    uint32_t nodes_visited = 0;
    uint32_t cache_hits    = 0;
    uint32_t cache_misses  = 0;
    uint32_t n_ios         = 0;
    uint32_t expensive_miss_node = 0;
    float    expensive_miss_us   = 0;
};

class WorkloadLogger
{
  public:
    WorkloadLogger() : _enabled(false), _query_count(0) {}

    void enable(size_t num_queries)
    {
        _logs.assign(num_queries, QueryLog{});
        _enabled = true;
        _query_count.store(0);
        std::cout << "[WorkloadLogger] Enabled for " << num_queries << " queries." << std::endl;
    }

    void disable() { _enabled = false; }
    bool is_enabled() const { return _enabled; }

    void begin_query(uint64_t qid)
    {
        if (!_enabled || qid >= _logs.size()) return;
        auto &log            = _logs[qid];
        log.query_id         = qid;
        log.total_us         = 0;
        log.io_us            = 0;
        log.nodes_visited    = 0;
        log.cache_hits       = 0;
        log.cache_misses     = 0;
        log.n_ios            = 0;
        log.expensive_miss_node = 0;
        log.expensive_miss_us   = 0;
    }

    // node_id kept for future per-node hit tracking
    void record_cache_hit(uint64_t qid, uint32_t node_id)
    {
        if (!_enabled || qid >= _logs.size()) return;
        (void)node_id;
        _logs[qid].cache_hits++;
        _logs[qid].nodes_visited++;
    }

    void record_miss_batch(uint64_t qid, uint32_t node_id, float io_us, uint32_t batch_size)
    {
        if (!_enabled || qid >= _logs.size()) return;
        auto &log         = _logs[qid];
        log.cache_misses += batch_size;
        log.nodes_visited += batch_size;
        log.n_ios++;
        log.io_us += io_us;
        float per_node = io_us / (float)batch_size;
        if (per_node > log.expensive_miss_us)
        {
            log.expensive_miss_us   = per_node;
            log.expensive_miss_node = node_id;
        }
    }

    void end_query(uint64_t qid, float total_us)
    {
        if (!_enabled || qid >= _logs.size()) return;
        _logs[qid].total_us = total_us;
        _query_count.fetch_add(1, std::memory_order_relaxed);
    }

    void save(const std::string &path)
    {
        std::ofstream f(path);
        if (!f.is_open())
        {
            std::cerr << "[WorkloadLogger] ERROR: cannot open " << path << std::endl;
            return;
        }
        f << "query_id,total_us,io_us,nodes_visited,cache_hits,"
             "cache_misses,n_ios,expensive_miss_node,expensive_miss_us\n";
        uint32_t written = 0;
        for (auto &log : _logs)
        {
            if (log.total_us == 0) continue;
            f << log.query_id      << ","
              << log.total_us      << ","
              << log.io_us         << ","
              << log.nodes_visited << ","
              << log.cache_hits    << ","
              << log.cache_misses  << ","
              << log.n_ios         << ","
              << log.expensive_miss_node << ","
              << log.expensive_miss_us   << "\n";
            written++;
        }
        std::cout << "[WorkloadLogger] Saved " << written
                  << " query logs → " << path << std::endl;
    }

  private:
    bool                  _enabled;
    std::atomic<uint32_t> _query_count;
    std::vector<QueryLog> _logs;
};

} // namespace diskann

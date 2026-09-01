// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
// pq_flash_index.h
// Modified by VectorTailCache: added PCS (Path Criticality Score) cache policy

#pragma once
#include "common_includes.h"

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq.h"
#include "utils.h"
#include "windows_customizations.h"
#include "scratch.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "workload_logger.h"  // VectorTailCache

#define FULL_PRECISION_REORDER_MULTIPLIER 3

namespace diskann
{

// ── VectorTailCache: per-node profiling stats ─────────────────────────────────
// Collected during the PCS profiling window.
// All fields are atomic so they can be updated from parallel search threads.
// struct NodePCSStats
// {
//     std::atomic<uint32_t> total_visits{0};  // how often this node is visited
//     std::atomic<uint32_t> tail_visits{0};   // visits from queries that exceeded tail_threshold_us
//     std::atomic<float>    miss_latency_us{0.0f}; // cumulative SSD read latency on cache misses (us)
//     std::atomic<uint32_t> miss_count{0};    // number of cache misses for this node
// };
struct NodePCSStats
{
    std::atomic<uint32_t> total_visits{0};
    std::atomic<uint32_t> tail_visits{0};
    std::atomic<float>    miss_latency_us{0.0f};
    std::atomic<uint32_t> miss_count{0};

    // std::atomic is not copyable/movable by default.
    // These constructors allow std::vector<NodePCSStats>::resize().
    NodePCSStats() = default;

    NodePCSStats(NodePCSStats &&other) noexcept
        : total_visits(other.total_visits.load())
        , tail_visits(other.tail_visits.load())
        , miss_latency_us(other.miss_latency_us.load())
        , miss_count(other.miss_count.load())
    {}

    NodePCSStats &operator=(NodePCSStats &&other) noexcept
    {
        total_visits.store(other.total_visits.load());
        tail_visits.store(other.tail_visits.load());
        miss_latency_us.store(other.miss_latency_us.load());
        miss_count.store(other.miss_count.load());
        return *this;
    }
};
// ─────────────────────────────────────────────────────────────────────────────

template <typename T, typename LabelT = uint32_t> class PQFlashIndex
{
  public:
    DISKANN_DLLEXPORT void enable_workload_logging(size_t num_queries) {
        _workload_logger.enable(num_queries);
    }
    DISKANN_DLLEXPORT void save_workload_log(const std::string &path) {
        _workload_logger.save(path);
    }
    DISKANN_DLLEXPORT PQFlashIndex(std::shared_ptr<AlignedFileReader> &fileReader,
                                   diskann::Metric metric = diskann::Metric::L2);
    DISKANN_DLLEXPORT ~PQFlashIndex();

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load(diskann::MemoryMappedFiles &files, uint32_t num_threads, const char *index_prefix);
#else
    DISKANN_DLLEXPORT int load(uint32_t num_threads, const char *index_prefix);
#endif

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load_from_separate_paths(diskann::MemoryMappedFiles &files, uint32_t num_threads,
                                                   const char *index_filepath, const char *pivots_filepath,
                                                   const char *compressed_filepath);
#else
    DISKANN_DLLEXPORT int load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                   const char *pivots_filepath, const char *compressed_filepath);
#endif

    DISKANN_DLLEXPORT void load_cache_list(std::vector<uint32_t> &node_list);

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(MemoryMappedFiles &files, std::string sample_bin,
                                                                   uint64_t l_search, uint64_t beamwidth,
                                                                   uint64_t num_nodes_to_cache, uint32_t nthreads,
                                                                   std::vector<uint32_t> &node_list);
#else
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(std::string sample_bin, uint64_t l_search,
                                                                   uint64_t beamwidth, uint64_t num_nodes_to_cache,
                                                                   uint32_t num_threads,
                                                                   std::vector<uint32_t> &node_list);
#endif

    DISKANN_DLLEXPORT void cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list,
                                            const bool shuffle = false);

    // ── VectorTailCache: PCS cache generation ────────────────────────────────
    // Runs a profiling window of `num_sample_queries` queries from `query_bin`,
    // collects per-node (tail_visits, total_visits, miss_latency) stats,
    // then selects the top-num_nodes_to_cache nodes by PCS score:
    //   PCS(v) = (tail_visits(v) / total_visits(v)) * avg_miss_latency(v)
    // `tail_percentile` (default 0.9) defines what counts as a "tail query".
    DISKANN_DLLEXPORT void generate_pcs_cache_list(std::string query_bin,
                                                   uint64_t l_search,
                                                   uint64_t beamwidth,
                                                   uint64_t num_nodes_to_cache,
                                                   uint32_t num_threads,
                                                   std::vector<uint32_t> &node_list,
                                                   float tail_percentile = 0.90f);
    DISKANN_DLLEXPORT void enable_per_node_latency_log();
    DISKANN_DLLEXPORT void dump_per_node_latency(const std::string &outfile);
    // ─────────────────────────────────────────────────────────────────────────

    DISKANN_DLLEXPORT void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const bool use_reorder_data = false, QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const bool use_filter, const LabelT &filter_label,
                                              const bool use_reorder_data = false, QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const uint32_t io_limit, const bool use_reorder_data = false,
                                              QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const bool use_filter, const LabelT &filter_label,
                                              const uint32_t io_limit, const bool use_reorder_data = false,
                                              QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT LabelT get_converted_label(const std::string &filter_label);

    DISKANN_DLLEXPORT uint32_t range_search(const T *query1, const double range, const uint64_t min_l_search,
                                            const uint64_t max_l_search, std::vector<uint64_t> &indices,
                                            std::vector<float> &distances, const uint64_t min_beam_width,
                                            QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT uint64_t get_data_dim();

    std::shared_ptr<AlignedFileReader> &reader;

    DISKANN_DLLEXPORT diskann::Metric get_metric();

    DISKANN_DLLEXPORT std::vector<bool> read_nodes(const std::vector<uint32_t> &node_ids,
                                                   std::vector<T *> &coord_buffers,
                                                   std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers);

    DISKANN_DLLEXPORT std::vector<std::uint8_t> get_pq_vector(std::uint64_t vid);
    DISKANN_DLLEXPORT uint64_t get_num_points();
    DISKANN_DLLEXPORT uint64_t get_max_degree();

  protected:
    DISKANN_DLLEXPORT void use_medoids_data_as_centroids();
    DISKANN_DLLEXPORT void setup_thread_data(uint64_t nthreads, uint64_t visited_reserve = 4096);
    DISKANN_DLLEXPORT void set_universal_label(const LabelT &label);

  private:
    DISKANN_DLLEXPORT inline bool point_has_label(uint32_t point_id, LabelT label_id);
    std::unordered_map<std::string, LabelT> load_label_map(std::basic_istream<char> &infile);
    DISKANN_DLLEXPORT void parse_label_file(std::basic_istream<char> &infile, size_t &num_pts_labels);
    DISKANN_DLLEXPORT void get_label_file_metadata(const std::string &fileContent, uint32_t &num_pts,
                                                   uint32_t &num_total_labels);
    DISKANN_DLLEXPORT void generate_random_labels(std::vector<LabelT> &labels, const uint32_t num_labels,
                                                  const uint32_t nthreads);
    void reset_stream_for_reading(std::basic_istream<char> &infile);

    DISKANN_DLLEXPORT uint64_t get_node_sector(uint64_t node_id);
    DISKANN_DLLEXPORT char *offset_to_node(char *sector_buf, uint64_t node_id);
    DISKANN_DLLEXPORT uint32_t *offset_to_node_nhood(char *node_buf);
    DISKANN_DLLEXPORT T *offset_to_node_coords(char *node_buf);

    uint64_t _max_node_len = 0;
    uint64_t _nnodes_per_sector = 0;
    uint64_t _max_degree = 0;

    uint64_t _ndims_reorder_vecs = 0;
    uint64_t _reorder_data_start_sector = 0;
    uint64_t _nvecs_per_sector = 0;

    diskann::Metric metric = diskann::Metric::L2;
    float _max_base_norm = 0.0f;

    uint64_t _num_points = 0;
    uint64_t _num_frozen_points = 0;
    uint64_t _frozen_location = 0;
    uint64_t _data_dim = 0;
    uint64_t _aligned_dim = 0;
    uint64_t _disk_bytes_per_point = 0;

    std::string _disk_index_file;
    std::vector<std::pair<uint32_t, uint32_t>> _node_visit_counter;

    // ── VectorTailCache: PCS profiling state ─────────────────────────────────
    // Allocated in generate_pcs_cache_list(), used during profiling window.
    std::vector<NodePCSStats> _pcs_stats;       // per-node stats [0.._num_points)
    // VectorTailCache: per-query workload logger
    WorkloadLogger _workload_logger;
    bool _collect_pcs_stats = false;            // gate: only active during profiling
    float _pcs_tail_threshold_us = 0.0f;        // latency above which a query is "tail"
    // ── VectorTailCache: per-node read latency log ──────────────────────────
    bool _log_per_node_latency = false;  // set false after collection
    std::vector<std::pair<uint32_t,float>> _per_node_lat_log;
    mutable std::mutex _per_node_lat_mutex;
    // ─────────────────────────────────────────────────────────────────────────

    uint8_t *data = nullptr;
    uint64_t _n_chunks;
    FixedChunkPQTable _pq_table;

    std::shared_ptr<Distance<T>> _dist_cmp;
    std::shared_ptr<Distance<float>> _dist_cmp_float;

    bool _use_disk_index_pq = false;
    uint64_t _disk_pq_n_chunks = 0;
    FixedChunkPQTable _disk_pq_table;

    uint32_t *_medoids = nullptr;
    size_t _num_medoids;
    float *_centroid_data = nullptr;

    unsigned *_nhood_cache_buf = nullptr;
    tsl::robin_map<uint32_t, std::pair<uint32_t, uint32_t *>> _nhood_cache;

    T *_coord_cache_buf = nullptr;
    tsl::robin_map<uint32_t, T *> _coord_cache;

    ConcurrentQueue<SSDThreadData<T> *> _thread_data;
    uint64_t _max_nthreads;
    bool _load_flag = false;
    bool _count_visited_nodes = false;
    bool _reorder_data_exists = false;
    uint64_t _reoreder_data_offset = 0;

    uint32_t *_pts_to_label_offsets = nullptr;
    uint32_t *_pts_to_label_counts = nullptr;
    LabelT *_pts_to_labels = nullptr;
    std::unordered_map<LabelT, std::vector<uint32_t>> _filter_to_medoid_ids;
    bool _use_universal_label = false;
    LabelT _universal_filter_label;
    tsl::robin_set<uint32_t> _dummy_pts;
    tsl::robin_set<uint32_t> _has_dummy_pts;
    tsl::robin_map<uint32_t, uint32_t> _dummy_to_real_map;
    tsl::robin_map<uint32_t, std::vector<uint32_t>> _real_to_dummy_map;
    std::unordered_map<std::string, LabelT> _label_map;

#ifdef EXEC_ENV_OLS
    static const int HEADER_SIZE = defaults::SECTOR_LEN;
    char *getHeaderBytes();
#endif
};
} // namespace diskann

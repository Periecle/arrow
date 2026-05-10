// Reproducer for apache/arrow#45847.
//
// Reproduces the topology reported by uchenily:
//
//        aggregate
//            |
//          hashjoin
//          /      \
//   source_0     source_1
//   (probe)      (build)
//
// The reporter sees ~9000x slowdown and single-core utilization at large
// build_batch counts. This program runs the same topology with controllable
// dimensions and prints wall-clock timings and effective parallelism.
//
// Usage:
//   repro_45847 [build_batches] [probe_batches] [batch_size] [threads] [runs]
//
// Defaults match the issue: batch_size=1<<15, threads=hw, runs=3.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "arrow/acero/exec_plan.h"
#include "arrow/acero/options.h"
#include "arrow/api.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/initialize.h"
#include "arrow/util/thread_pool.h"

using arrow::Field;
using arrow::FieldRef;
using arrow::Int64Builder;
using arrow::Schema;
using arrow::Status;
using arrow::Table;
using arrow::TableBatchReader;
using arrow::compute::Aggregate;
using arrow::acero::AggregateNodeOptions;
using arrow::acero::Declaration;
using arrow::acero::DeclarationToTable;
using arrow::acero::HashJoinNodeOptions;
using arrow::acero::JoinKeyCmp;
using arrow::acero::JoinType;
using arrow::acero::QueryOptions;
using arrow::acero::TableSourceNodeOptions;

#define ABORT_NOT_OK(expr)                                  \
  do {                                                      \
    auto _st = (expr);                                      \
    if (!_st.ok()) {                                        \
      std::cerr << "FATAL: " << _st.ToString() << "\n";     \
      std::abort();                                         \
    }                                                       \
  } while (0)

namespace {

// Build a table of `num_batches * batch_size` rows with two int64 columns:
//   key:  start + i (each row gets a distinct key within [start, start+rows))
//   payload: i
// The returned Table is single-chunk; TableSourceNodeOptions will slice it
// into per-batch_size morsels.
arrow::Result<std::shared_ptr<Table>> MakeKeyedTable(int64_t num_batches,
                                                      int64_t batch_size,
                                                      int64_t key_start,
                                                      const std::string& key_name,
                                                      const std::string& payload_name) {
  const int64_t total_rows = num_batches * batch_size;
  Int64Builder key_b, pay_b;
  ARROW_RETURN_NOT_OK(key_b.Resize(total_rows));
  ARROW_RETURN_NOT_OK(pay_b.Resize(total_rows));
  for (int64_t i = 0; i < total_rows; ++i) {
    key_b.UnsafeAppend(key_start + i);
    pay_b.UnsafeAppend(i);
  }
  std::shared_ptr<arrow::Array> key_arr, pay_arr;
  ARROW_RETURN_NOT_OK(key_b.Finish(&key_arr));
  ARROW_RETURN_NOT_OK(pay_b.Finish(&pay_arr));
  auto schema = arrow::schema({arrow::field(key_name, arrow::int64()),
                               arrow::field(payload_name, arrow::int64())});
  return Table::Make(schema, {key_arr, pay_arr}, total_rows);
}

struct Args {
  int64_t build_batches = 32;
  int64_t probe_batches = 1;
  int64_t batch_size = 1 << 15;  // 32 768
  int threads = 0;               // 0 -> hardware
  int runs = 3;
  int num_aggs = 1;  // number of hash_count aggregations stacked on the same key
};

Args ParseArgs(int argc, char** argv) {
  Args a;
  if (argc > 1) a.build_batches = std::atoll(argv[1]);
  if (argc > 2) a.probe_batches = std::atoll(argv[2]);
  if (argc > 3) a.batch_size = std::atoll(argv[3]);
  if (argc > 4) a.threads = std::atoi(argv[4]);
  if (argc > 5) a.runs = std::atoi(argv[5]);
  if (argc > 6) a.num_aggs = std::atoi(argv[6]);
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  ABORT_NOT_OK(arrow::compute::Initialize());
  Args args = ParseArgs(argc, argv);
  if (args.threads > 0) {
    ABORT_NOT_OK(arrow::SetCpuThreadPoolCapacity(args.threads));
  }
  const int reported_threads =
      arrow::internal::GetCpuThreadPool()->GetCapacity();

  std::cout << "build_batches=" << args.build_batches
            << " probe_batches=" << args.probe_batches
            << " batch_size=" << args.batch_size
            << " threads=" << reported_threads
            << " runs=" << args.runs << "\n";

  // Both sides start at key 0, so probe rows match build rows 1:1 within the
  // overlap [0, min(build_rows, probe_rows)). This is intentionally cheap on
  // the join side so the cost we measure is the aggregator's.
  auto build_table = MakeKeyedTable(args.build_batches, args.batch_size,
                                    /*key_start=*/0, "rk", "rp")
                         .ValueOrDie();
  auto probe_table = MakeKeyedTable(args.probe_batches, args.batch_size,
                                    /*key_start=*/0, "lk", "lp")
                         .ValueOrDie();

  std::cout << "build_rows=" << build_table->num_rows()
            << " probe_rows=" << probe_table->num_rows() << "\n";

  for (int run = 0; run < args.runs; ++run) {
    Declaration probe_src{
        "table_source", TableSourceNodeOptions{probe_table, args.batch_size}};
    Declaration build_src{
        "table_source", TableSourceNodeOptions{build_table, args.batch_size}};

    HashJoinNodeOptions join_opts{JoinType::INNER,
                                  /*left_keys=*/{FieldRef("lk")},
                                  /*right_keys=*/{FieldRef("rk")}};
    Declaration join{"hashjoin",
                     {std::move(probe_src), std::move(build_src)},
                     std::move(join_opts)};

    std::vector<Aggregate> aggs;
    for (int i = 0; i < args.num_aggs; ++i) {
      aggs.push_back(Aggregate{"hash_count", nullptr, FieldRef("lk"),
                               "cnt" + std::to_string(i)});
    }
    AggregateNodeOptions agg_opts{std::move(aggs), /*keys=*/{FieldRef("lk")}};
    Declaration agg{"aggregate", {std::move(join)}, std::move(agg_opts)};

    auto t0 = std::chrono::steady_clock::now();
    auto table_or = DeclarationToTable(std::move(agg), /*use_threads=*/true);
    auto t1 = std::chrono::steady_clock::now();
    ABORT_NOT_OK(table_or.status());
    double secs = std::chrono::duration<double>(t1 - t0).count();
    auto table = table_or.ValueOrDie();

    // Validate: each row's cnt0 must equal 1 (unique keys, 1:1 inner join).
    // The aggregator runs Merge across all threads on every run; if
    // transposition or kernel-state ordering breaks, this will fail.
    int64_t sum_cnt = 0;
    auto cnt_chunked = table->GetColumnByName("cnt0");
    for (int c = 0; c < cnt_chunked->num_chunks(); ++c) {
      auto arr =
          std::static_pointer_cast<arrow::Int64Array>(cnt_chunked->chunk(c));
      for (int64_t i = 0; i < arr->length(); ++i) {
        if (arr->Value(i) != 1) {
          std::cerr << "FATAL: chunk=" << c << " row=" << i
                    << " cnt=" << arr->Value(i) << " (expected 1)\n";
          std::abort();
        }
        sum_cnt += arr->Value(i);
      }
    }
    const int64_t expected =
        std::min(build_table->num_rows(), probe_table->num_rows());
    if (sum_cnt != expected) {
      std::cerr << "FATAL: sum_cnt=" << sum_cnt << " expected=" << expected
                << "\n";
      std::abort();
    }
    std::cout << "run=" << run << " seconds=" << secs
              << " out_rows=" << table->num_rows() << " sum_cnt=" << sum_cnt
              << "\n";
  }
  return 0;
}

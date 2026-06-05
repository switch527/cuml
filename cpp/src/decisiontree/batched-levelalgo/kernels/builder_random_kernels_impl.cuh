/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "builder_kernels.cuh"

#include <common/grid_sync.cuh>

#include <raft/util/cuda_utils.cuh>

#include <type_traits>

namespace ML {
namespace DT {

// Translation-unit-local; each random_*.cu includes exactly one impl header
// so this does not collide with the same constant in builder_kernels_impl.cuh.
static constexpr int TPB_DEFAULT = 128;

// ExtraTrees random-split kernel: one threshold per (node, feature) via
// et_split_position; companion-buffer region collapses from max_n_bins to 2.
template <typename DataT,
          typename LabelT,
          typename IdxT,
          int TPB,
          typename ObjectiveT,
          typename BinT>
static __global__ void randomSplitKernel(BinT* histograms,
                                         int* unweighted_histograms,
                                         double* weighted_count_histograms,
                                         IdxT max_n_bins,
                                         IdxT min_samples_split,
                                         IdxT max_leaves,
                                         const Dataset<DataT, LabelT, IdxT> dataset,
                                         const Quantiles<DataT, IdxT> quantiles,
                                         const NodeWorkItem* work_items,
                                         IdxT colStart,
                                         const IdxT* colids,
                                         int* done_count,
                                         int* mutex,
                                         volatile Split<DataT, IdxT>* splits,
                                         ObjectiveT objective,
                                         IdxT treeid,
                                         const WorkloadInfo<IdxT>* workload_info,
                                         uint64_t seed)
{
  // Signature parity with launchComputeSplitKernel; min_samples_split /
  // max_leaves are enforced upstream (NodeQueue::Push), not inside the kernel.
  (void)min_samples_split;
  (void)max_leaves;
  extern __shared__ char smem[];

  WorkloadInfo<IdxT> workload_info_cta = workload_info[blockIdx.x];
  IdxT nid                             = workload_info_cta.nodeid;
  IdxT large_nid                       = workload_info_cta.large_nodeid;
  const auto work_item                 = work_items[nid];
  auto range_start                     = work_item.instances.begin;
  auto range_len                       = work_item.instances.count;

  IdxT offset_blockid = workload_info_cta.offset_blockid;
  IdxT num_blocks     = workload_info_cta.num_blocks;

  IdxT col;
  if (dataset.n_sampled_cols == dataset.N) {
    col = colStart + blockIdx.y;
  } else {
    IdxT colIndex = colStart + blockIdx.y;
    col           = colids[nid * dataset.n_sampled_cols + colIndex];
  }

  int n_bins = quantiles.n_bins_array[col];
  // Constant-feature guard. All blocks for a (node, feature) read the same
  // n_bins from quantiles.n_bins_array[col] so the uniform return is
  // deadlock-safe: no block is left waiting on a signalDone that never fires.
  if (n_bins < 2) { return; }

  constexpr bool kIsClassifier = std::is_same<BinT, CountBin>::value;
  constexpr bool kIsRegressor  = std::is_same<BinT, AggregateBin>::value;
  static_assert(kIsClassifier || kIsRegressor, "unknown BinT in randomSplitKernel");

  auto n_classes                      = objective.NumClasses();
  auto* shared_left                   = alignPointer<BinT>(smem);
  auto* shared_parent                 = alignPointer<BinT>(shared_left + n_classes);
  int* shared_companion_left_int      = nullptr;
  int* shared_companion_parent_int    = nullptr;
  double* shared_companion_left_dbl   = nullptr;
  double* shared_companion_parent_dbl = nullptr;
  int* shared_done;
  if constexpr (kIsClassifier) {
    shared_companion_left_int   = alignPointer<int>(shared_parent + n_classes);
    shared_companion_parent_int = alignPointer<int>(shared_companion_left_int + 1);
    shared_done                 = alignPointer<int>(shared_companion_parent_int + 1);
  } else {
    shared_companion_left_dbl   = alignPointer<double>(shared_parent + n_classes);
    shared_companion_parent_dbl = alignPointer<double>(shared_companion_left_dbl + 1);
    shared_done                 = alignPointer<int>(shared_companion_parent_dbl + 1);
  }

  // Initialize shared-memory cells.
  for (IdxT j = threadIdx.x; j < n_classes; j += blockDim.x) {
    shared_left[j]   = BinT();
    shared_parent[j] = BinT();
  }
  if (threadIdx.x == 0) {
    if constexpr (kIsClassifier) {
      shared_companion_left_int[0]   = 0;
      shared_companion_parent_int[0] = 0;
    } else {
      shared_companion_left_dbl[0]   = 0.0;
      shared_companion_parent_dbl[0] = 0.0;
    }
    *shared_done = 0;
  }
  __syncthreads();

  // Threshold drawn from the global quantile grid (not the per-node feature
  // range as sklearn's RandomSplitter does); min_samples_leaf rejects any
  // candidate that lands outside the local data range at deeper nodes.
  // n_bins >= 2 is guaranteed by the upstream validity_check + the local
  // constant-feature early-return above.
  IdxT split_pos =
    et_split_position<IdxT>(seed, uint64_t(treeid), uint64_t(work_item.idx), uint64_t(col), n_bins);
  DataT threshold = quantiles.quantiles_array[max_n_bins * col + split_pos];

  IdxT stride            = blockDim.x * num_blocks;
  IdxT tid               = threadIdx.x + offset_blockid * blockDim.x;
  std::size_t col_offset = std::size_t(col) * dataset.M;
  auto end               = range_start + range_len;
  bool has_weight        = (dataset.sample_weight != nullptr);

  for (auto i = range_start + tid; i < end; i += stride) {
    auto row = dataset.row_ids[i];
    // INVARIANT: row is a valid index into dataset.sample_weight; the
    // per_tree_weights silent-zero (per_tree_weights.cu) is unreachable here.
    auto data     = dataset.data[row + col_offset];
    auto label    = dataset.labels[row];
    double weight = has_weight ? static_cast<double>(dataset.sample_weight[row]) : 1.0;

    BinT::IncrementHistogram(shared_parent, /*n_bins=*/1, /*b=*/0, label, weight);
    if constexpr (kIsClassifier) {
      atomicAdd(&shared_companion_parent_int[0], 1);
    } else {
      atomicAdd(&shared_companion_parent_dbl[0], weight);
    }
    if (data <= threshold) {
      BinT::IncrementHistogram(shared_left, /*n_bins=*/1, /*b=*/0, label, weight);
      if constexpr (kIsClassifier) {
        atomicAdd(&shared_companion_left_int[0], 1);
      } else {
        atomicAdd(&shared_companion_left_dbl[0], weight);
      }
    }
  }

  __syncthreads();

  if (num_blocks > 1) {
    // Cross-block merge: BinT::AtomicAdd is non-associative under FP
    // contention, same trade-off as the SPLITTER_BEST histograms path.
    auto bin_offset       = ((large_nid * gridDim.y) + blockIdx.y) * IdxT(2) * n_classes;
    auto companion_offset = ((large_nid * gridDim.y) + blockIdx.y) * IdxT(2);

    for (IdxT j = threadIdx.x; j < n_classes; j += blockDim.x) {
      BinT::AtomicAdd(histograms + bin_offset + j, shared_left[j]);
      BinT::AtomicAdd(histograms + bin_offset + n_classes + j, shared_parent[j]);
    }
    if constexpr (kIsClassifier) {
      if (threadIdx.x == 0) {
        atomicAdd(&unweighted_histograms[companion_offset + 0], shared_companion_left_int[0]);
        atomicAdd(&unweighted_histograms[companion_offset + 1], shared_companion_parent_int[0]);
      }
    } else {
      if (threadIdx.x == 0) {
        atomicAdd(&weighted_count_histograms[companion_offset + 0], shared_companion_left_dbl[0]);
        atomicAdd(&weighted_count_histograms[companion_offset + 1], shared_companion_parent_dbl[0]);
      }
    }

    __threadfence();
    __syncthreads();

    bool last = MLCommon::signalDone(
      done_count + nid * gridDim.y + blockIdx.y, num_blocks, offset_blockid == 0, shared_done);
    if (!last) return;

    // Last block reads merged values back into smem for scoring.
    for (IdxT j = threadIdx.x; j < n_classes; j += blockDim.x) {
      shared_left[j]   = histograms[bin_offset + j];
      shared_parent[j] = histograms[bin_offset + n_classes + j];
    }
    if (threadIdx.x == 0) {
      if constexpr (kIsClassifier) {
        shared_companion_left_int[0]   = unweighted_histograms[companion_offset + 0];
        shared_companion_parent_int[0] = unweighted_histograms[companion_offset + 1];
      } else {
        shared_companion_left_dbl[0]   = weighted_count_histograms[companion_offset + 0];
        shared_companion_parent_dbl[0] = weighted_count_histograms[companion_offset + 1];
      }
    }
    __syncthreads();
  }

  // Thread 0 scores; other threads carry default Split() per split.cuh's
  // contract that all threads enter evalBestSplit unconditionally.
  Split<DataT, IdxT> sp;
  if (threadIdx.x == 0) {
    DataT gain;
    IdxT nLeft;
    if constexpr (kIsClassifier) {
      gain = objective.GainFromSideStats(
        shared_left, shared_parent, shared_companion_left_int[0], shared_companion_parent_int[0]);
      nLeft = static_cast<IdxT>(shared_companion_left_int[0]);
    } else {
      gain = objective.GainFromSideStats(
        shared_left, shared_parent, shared_companion_left_dbl[0], shared_companion_parent_dbl[0]);
      // Regressor: nLeft is the unweighted integer row count (AggregateBin.count
      // is incremented unconditionally per row); SplitNotValid checks it against
      // min_samples_leaf in the same integer space as sklearn.
      nLeft = static_cast<IdxT>(shared_left[0].count);
    }
    sp.update({threshold, col, gain, nLeft});
  }

  __syncthreads();
  sp.evalBestSplit(smem, splits + nid, mutex + nid);
}

template <typename DataT,
          typename LabelT,
          typename IdxT,
          int TPB,
          typename ObjectiveT,
          typename BinT>
void launchRandomSplitKernel(BinT* histograms,
                             int* unweighted_histograms,
                             double* weighted_count_histograms,
                             IdxT max_n_bins,
                             IdxT min_samples_split,
                             IdxT max_leaves,
                             const Dataset<DataT, LabelT, IdxT>& dataset,
                             const Quantiles<DataT, IdxT>& quantiles,
                             const NodeWorkItem* work_items,
                             IdxT colStart,
                             const IdxT* colids,
                             int* done_count,
                             int* mutex,
                             volatile Split<DataT, IdxT>* splits,
                             ObjectiveT& objective,
                             IdxT treeid,
                             const WorkloadInfo<IdxT>* workload_info,
                             uint64_t seed,
                             dim3 grid,
                             size_t smem_size,
                             cudaStream_t builder_stream)
{
  randomSplitKernel<DataT, LabelT, IdxT, TPB, ObjectiveT, BinT>
    <<<grid, TPB, smem_size, builder_stream>>>(histograms,
                                               unweighted_histograms,
                                               weighted_count_histograms,
                                               max_n_bins,
                                               min_samples_split,
                                               max_leaves,
                                               dataset,
                                               quantiles,
                                               work_items,
                                               colStart,
                                               colids,
                                               done_count,
                                               mutex,
                                               splits,
                                               objective,
                                               treeid,
                                               workload_info,
                                               seed);
}

// Explicit instantiation; each random_<objective>-{float,double}.cu
// translation unit defines the _DataT / _LabelT / _IdxT / _ObjectiveT / _BinT
// aliases and includes this header.
template void launchRandomSplitKernel<_DataT, _LabelT, _IdxT, TPB_DEFAULT, _ObjectiveT, _BinT>(
  _BinT* histograms,
  int* unweighted_histograms,
  double* weighted_count_histograms,
  _IdxT max_n_bins,
  _IdxT min_samples_split,
  _IdxT max_leaves,
  const Dataset<_DataT, _LabelT, _IdxT>& dataset,
  const Quantiles<_DataT, _IdxT>& quantiles,
  const NodeWorkItem* work_items,
  _IdxT colStart,
  const _IdxT* colids,
  int* done_count,
  int* mutex,
  volatile Split<_DataT, _IdxT>* splits,
  _ObjectiveT& objective,
  _IdxT treeid,
  const WorkloadInfo<_IdxT>* workload_info,
  uint64_t seed,
  dim3 grid,
  size_t smem_size,
  cudaStream_t builder_stream);

}  // namespace DT
}  // namespace ML

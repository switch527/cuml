/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// ExtraTrees oracle gtests: et_split_position bin-draw byte-identity +
// uniformity + pairwise-column independence, and Gini/Entropy
// GainFromSideStats hand-derived expected values.

#include "extratrees_rng_reference.h"

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <decisiontree/batched-levelalgo/bins.cuh>
#include <decisiontree/batched-levelalgo/kernels/builder_kernels.cuh>
#include <decisiontree/batched-levelalgo/objectives.cuh>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace ML {
namespace DT {

// -----------------------------------------------------------------------------
// Oracle 1: et_split_position RNG byte-identity vs the Python-generated table.
// -----------------------------------------------------------------------------

__global__ void RunEtSplitPositionKernel(EtRngRefEntry const* table,
                                         int* out,
                                         int n_bins,
                                         int count)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  auto const& e = table[i];
  out[i]        = et_split_position<int>(e.seed, e.treeid, e.nodeid, e.col, n_bins);
}

TEST(ExtraTreesTests, EtSplitPositionRngByteIdentity)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  rmm::device_uvector<EtRngRefEntry> d_table(kEtRngRefCount, stream);
  rmm::device_uvector<int> d_out(kEtRngRefCount, stream);
  RAFT_CUDA_TRY(cudaMemcpyAsync(d_table.data(),
                                kEtRngRefTable,
                                sizeof(EtRngRefEntry) * kEtRngRefCount,
                                cudaMemcpyHostToDevice,
                                stream));

  int const tpb = 64;
  int const nb  = (kEtRngRefCount + tpb - 1) / tpb;
  RunEtSplitPositionKernel<<<nb, tpb, 0, stream>>>(
    d_table.data(), d_out.data(), kEtRngRefMaxNBins, kEtRngRefCount);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  std::vector<int> h_out(kEtRngRefCount);
  raft::update_host(h_out.data(), d_out.data(), kEtRngRefCount, stream);
  handle.sync_stream(stream);

  for (int i = 0; i < kEtRngRefCount; ++i) {
    auto const& e = kEtRngRefTable[i];
    EXPECT_EQ(h_out[i], e.expected)
      << "tuple (seed=" << e.seed << ", treeid=" << e.treeid << ", nodeid=" << e.nodeid
      << ", col=" << e.col << ") at index " << i;
  }
}

// -----------------------------------------------------------------------------
// Oracle 2: Monte-Carlo chi-square uniformity over [0, n_bins - 1).
// -----------------------------------------------------------------------------

__global__ void RunEtSplitPositionDrawKernel(
  int* out, int n, int n_bins, uint64_t seed, uint64_t treeid, uint64_t nodeid_base)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // Vary both nodeid and col across the N draws so a per-axis correlation
  // cannot bias the chi-square histogram (XOR with golden ratio decorrelates).
  uint64_t nodeid = nodeid_base + uint64_t(i);
  uint64_t col    = uint64_t(i) ^ uint64_t(0x9E3779B97F4A7C15ull);
  out[i]          = et_split_position<int>(seed, treeid, nodeid, col, n_bins);
}

TEST(ExtraTreesTests, EtSplitPositionMonteCarloUniformity)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  constexpr int N      = 10000;
  constexpr int n_bins = 16;
  constexpr int K      = n_bins - 1;  // half-open [0, n_bins-1) -> K buckets

  rmm::device_uvector<int> d_out(N, stream);
  int const tpb = 128;
  int const nb  = (N + tpb - 1) / tpb;
  RunEtSplitPositionDrawKernel<<<nb, tpb, 0, stream>>>(
    d_out.data(), N, n_bins, /*seed=*/42ull, /*treeid=*/7ull, /*nodeid_base=*/0ull);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  std::vector<int> h(N);
  raft::update_host(h.data(), d_out.data(), N, stream);
  handle.sync_stream(stream);

  std::vector<int> counts(K, 0);
  for (int v : h) {
    ASSERT_GE(v, 0);
    ASSERT_LT(v, K);
    ++counts[v];
  }

  // Chi-square statistic against expected uniform N/K per bucket.
  double const expected = double(N) / double(K);
  double chi2_stat      = 0.0;
  for (int c : counts) {
    double d = double(c) - expected;
    chi2_stat += d * d / expected;
  }
  // df = K - 1 = 14. chi2.ppf(0.999, 14) = 36.123; chi2.ppf(0.9995, 14) = 38.109.
  // Use 40.0 as a further-conservative band to absorb single-run flake risk.
  double const critical = 40.0;
  EXPECT_LT(chi2_stat, critical)
    << "chi-square statistic " << chi2_stat << " exceeded critical value " << critical
    << " (df=14, alpha=0.0005); the et_split_position draw appears non-uniform";
}

// -----------------------------------------------------------------------------
// Oracle 3: Pairwise-column independence via Spearman's rho.
// -----------------------------------------------------------------------------

__global__ void RunEtSplitPositionPairKernel(
  int* out_a, int* out_b, int n, int n_bins, uint64_t seed, uint64_t treeid)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // For each i, draw two adjacent columns (col=0, col=1) at the same
  // (seed, treeid, nodeid). If the chain is well-mixed they should be
  // pairwise independent.
  uint64_t nodeid = uint64_t(i);
  out_a[i]        = et_split_position<int>(seed, treeid, nodeid, /*col=*/0ull, n_bins);
  out_b[i]        = et_split_position<int>(seed, treeid, nodeid, /*col=*/1ull, n_bins);
}

static double spearman_rho(std::vector<int> const& a, std::vector<int> const& b)
{
  // For integer-valued samples on [0, K), rank-average within ties via a
  // count-frequency table. Spearman's rho on ranks reduces to Pearson on
  // the rank pairs.
  auto rank_of = [](std::vector<int> const& v) {
    int const N = static_cast<int>(v.size());
    // Sort indices by value, breaking ties by index (stable) for the
    // average-rank computation.
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int x, int y) { return v[x] < v[y]; });
    std::vector<double> r(N);
    int i = 0;
    while (i < N) {
      int j = i;
      while (j < N && v[idx[j]] == v[idx[i]])
        ++j;
      double avg = double(i + j - 1) / 2.0 + 1.0;  // 1-based average rank
      for (int k = i; k < j; ++k)
        r[idx[k]] = avg;
      i = j;
    }
    return r;
  };

  auto ra        = rank_of(a);
  auto rb        = rank_of(b);
  double const N = double(a.size());
  double mean_a  = (N + 1.0) / 2.0;
  double mean_b  = mean_a;
  double num     = 0.0;
  double da2     = 0.0;
  double db2     = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double xa = ra[i] - mean_a;
    double xb = rb[i] - mean_b;
    num += xa * xb;
    da2 += xa * xa;
    db2 += xb * xb;
  }
  return num / std::sqrt(da2 * db2);
}

TEST(ExtraTreesTests, EtSplitPositionSpearmanPairCorrelation)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  constexpr int N      = 10000;
  constexpr int n_bins = 16;

  rmm::device_uvector<int> d_a(N, stream);
  rmm::device_uvector<int> d_b(N, stream);
  int const tpb = 128;
  int const nb  = (N + tpb - 1) / tpb;
  RunEtSplitPositionPairKernel<<<nb, tpb, 0, stream>>>(
    d_a.data(), d_b.data(), N, n_bins, /*seed=*/42ull, /*treeid=*/7ull);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  std::vector<int> ha(N), hb(N);
  raft::update_host(ha.data(), d_a.data(), N, stream);
  raft::update_host(hb.data(), d_b.data(), N, stream);
  handle.sync_stream(stream);

  double const rho = spearman_rho(ha, hb);
  // Under H0 (independence) and N=10000, |rho| should sit well within
  // ~0.04 by the 1/sqrt(N) heuristic. The 0.05 threshold matches the plan.
  EXPECT_LT(std::abs(rho), 0.05)
    << "Spearman rho between col=0 and col=1 draws (same seed, treeid, nodeid) "
       "exceeded 0.05; the per-column fnv1a32 mix may be underdiffusing";
}

// Gini / Entropy ::GainFromSideStats hand-derived references. The helpers
// are HDI so we call them on host directly, matching the precedent in
// rf_test.cu (e.g. MSEWeightedGainPerSplitGroundTruth at :894).

// Hand-derived weighted-Gini reference. Mirrors the helper body exactly so
// the test pins the formula, not the entire stack: any drift in the helper
// surfaces here, but a refactor that preserves the formula stays green.
template <typename DataT>
static DataT WeightedGiniReference(std::vector<double> const& left_x,
                                   std::vector<double> const& parent_x)
{
  double W_total = std::accumulate(parent_x.begin(), parent_x.end(), 0.0);
  double W_left  = std::accumulate(left_x.begin(), left_x.end(), 0.0);
  double W_right = W_total - W_left;
  double invW    = 1.0 / W_total;
  double invWL   = 1.0 / W_left;
  double invWR   = 1.0 / W_right;
  double gain    = 0.0;
  for (size_t j = 0; j < parent_x.size(); ++j) {
    double lval = left_x[j];
    double tot  = parent_x[j];
    double rval = tot - lval;
    gain += lval * invWL * lval * invW;
    gain += rval * invWR * rval * invW;
    double val = tot * invW;
    gain -= val * val;
  }
  return DataT(gain);
}

// Hand-derived weighted-Entropy reference. Uses base-2 log to match the
// device helper.
template <typename DataT>
static DataT WeightedEntropyReference(std::vector<double> const& left_x,
                                      std::vector<double> const& parent_x)
{
  double W_total    = std::accumulate(parent_x.begin(), parent_x.end(), 0.0);
  double W_left     = std::accumulate(left_x.begin(), left_x.end(), 0.0);
  double W_right    = W_total - W_left;
  double invW       = 1.0 / W_total;
  double invWL      = 1.0 / W_left;
  double invWR      = 1.0 / W_right;
  double const log2 = std::log(2.0);
  double gain       = 0.0;
  for (size_t c = 0; c < parent_x.size(); ++c) {
    double lval = left_x[c];
    double tot  = parent_x[c];
    double rval = tot - lval;
    if (lval > 0.0) gain += std::log(lval * invWL) / log2 * lval * invW;
    if (rval > 0.0) gain += std::log(rval * invWR) / log2 * rval * invW;
    if (tot > 0.0) gain -= std::log(tot * invW) / log2 * tot * invW;
  }
  return DataT(gain);
}

// Tie-free per-class histogram fixtures. Per-class entries are integer
// weighted samples, so n_left/n_parent are the row sums.
struct GainFixture {
  std::vector<double> left;
  std::vector<double> parent;
  int n_left;
  int n_parent;
};

static std::vector<GainFixture> const kGainFixtures = {
  // Balanced left split, 3 classes, integer-equivalent weights.
  {{2.0, 4.0, 1.0}, {6.0, 5.0, 3.0}, 7, 14},
  // Skewed-by-weight, 3 classes. The weighted sums and integer counts
  // diverge so the helper's distinct n vs W tracking matters.
  {{0.5, 0.0, 2.5}, {1.5, 1.0, 4.0}, 4, 9},
  // 4 classes, parent split such that one class is left-only.
  {{3.0, 0.0, 0.0, 1.0}, {4.0, 2.0, 1.0, 3.0}, 4, 12},
};

// Build {left, parent} CountBin vectors from the fixture's class-weighted-x
// arrays. Shared across the four typed oracle TESTs below.
template <typename BinT>
static std::pair<std::vector<BinT>, std::vector<BinT>> BuildBins(GainFixture const& fx)
{
  std::vector<BinT> left(fx.left.size());
  std::vector<BinT> parent(fx.parent.size());
  for (size_t i = 0; i < fx.left.size(); ++i) {
    left[i].x   = fx.left[i];
    parent[i].x = fx.parent[i];
  }
  return {left, parent};
}

TEST(ExtraTreesTests, GiniGainFromSideStatsOracleFloat)
{
  using ObjT = GiniObjectiveFunction<float, int, int>;
  using BinT = ObjT::BinT;
  for (auto const& fx : kGainFixtures) {
    auto [left, parent] = BuildBins<BinT>(fx);
    ObjT obj(static_cast<int>(fx.parent.size()), /*min_samples_leaf=*/1);
    float const expected = WeightedGiniReference<float>(fx.left, fx.parent);
    float const got = obj.GainFromSideStats(left.data(), parent.data(), fx.n_left, fx.n_parent);
    EXPECT_NEAR(got, expected, 1e-5f);
  }
}

TEST(ExtraTreesTests, GiniGainFromSideStatsOracleDouble)
{
  using ObjT = GiniObjectiveFunction<double, int, int>;
  using BinT = ObjT::BinT;
  for (auto const& fx : kGainFixtures) {
    auto [left, parent] = BuildBins<BinT>(fx);
    ObjT obj(static_cast<int>(fx.parent.size()), /*min_samples_leaf=*/1);
    double const expected = WeightedGiniReference<double>(fx.left, fx.parent);
    double const got = obj.GainFromSideStats(left.data(), parent.data(), fx.n_left, fx.n_parent);
    EXPECT_NEAR(got, expected, 1e-12);
  }
}

TEST(ExtraTreesTests, EntropyGainFromSideStatsOracleFloat)
{
  using ObjT = EntropyObjectiveFunction<float, int, int>;
  using BinT = ObjT::BinT;
  for (auto const& fx : kGainFixtures) {
    auto [left, parent] = BuildBins<BinT>(fx);
    ObjT obj(static_cast<int>(fx.parent.size()), /*min_samples_leaf=*/1);
    float const expected = WeightedEntropyReference<float>(fx.left, fx.parent);
    float const got = obj.GainFromSideStats(left.data(), parent.data(), fx.n_left, fx.n_parent);
    EXPECT_NEAR(got, expected, 1e-5f);
  }
}

TEST(ExtraTreesTests, EntropyGainFromSideStatsOracleDouble)
{
  using ObjT = EntropyObjectiveFunction<double, int, int>;
  using BinT = ObjT::BinT;
  for (auto const& fx : kGainFixtures) {
    auto [left, parent] = BuildBins<BinT>(fx);
    ObjT obj(static_cast<int>(fx.parent.size()), /*min_samples_leaf=*/1);
    double const expected = WeightedEntropyReference<double>(fx.left, fx.parent);
    double const got = obj.GainFromSideStats(left.data(), parent.data(), fx.n_left, fx.n_parent);
    EXPECT_NEAR(got, expected, 1e-12);
  }
}

// min_samples_leaf gate: helper returns -max() under the threshold. Both
// fail and floor-pass cases so a flipped comparison is caught.
TEST(ExtraTreesTests, GainFromSideStatsRespectsMinSamplesLeaf)
{
  using ObjT          = GiniObjectiveFunction<double, int, int>;
  using BinT          = ObjT::BinT;
  auto const& fx      = kGainFixtures.front();
  auto [left, parent] = BuildBins<BinT>(fx);
  // fx.n_left = 7, fx.n_parent = 14, so n_right = 7.
  // Fail case: min_samples_leaf = 8 > min(n_left, n_right) = 7.
  ObjT fail_obj(static_cast<int>(fx.parent.size()), /*min_samples_leaf=*/8);
  double const fail =
    fail_obj.GainFromSideStats(left.data(), parent.data(), fx.n_left, fx.n_parent);
  EXPECT_EQ(fail, -std::numeric_limits<double>::max());
  // Floor-pass: min_samples_leaf = 7 exactly meets the gate on both sides.
  // Returns a finite, positive gain.
  ObjT floor_obj(static_cast<int>(fx.parent.size()), /*min_samples_leaf=*/7);
  double const floor =
    floor_obj.GainFromSideStats(left.data(), parent.data(), fx.n_left, fx.n_parent);
  EXPECT_GT(floor, 0.0);
  EXPECT_LT(floor, std::numeric_limits<double>::max());
}

// Regressor ::GainFromSideStats hand-derived references. AggregateBin carries
// {label_sum, count}; the adapter synthesizes a 2-bin histogram and reuses
// the per-objective GainPerSplit body at i=0.
struct RegFixture {
  double L_left;
  double L_parent;
  double W_left;
  double W_parent;
  int n_left;
  int n_parent;
};

static std::vector<RegFixture> const kRegFixtures = {
  // Balanced labels, balanced weights, integer-equivalent counts.
  {3.0, 8.0, 2.0, 4.0, 2, 4},
  // Skewed weights vs counts so the W vs n separation matters.
  {2.5, 6.0, 1.5, 4.0, 3, 7},
  // Per-side means 6/3=2.0 (left), 6/5=1.2 (right), 12/8=1.5 (parent);
  // values are not a scalar multiple of fixture 1 so each objective
  // evaluates a distinct gain value here.
  {6.0, 12.0, 3.0, 8.0, 4, 10},
};

template <typename BinT>
static std::pair<std::vector<BinT>, std::vector<BinT>> BuildRegBins(RegFixture const& fx)
{
  std::vector<BinT> left(1);
  std::vector<BinT> parent(1);
  left[0].label_sum   = fx.L_left;
  left[0].count       = fx.n_left;
  parent[0].label_sum = fx.L_parent;
  parent[0].count     = fx.n_parent;
  return {left, parent};
}

template <typename DataT>
static DataT MSEReference(RegFixture const& fx)
{
  double L_R        = fx.L_parent - fx.L_left;
  double W_right    = fx.W_parent - fx.W_left;
  double parent_obj = -fx.L_parent * fx.L_parent / fx.W_parent;
  double left_obj   = -fx.L_left * fx.L_left / fx.W_left;
  double right_obj  = -L_R * L_R / W_right;
  double gain       = (parent_obj - (left_obj + right_obj)) * 0.5 / fx.W_parent;
  return DataT(gain);
}

template <typename DataT>
static DataT PoissonReference(RegFixture const& fx)
{
  double L_R        = fx.L_parent - fx.L_left;
  double W_right    = fx.W_parent - fx.W_left;
  double parent_obj = -fx.L_parent * std::log(fx.L_parent / fx.W_parent);
  double left_obj   = -fx.L_left * std::log(fx.L_left / fx.W_left);
  double right_obj  = -L_R * std::log(L_R / W_right);
  double gain       = (parent_obj - (left_obj + right_obj)) / fx.W_parent;
  return DataT(gain);
}

template <typename DataT>
static DataT GammaReference(RegFixture const& fx)
{
  double L_R        = fx.L_parent - fx.L_left;
  double W_right    = fx.W_parent - fx.W_left;
  double parent_obj = fx.W_parent * std::log(fx.L_parent / fx.W_parent);
  double left_obj   = fx.W_left * std::log(fx.L_left / fx.W_left);
  double right_obj  = W_right * std::log(L_R / W_right);
  double gain       = (parent_obj - (left_obj + right_obj)) / fx.W_parent;
  return DataT(gain);
}

template <typename DataT>
static DataT InverseGaussianReference(RegFixture const& fx)
{
  double L_R        = fx.L_parent - fx.L_left;
  double W_right    = fx.W_parent - fx.W_left;
  double parent_obj = -fx.W_parent * fx.W_parent / fx.L_parent;
  double left_obj   = -fx.W_left * fx.W_left / fx.L_left;
  double right_obj  = -W_right * W_right / L_R;
  double gain       = (parent_obj - (left_obj + right_obj)) / (2.0 * fx.W_parent);
  return DataT(gain);
}

TEST(ExtraTreesTests, MSEGainFromSideStatsOracleFloat)
{
  using ObjT = MSEObjectiveFunction<float, float, int>;
  using BinT = ObjT::BinT;
  for (auto const& fx : kRegFixtures) {
    auto [left, parent] = BuildRegBins<BinT>(fx);
    ObjT obj(/*nclasses=*/1, /*min_samples_leaf=*/1);
    float const expected = MSEReference<float>(fx);
    float const got = obj.GainFromSideStats(left.data(), parent.data(), fx.W_left, fx.W_parent);
    EXPECT_NEAR(got, expected, 1e-5f);
  }
}

TEST(ExtraTreesTests, MSEGainFromSideStatsOracleDouble)
{
  using ObjT = MSEObjectiveFunction<double, double, int>;
  using BinT = ObjT::BinT;
  for (auto const& fx : kRegFixtures) {
    auto [left, parent] = BuildRegBins<BinT>(fx);
    ObjT obj(/*nclasses=*/1, /*min_samples_leaf=*/1);
    double const expected = MSEReference<double>(fx);
    double const got = obj.GainFromSideStats(left.data(), parent.data(), fx.W_left, fx.W_parent);
    EXPECT_NEAR(got, expected, 1e-12);
  }
}

TEST(ExtraTreesTests, PoissonGainFromSideStatsOracleFloat)
{
  using ObjT = PoissonObjectiveFunction<float, float, int>;
  using BinT = ObjT::BinT;
  for (auto const& fx : kRegFixtures) {
    auto [left, parent] = BuildRegBins<BinT>(fx);
    ObjT obj(/*nclasses=*/1, /*min_samples_leaf=*/1);
    float const expected = PoissonReference<float>(fx);
    float const got = obj.GainFromSideStats(left.data(), parent.data(), fx.W_left, fx.W_parent);
    EXPECT_NEAR(got, expected, 1e-5f);
  }
}

TEST(ExtraTreesTests, PoissonGainFromSideStatsOracleDouble)
{
  using ObjT = PoissonObjectiveFunction<double, double, int>;
  using BinT = ObjT::BinT;
  for (auto const& fx : kRegFixtures) {
    auto [left, parent] = BuildRegBins<BinT>(fx);
    ObjT obj(/*nclasses=*/1, /*min_samples_leaf=*/1);
    double const expected = PoissonReference<double>(fx);
    double const got = obj.GainFromSideStats(left.data(), parent.data(), fx.W_left, fx.W_parent);
    EXPECT_NEAR(got, expected, 1e-12);
  }
}

TEST(ExtraTreesTests, GammaGainFromSideStatsOracleFloat)
{
  using ObjT = GammaObjectiveFunction<float, float, int>;
  using BinT = ObjT::BinT;
  for (auto const& fx : kRegFixtures) {
    auto [left, parent] = BuildRegBins<BinT>(fx);
    ObjT obj(/*nclasses=*/1, /*min_samples_leaf=*/1);
    float const expected = GammaReference<float>(fx);
    float const got = obj.GainFromSideStats(left.data(), parent.data(), fx.W_left, fx.W_parent);
    EXPECT_NEAR(got, expected, 1e-5f);
  }
}

TEST(ExtraTreesTests, GammaGainFromSideStatsOracleDouble)
{
  using ObjT = GammaObjectiveFunction<double, double, int>;
  using BinT = ObjT::BinT;
  for (auto const& fx : kRegFixtures) {
    auto [left, parent] = BuildRegBins<BinT>(fx);
    ObjT obj(/*nclasses=*/1, /*min_samples_leaf=*/1);
    double const expected = GammaReference<double>(fx);
    double const got = obj.GainFromSideStats(left.data(), parent.data(), fx.W_left, fx.W_parent);
    EXPECT_NEAR(got, expected, 1e-12);
  }
}

TEST(ExtraTreesTests, InverseGaussianGainFromSideStatsOracleFloat)
{
  using ObjT = InverseGaussianObjectiveFunction<float, float, int>;
  using BinT = ObjT::BinT;
  for (auto const& fx : kRegFixtures) {
    auto [left, parent] = BuildRegBins<BinT>(fx);
    ObjT obj(/*nclasses=*/1, /*min_samples_leaf=*/1);
    float const expected = InverseGaussianReference<float>(fx);
    float const got = obj.GainFromSideStats(left.data(), parent.data(), fx.W_left, fx.W_parent);
    EXPECT_NEAR(got, expected, 1e-5f);
  }
}

TEST(ExtraTreesTests, InverseGaussianGainFromSideStatsOracleDouble)
{
  using ObjT = InverseGaussianObjectiveFunction<double, double, int>;
  using BinT = ObjT::BinT;
  for (auto const& fx : kRegFixtures) {
    auto [left, parent] = BuildRegBins<BinT>(fx);
    ObjT obj(/*nclasses=*/1, /*min_samples_leaf=*/1);
    double const expected = InverseGaussianReference<double>(fx);
    double const got = obj.GainFromSideStats(left.data(), parent.data(), fx.W_left, fx.W_parent);
    EXPECT_NEAR(got, expected, 1e-12);
  }
}

// MSE min_samples_leaf gate. The same gate fires on all 4 regressor
// objectives via shared GainPerSplit shape; MSE is sufficient to pin it.
TEST(ExtraTreesTests, MSEGainFromSideStatsRespectsMinSamplesLeaf)
{
  using ObjT          = MSEObjectiveFunction<double, double, int>;
  using BinT          = ObjT::BinT;
  auto const& fx      = kRegFixtures.front();  // n_left=2, n_parent=4 → n_right=2
  auto [left, parent] = BuildRegBins<BinT>(fx);
  ObjT fail_obj(/*nclasses=*/1, /*min_samples_leaf=*/3);
  double const fail =
    fail_obj.GainFromSideStats(left.data(), parent.data(), fx.W_left, fx.W_parent);
  EXPECT_EQ(fail, -std::numeric_limits<double>::max());
  ObjT floor_obj(/*nclasses=*/1, /*min_samples_leaf=*/2);
  double const floor =
    floor_obj.GainFromSideStats(left.data(), parent.data(), fx.W_left, fx.W_parent);
  // MSE proxy gain is non-negative; tighter than > -max so a sign-flipped
  // formula that survives the gate would surface here.
  EXPECT_GT(floor, 0.0);
  EXPECT_LT(floor, std::numeric_limits<double>::max());
}

TEST(ExtraTreesTests, PoissonGainFromSideStatsRespectsMinSamplesLeaf)
{
  using ObjT          = PoissonObjectiveFunction<double, double, int>;
  using BinT          = ObjT::BinT;
  auto const& fx      = kRegFixtures.front();
  auto [left, parent] = BuildRegBins<BinT>(fx);
  ObjT fail_obj(/*nclasses=*/1, /*min_samples_leaf=*/3);
  double const fail =
    fail_obj.GainFromSideStats(left.data(), parent.data(), fx.W_left, fx.W_parent);
  EXPECT_EQ(fail, -std::numeric_limits<double>::max());
  ObjT floor_obj(/*nclasses=*/1, /*min_samples_leaf=*/2);
  double const floor =
    floor_obj.GainFromSideStats(left.data(), parent.data(), fx.W_left, fx.W_parent);
  EXPECT_GT(floor, -std::numeric_limits<double>::max());
  EXPECT_LT(floor, std::numeric_limits<double>::max());
}

TEST(ExtraTreesTests, GammaGainFromSideStatsRespectsMinSamplesLeaf)
{
  using ObjT          = GammaObjectiveFunction<double, double, int>;
  using BinT          = ObjT::BinT;
  auto const& fx      = kRegFixtures.front();
  auto [left, parent] = BuildRegBins<BinT>(fx);
  ObjT fail_obj(/*nclasses=*/1, /*min_samples_leaf=*/3);
  double const fail =
    fail_obj.GainFromSideStats(left.data(), parent.data(), fx.W_left, fx.W_parent);
  EXPECT_EQ(fail, -std::numeric_limits<double>::max());
  ObjT floor_obj(/*nclasses=*/1, /*min_samples_leaf=*/2);
  double const floor =
    floor_obj.GainFromSideStats(left.data(), parent.data(), fx.W_left, fx.W_parent);
  EXPECT_GT(floor, -std::numeric_limits<double>::max());
  EXPECT_LT(floor, std::numeric_limits<double>::max());
}

TEST(ExtraTreesTests, InverseGaussianGainFromSideStatsRespectsMinSamplesLeaf)
{
  using ObjT          = InverseGaussianObjectiveFunction<double, double, int>;
  using BinT          = ObjT::BinT;
  auto const& fx      = kRegFixtures.front();
  auto [left, parent] = BuildRegBins<BinT>(fx);
  ObjT fail_obj(/*nclasses=*/1, /*min_samples_leaf=*/3);
  double const fail =
    fail_obj.GainFromSideStats(left.data(), parent.data(), fx.W_left, fx.W_parent);
  EXPECT_EQ(fail, -std::numeric_limits<double>::max());
  ObjT floor_obj(/*nclasses=*/1, /*min_samples_leaf=*/2);
  double const floor =
    floor_obj.GainFromSideStats(left.data(), parent.data(), fx.W_left, fx.W_parent);
  EXPECT_GT(floor, -std::numeric_limits<double>::max());
  EXPECT_LT(floor, std::numeric_limits<double>::max());
}

}  // namespace DT
}  // namespace ML

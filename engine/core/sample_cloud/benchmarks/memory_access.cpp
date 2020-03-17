/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <memory>
#include <random>

#include "benchmark/benchmark.h"
#include "cuda_runtime.h"
#include "engine/core/sample_cloud/benchmarks/memory_access.cu.hpp"
#include "engine/core/sample_cloud/sample_cloud.hpp"
#include "engine/gems/sample_cloud/utils.hpp"

namespace isaac {

// Helper function generate interleaved data.
template <size_t Channels>
SampleCloud<float, Channels> RandomInterleavedPointData(size_t num_points) {
  SampleCloud<float, Channels> out(num_points);
  uint32_t seed;
  for (size_t index = 0; index < num_points; index++) {
    for (size_t i = 0; i < Channels; i++) {
      out[index][i] = rand_r(&seed);
    }
  }
  return out;
}

// Test reading interleaved memory and using the full data channel
template <size_t Channels>
void InterleavedMemoryFullAccess(benchmark::State& state) {
  size_t num_points = state.range(0);
  SampleCloud<float, Channels> input = RandomInterleavedPointData<Channels>(num_points);
  Eigen::Matrix<float, Channels, 1> out;
  for (auto _ : state) {
    for (size_t i = 3; i < num_points - 3; ++i) {
      out += input[i - 3] + input[i - 2] + input[i - 1] + input[i] + input[i + 1] + input[i + 2] +
             input[i + 3];
    }
  }
  benchmark::DoNotOptimize(out);
}

// Test Single channel access for interleaved memory
template <size_t Channels>
void InterleavedMemorySingleChannelAccess(benchmark::State& state) {
  size_t num_points = 0;
  SampleCloud<float, Channels> input;
  float out = 0.0f;
  num_points = state.range(0);
  input = RandomInterleavedPointData<Channels>(num_points);
  for (auto _ : state) {
    for (size_t i = 3; i < num_points - 3; ++i) {
      out += input[i - 3][0] + input[i - 2][0] + input[i - 1][0] + input[i][0] + input[i + 1][0] +
             input[i + 2][0] + input[i + 3][0];
    }
  }
  benchmark::DoNotOptimize(out);
}

// Test reading interleaved memory and using the full data channel
template <size_t Channels>
void InterleavedMemoryFullRandomAccess(benchmark::State& state) {
  size_t num_points = state.range(0);
  std::random_device rd;
  std::mt19937 eng(rd());
  std::uniform_int_distribution<> distr(3, num_points - 3);
  SampleCloud<float, Channels> input = RandomInterleavedPointData<Channels>(num_points);
  Eigen::Matrix<float, Channels, 1> out;
  for (auto _ : state) {
    for (size_t j = 3; j < num_points - 3; ++j) {
      size_t i = distr(eng);
      out += input[i - 3] + input[i - 2] + input[i - 1] + input[i] + input[i + 1] + input[i + 2] +
             input[i + 3];
    }
  }
  benchmark::DoNotOptimize(out);
}

// Test Single channel access for interleaved memory
template <size_t Channels>
void InterleavedMemorySingleChannelRandomAccess(benchmark::State& state) {
  size_t num_points = state.range(0);
  std::random_device rd;
  std::mt19937 eng(rd());
  std::uniform_int_distribution<> distr(3, num_points - 3);
  SampleCloud<float, Channels> input;
  float out = 0.0f;
  input = RandomInterleavedPointData<Channels>(num_points);
  for (auto _ : state) {
    for (size_t j = 3; j < num_points - 3; ++j) {
      size_t i = distr(eng);
      out += input[i - 3][0] + input[i - 2][0] + input[i - 1][0] + input[i][0] + input[i + 1][0] +
             input[i + 2][0] + input[i + 3][0];
    }
  }
  benchmark::DoNotOptimize(out);
}

// Test cuda full channel access for interleaved memory.
template <size_t Channels>
void CudaInterleavedMemoryFullAccess(benchmark::State& state) {
  size_t num_points = state.range(0);
  SampleCloud<float, Channels> input;
  CudaSampleCloud<float, Channels> cuda_input(num_points);
  input = RandomInterleavedPointData<Channels>(num_points);
  Copy(input, cuda_input);
  for (auto _ : state) {
    float ms = WindowedSumInterleaved(reinterpret_cast<float*>(cuda_input.data().pointer().get()),
                                      Channels, num_points);
    state.SetIterationTime(ms / 1000.);
  }
}

// 4 Channels

BENCHMARK_TEMPLATE(CudaInterleavedMemoryFullAccess, 4)
    ->Range(4096, 16777216)
    ->Unit(benchmark::kNanosecond)
    ->UseManualTime();
BENCHMARK_TEMPLATE(InterleavedMemoryFullAccess, 4)
    ->Range(16, 16777216)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(InterleavedMemorySingleChannelAccess, 4)
    ->Range(16, 16777216)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(InterleavedMemoryFullRandomAccess, 4)
    ->Range(16, 16777216)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(InterleavedMemorySingleChannelRandomAccess, 4)
    ->Range(16, 16777216)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

// 8 Channels
BENCHMARK_TEMPLATE(CudaInterleavedMemoryFullAccess, 8)
    ->Range(4096, 16777216)
    ->Unit(benchmark::kNanosecond)
    ->UseManualTime();
BENCHMARK_TEMPLATE(InterleavedMemoryFullAccess, 8)
    ->Range(16, 16777216)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(InterleavedMemorySingleChannelAccess, 8)
    ->Range(16, 16777216)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(InterleavedMemoryFullRandomAccess, 8)
    ->Range(16, 16777216)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(InterleavedMemorySingleChannelRandomAccess, 8)
    ->Range(16, 16777216)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

// 16 Channels
BENCHMARK_TEMPLATE(CudaInterleavedMemoryFullAccess, 16)
    ->Range(4096, 16777216)
    ->Unit(benchmark::kNanosecond)
    ->UseManualTime();
BENCHMARK_TEMPLATE(InterleavedMemoryFullAccess, 16)
    ->Range(16, 16777216)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(InterleavedMemorySingleChannelAccess, 16)
    ->Range(16, 16777216)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(InterleavedMemoryFullRandomAccess, 16)
    ->Range(16, 16777216)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(InterleavedMemorySingleChannelRandomAccess, 16)
    ->Range(16, 16777216)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

/*
As run on a Intel(R) Core(TM) i7-6800K CPU @ 3.40GHz with a Titan V

Interpretation of results: In general doing planar memory operations is a win on CUDA,
but if the all the data is to be consumed as oneInterleaved is a win for the CPU.
If the focus is on accessing data a channel at a time planar can be a win on the CPU, especially
under random access patterns.

When performing random access on the cpu the interelaved is king for full data
access, but falls down if the goal is accessing a single data channel randomly.

Random access is mostly just a bad idea if the goal is performance and should
be especially avoided on the GPU due to it's architecture.

The bigger the data, the more noticebale these differences will become.

---------------------------------------------------------------------------------------------------------
Benchmark                                                                  Time           CPU Iterations
---------------------------------------------------------------------------------------------------------
CudaInterleavedMemoryFullAccess<4>/4096/manual_time                     5481 ns      13377 ns     121338
CudaInterleavedMemoryFullAccess<4>/32768/manual_time                    6548 ns      14622 ns     106741
CudaInterleavedMemoryFullAccess<4>/262144/manual_time                  20137 ns      28439 ns      34717
CudaInterleavedMemoryFullAccess<4>/2097152/manual_time                126761 ns     146925 ns       5517
CudaInterleavedMemoryFullAccess<4>/16777216/manual_time               971305 ns    1169745 ns        721
CudaPlanarMemoryFullAccess<4>/4096/manual_time                          4998 ns      12848 ns     139636
CudaPlanarMemoryFullAccess<4>/32768/manual_time                         5332 ns      13324 ns     131277
CudaPlanarMemoryFullAccess<4>/262144/manual_time                       11993 ns      20311 ns      58574
CudaPlanarMemoryFullAccess<4>/2097152/manual_time                      75669 ns      86628 ns       9294
CudaPlanarMemoryFullAccess<4>/16777216/manual_time                    601065 ns     631046 ns       1160

InterleavedMemoryFullAccess<4>/16/real_time                               20 ns         20 ns   35348649
InterleavedMemoryFullAccess<4>/64/real_time                              115 ns        115 ns    6155124
InterleavedMemoryFullAccess<4>/512/real_time                             995 ns        995 ns     702016
InterleavedMemoryFullAccess<4>/4096/real_time                           8028 ns       8028 ns      86132
InterleavedMemoryFullAccess<4>/32768/real_time                         64864 ns      64864 ns      10870
InterleavedMemoryFullAccess<4>/262144/real_time                       525746 ns     525741 ns       1340
InterleavedMemoryFullAccess<4>/2097152/real_time                     4770496 ns    4770453 ns        146
InterleavedMemoryFullAccess<4>/16777216/real_time                   38075752 ns   38075481 ns         18
PlanarMemoryFullAccess<4>/16/real_time                                    78 ns         78 ns    8934561
PlanarMemoryFullAccess<4>/64/real_time                                   454 ns        454 ns    1541254
PlanarMemoryFullAccess<4>/512/real_time                                 3953 ns       3953 ns     176959
PlanarMemoryFullAccess<4>/4096/real_time                               31936 ns      31936 ns      21927
PlanarMemoryFullAccess<4>/32768/real_time                             255659 ns     255659 ns       2734
PlanarMemoryFullAccess<4>/262144/real_time                           2050801 ns    2050802 ns        341
PlanarMemoryFullAccess<4>/2097152/real_time                         16489216 ns   16489101 ns         42
PlanarMemoryFullAccess<4>/16777216/real_time                       137819052 ns  137817490 ns          4

InterleavedMemorySingleChannelAccess<4>/16/real_time                      25 ns         25 ns   27919642
InterleavedMemorySingleChannelAccess<4>/64/real_time                     146 ns        146 ns    4809997
InterleavedMemorySingleChannelAccess<4>/512/real_time                   1271 ns       1271 ns     551124
InterleavedMemorySingleChannelAccess<4>/4096/real_time                 10259 ns      10259 ns      68210
InterleavedMemorySingleChannelAccess<4>/32768/real_time                82193 ns      82193 ns       8513
InterleavedMemorySingleChannelAccess<4>/262144/real_time              660975 ns     660974 ns       1061
InterleavedMemorySingleChannelAccess<4>/2097152/real_time            5541503 ns    5541462 ns        128
InterleavedMemorySingleChannelAccess<4>/16777216/real_time          44257089 ns   44256624 ns         16
PlanarMemorySingleChannelAccess<4>/16/real_time                           25 ns         25 ns   27891024
PlanarMemorySingleChannelAccess<4>/64/real_time                          146 ns        146 ns    4800756
PlanarMemorySingleChannelAccess<4>/512/real_time                        1270 ns       1270 ns     549856
PlanarMemorySingleChannelAccess<4>/4096/real_time                      10295 ns      10295 ns      67909
PlanarMemorySingleChannelAccess<4>/32768/real_time                     82210 ns      82210 ns       8462
PlanarMemorySingleChannelAccess<4>/262144/real_time                   657492 ns     657492 ns       1057
PlanarMemorySingleChannelAccess<4>/2097152/real_time                 5288156 ns    5288059 ns        133
PlanarMemorySingleChannelAccess<4>/16777216/real_time               42297111 ns   42296660 ns         17

InterleavedMemoryFullRandomAccess<4>/16/real_time                        259 ns        259 ns    2703917
InterleavedMemoryFullRandomAccess<4>/64/real_time                       1520 ns       1520 ns     453336
InterleavedMemoryFullRandomAccess<4>/512/real_time                     13303 ns      13303 ns      52607
InterleavedMemoryFullRandomAccess<4>/4096/real_time                   105829 ns     105829 ns       6618
InterleavedMemoryFullRandomAccess<4>/32768/real_time                  955227 ns     955229 ns        733
InterleavedMemoryFullRandomAccess<4>/262144/real_time                8272755 ns    8272697 ns         80
InterleavedMemoryFullRandomAccess<4>/2097152/real_time             147165298 ns  147164243 ns          5
InterleavedMemoryFullRandomAccess<4>/16777216/real_time           1294646025 ns 1294639207 ns          1
PlanarMemoryFullRandomAccess<4>/16/real_time                             354 ns        354 ns    1975303
PlanarMemoryFullRandomAccess<4>/64/real_time                            2042 ns       2042 ns     341748
PlanarMemoryFullRandomAccess<4>/512/real_time                          17821 ns      17821 ns      39289
PlanarMemoryFullRandomAccess<4>/4096/real_time                        148322 ns     148323 ns       4723
PlanarMemoryFullRandomAccess<4>/32768/real_time                      1275710 ns    1275712 ns        548
PlanarMemoryFullRandomAccess<4>/262144/real_time                    11000481 ns   11000494 ns         64
PlanarMemoryFullRandomAccess<4>/2097152/real_time                  153722167 ns  153722267 ns          4
PlanarMemoryFullRandomAccess<4>/16777216/real_time                1865091801 ns 1865074429 ns          1

InterleavedMemorySingleChannelRandomAccess<4>/16/real_time               249 ns        249 ns    2810152
InterleavedMemorySingleChannelRandomAccess<4>/64/real_time              1453 ns       1453 ns     481826
InterleavedMemorySingleChannelRandomAccess<4>/512/real_time            12669 ns      12669 ns      55003
InterleavedMemorySingleChannelRandomAccess<4>/4096/real_time          103215 ns     103215 ns       6790
InterleavedMemorySingleChannelRandomAccess<4>/32768/real_time         932487 ns     932482 ns        743
InterleavedMemorySingleChannelRandomAccess<4>/262144/real_time       8629322 ns    8629333 ns         76
InterleavedMemorySingleChannelRandomAccess<4>/2097152/real_time    143217945 ns  143216588 ns          5
InterleavedMemorySingleChannelRandomAccess<4>/16777216/real_time  1266137362 ns 1266127801 ns          1
PlanarMemorySingleChannelRandomAccess<4>/16/real_time                    251 ns        251 ns    2783576
PlanarMemorySingleChannelRandomAccess<4>/64/real_time                   1472 ns       1472 ns     475375
PlanarMemorySingleChannelRandomAccess<4>/512/real_time                 12782 ns      12782 ns      54704
PlanarMemorySingleChannelRandomAccess<4>/4096/real_time               103737 ns     103737 ns       6803
PlanarMemorySingleChannelRandomAccess<4>/32768/real_time              826138 ns     826135 ns        831
PlanarMemorySingleChannelRandomAccess<4>/262144/real_time            7514152 ns    7514162 ns         91
PlanarMemorySingleChannelRandomAccess<4>/2097152/real_time          99569287 ns   99568388 ns          7
PlanarMemorySingleChannelRandomAccess<4>/16777216/real_time       1154706240 ns 1154699518 ns          1

CudaInterleavedMemoryFullAccess<8>/4096/manual_time                     7803 ns      16233 ns      87051
CudaInterleavedMemoryFullAccess<8>/32768/manual_time                   11213 ns      19767 ns      62270
CudaInterleavedMemoryFullAccess<8>/262144/manual_time                  57161 ns      81401 ns      11910
CudaInterleavedMemoryFullAccess<8>/2097152/manual_time                408981 ns     459443 ns       1713
CudaInterleavedMemoryFullAccess<8>/16777216/manual_time              3368376 ns    3459725 ns        198
CudaPlanarMemoryFullAccess<8>/4096/manual_time                          5783 ns      14987 ns     122552
CudaPlanarMemoryFullAccess<8>/32768/manual_time                         6546 ns      15038 ns     106166
CudaPlanarMemoryFullAccess<8>/262144/manual_time                       23217 ns      32878 ns      30732
CudaPlanarMemoryFullAccess<8>/2097152/manual_time                     145731 ns     161146 ns       4797
CudaPlanarMemoryFullAccess<8>/16777216/manual_time                   1249592 ns    1385121 ns        565

InterleavedMemoryFullAccess<8>/16/real_time                               39 ns         39 ns   17700249
InterleavedMemoryFullAccess<8>/64/real_time                              227 ns        227 ns    3085155
InterleavedMemoryFullAccess<8>/512/real_time                            1989 ns       1988 ns     352084
InterleavedMemoryFullAccess<8>/4096/real_time                          16043 ns      16043 ns      42535
InterleavedMemoryFullAccess<8>/32768/real_time                        132538 ns     132532 ns       5313
InterleavedMemoryFullAccess<8>/262144/real_time                      1103569 ns    1103557 ns        632
InterleavedMemoryFullAccess<8>/2097152/real_time                     9070662 ns    9070296 ns         79
InterleavedMemoryFullAccess<8>/16777216/real_time                   72380376 ns   72379067 ns         10
PlanarMemoryFullAccess<8>/16/real_time                                   159 ns        159 ns    4409023
PlanarMemoryFullAccess<8>/64/real_time                                   909 ns        909 ns     764501
PlanarMemoryFullAccess<8>/512/real_time                                 7989 ns       7988 ns      87348
PlanarMemoryFullAccess<8>/4096/real_time                               65082 ns      65082 ns      10742
PlanarMemoryFullAccess<8>/32768/real_time                             574288 ns     574286 ns       1196
PlanarMemoryFullAccess<8>/262144/real_time                           4643683 ns    4643687 ns        151
PlanarMemoryFullAccess<8>/2097152/real_time                         36822959 ns   36821284 ns         19
PlanarMemoryFullAccess<8>/16777216/real_time                       331698418 ns  331698492 ns          2

InterleavedMemorySingleChannelAccess<8>/16/real_time                      25 ns         25 ns   27814925
InterleavedMemorySingleChannelAccess<8>/64/real_time                     146 ns        146 ns    4784811
InterleavedMemorySingleChannelAccess<8>/512/real_time                   1284 ns       1284 ns     549043
InterleavedMemorySingleChannelAccess<8>/4096/real_time                 10302 ns      10302 ns      67756
InterleavedMemorySingleChannelAccess<8>/32768/real_time                82867 ns      82864 ns       8503
InterleavedMemorySingleChannelAccess<8>/262144/real_time              698978 ns     698979 ns       1006
InterleavedMemorySingleChannelAccess<8>/2097152/real_time            6561932 ns    6561651 ns        107
InterleavedMemorySingleChannelAccess<8>/16777216/real_time          50721628 ns   50720730 ns         14
PlanarMemorySingleChannelAccess<8>/16/real_time                           24 ns         24 ns   29464491
PlanarMemorySingleChannelAccess<8>/64/real_time                          138 ns        138 ns    5071605
PlanarMemorySingleChannelAccess<8>/512/real_time                        1203 ns       1203 ns     581969
PlanarMemorySingleChannelAccess<8>/4096/real_time                       9749 ns       9749 ns      71770
PlanarMemorySingleChannelAccess<8>/32768/real_time                     77821 ns      77820 ns       8959
PlanarMemorySingleChannelAccess<8>/262144/real_time                   622997 ns     622997 ns       1124
PlanarMemorySingleChannelAccess<8>/2097152/real_time                 5001276 ns    5001226 ns        140
PlanarMemorySingleChannelAccess<8>/16777216/real_time               40139058 ns   40138699 ns         17

InterleavedMemoryFullRandomAccess<8>/16/real_time                        285 ns        285 ns    2458021
InterleavedMemoryFullRandomAccess<8>/64/real_time                       1647 ns       1647 ns     424273
InterleavedMemoryFullRandomAccess<8>/512/real_time                     14392 ns      14392 ns      48617
InterleavedMemoryFullRandomAccess<8>/4096/real_time                   116033 ns     116033 ns       6034
InterleavedMemoryFullRandomAccess<8>/32768/real_time                 1025911 ns    1025913 ns        681
InterleavedMemoryFullRandomAccess<8>/262144/real_time               14618797 ns   14618712 ns         50
InterleavedMemoryFullRandomAccess<8>/2097152/real_time             211309512 ns  211307708 ns          3
InterleavedMemoryFullRandomAccess<8>/16777216/real_time           2066451788 ns 2066422709 ns          1
PlanarMemoryFullRandomAccess<8>/16/real_time                             405 ns        405 ns    1732034
PlanarMemoryFullRandomAccess<8>/64/real_time                            2317 ns       2317 ns     302144
PlanarMemoryFullRandomAccess<8>/512/real_time                          20221 ns      20221 ns      34653
PlanarMemoryFullRandomAccess<8>/4096/real_time                        181313 ns     181312 ns       3853
PlanarMemoryFullRandomAccess<8>/32768/real_time                      1911968 ns    1911971 ns        369
PlanarMemoryFullRandomAccess<8>/262144/real_time                    22626996 ns   22627047 ns         34
PlanarMemoryFullRandomAccess<8>/2097152/real_time                  194876512 ns  194876653 ns          3
PlanarMemoryFullRandomAccess<8>/16777216/real_time                2170034170 ns 2170023099 ns          1

InterleavedMemorySingleChannelRandomAccess<8>/16/real_time               240 ns        240 ns    2948002
InterleavedMemorySingleChannelRandomAccess<8>/64/real_time              1398 ns       1398 ns     500617
InterleavedMemorySingleChannelRandomAccess<8>/512/real_time            12107 ns      12107 ns      57898
InterleavedMemorySingleChannelRandomAccess<8>/4096/real_time          100221 ns     100220 ns       7079
InterleavedMemorySingleChannelRandomAccess<8>/32768/real_time         965996 ns     965999 ns        740
InterleavedMemorySingleChannelRandomAccess<8>/262144/real_time      14025776 ns   14025701 ns         60
InterleavedMemorySingleChannelRandomAccess<8>/2097152/real_time    167531133 ns  167529464 ns          4
InterleavedMemorySingleChannelRandomAccess<8>/16777216/real_time  1448232651 ns 1448219907 ns          1
PlanarMemorySingleChannelRandomAccess<8>/16/real_time                    238 ns        238 ns    2908631
PlanarMemorySingleChannelRandomAccess<8>/64/real_time                   1395 ns       1394 ns     500456
PlanarMemorySingleChannelRandomAccess<8>/512/real_time                 12119 ns      12119 ns      57749
PlanarMemorySingleChannelRandomAccess<8>/4096/real_time                97360 ns      97361 ns       7202
PlanarMemorySingleChannelRandomAccess<8>/32768/real_time              784017 ns     784015 ns        895
PlanarMemorySingleChannelRandomAccess<8>/262144/real_time            7188013 ns    7188020 ns         97
PlanarMemorySingleChannelRandomAccess<8>/2097152/real_time         102618456 ns  102617524 ns          7
PlanarMemorySingleChannelRandomAccess<8>/16777216/real_time       1139981747 ns 1139972374 ns          1

CudaInterleavedMemoryFullAccess<16>/4096/manual_time                   15955 ns      25683 ns      41309
CudaInterleavedMemoryFullAccess<16>/32768/manual_time                  27604 ns      38627 ns      25295
CudaInterleavedMemoryFullAccess<16>/262144/manual_time                190620 ns     206390 ns       3670
CudaInterleavedMemoryFullAccess<16>/2097152/manual_time              1460162 ns    1526466 ns        480
CudaInterleavedMemoryFullAccess<16>/16777216/manual_time            12289388 ns   12359594 ns         57
CudaPlanarMemoryFullAccess<16>/4096/manual_time                         7369 ns      16763 ns      95675
CudaPlanarMemoryFullAccess<16>/32768/manual_time                        8618 ns      17847 ns      82445
CudaPlanarMemoryFullAccess<16>/262144/manual_time                      40327 ns      51157 ns      17230
CudaPlanarMemoryFullAccess<16>/2097152/manual_time                    293817 ns     314556 ns       2395
CudaPlanarMemoryFullAccess<16>/16777216/manual_time                  2677851 ns    2730765 ns        259

InterleavedMemoryFullAccess<16>/16/real_time                              74 ns         74 ns    9409483
InterleavedMemoryFullAccess<16>/64/real_time                             430 ns        430 ns    1630603
InterleavedMemoryFullAccess<16>/512/real_time                           3751 ns       3751 ns     186327
InterleavedMemoryFullAccess<16>/4096/real_time                         30297 ns      30297 ns      23108
InterleavedMemoryFullAccess<16>/32768/real_time                       243555 ns     243547 ns       2880
InterleavedMemoryFullAccess<16>/262144/real_time                     2149885 ns    2149845 ns        323
InterleavedMemoryFullAccess<16>/2097152/real_time                   17243707 ns   17243556 ns         40
InterleavedMemoryFullAccess<16>/16777216/real_time                 137234354 ns  137234587 ns          5
PlanarMemoryFullAccess<16>/16/real_time                                  300 ns        300 ns    2328155
PlanarMemoryFullAccess<16>/64/real_time                                 1736 ns       1736 ns     406896
PlanarMemoryFullAccess<16>/512/real_time                               15181 ns      15181 ns      46242
PlanarMemoryFullAccess<16>/4096/real_time                             163126 ns     163126 ns       4296
PlanarMemoryFullAccess<16>/32768/real_time                           1604607 ns    1604595 ns        435
PlanarMemoryFullAccess<16>/262144/real_time                         19374563 ns   19374531 ns         36
PlanarMemoryFullAccess<16>/2097152/real_time                        70295811 ns   70295810 ns          8
PlanarMemoryFullAccess<16>/16777216/real_time                     1083410025 ns 1083403214 ns          1

InterleavedMemorySingleChannelAccess<16>/16/real_time                     24 ns         24 ns   29262479
InterleavedMemorySingleChannelAccess<16>/64/real_time                    138 ns        138 ns    4997179
InterleavedMemorySingleChannelAccess<16>/512/real_time                  1207 ns       1207 ns     579982
InterleavedMemorySingleChannelAccess<16>/4096/real_time                 9812 ns       9812 ns      71761
InterleavedMemorySingleChannelAccess<16>/32768/real_time               78385 ns      78385 ns       8867
InterleavedMemorySingleChannelAccess<16>/262144/real_time            1110289 ns    1110290 ns        547
InterleavedMemorySingleChannelAccess<16>/2097152/real_time           9960196 ns    9960208 ns         66
InterleavedMemorySingleChannelAccess<16>/16777216/real_time         81768334 ns   81767353 ns          8
PlanarMemorySingleChannelAccess<16>/16/real_time                          24 ns         24 ns   29297636
PlanarMemorySingleChannelAccess<16>/64/real_time                         138 ns        138 ns    5080908
PlanarMemorySingleChannelAccess<16>/512/real_time                       1202 ns       1202 ns     581261
PlanarMemorySingleChannelAccess<16>/4096/real_time                      9738 ns       9738 ns      71929
PlanarMemorySingleChannelAccess<16>/32768/real_time                    77833 ns      77833 ns       8986
PlanarMemorySingleChannelAccess<16>/262144/real_time                  622884 ns     622871 ns       1124
PlanarMemorySingleChannelAccess<16>/2097152/real_time                5005522 ns    5005531 ns        100
PlanarMemorySingleChannelAccess<16>/16777216/real_time              40098373 ns   40098071 ns         17

InterleavedMemoryFullRandomAccess<16>/16/real_time                       337 ns        337 ns    2078762
InterleavedMemoryFullRandomAccess<16>/64/real_time                      1940 ns       1940 ns     360760
InterleavedMemoryFullRandomAccess<16>/512/real_time                    17085 ns      17086 ns      40959
InterleavedMemoryFullRandomAccess<16>/4096/real_time                  146891 ns     146889 ns       4762
InterleavedMemoryFullRandomAccess<16>/32768/real_time                1302403 ns    1302405 ns        536
InterleavedMemoryFullRandomAccess<16>/262144/real_time              27197737 ns   27197766 ns         26
InterleavedMemoryFullRandomAccess<16>/2097152/real_time            276990970 ns  276987619 ns          3
InterleavedMemoryFullRandomAccess<16>/16777216/real_time          3016383410 ns 3016365698 ns          1
PlanarMemoryFullRandomAccess<16>/16/real_time                            555 ns        555 ns    1245525
PlanarMemoryFullRandomAccess<16>/64/real_time                           3190 ns       3190 ns     219431
PlanarMemoryFullRandomAccess<16>/512/real_time                         28004 ns      28004 ns      24956
PlanarMemoryFullRandomAccess<16>/4096/real_time                       282436 ns     282436 ns       2467
PlanarMemoryFullRandomAccess<16>/32768/real_time                     3145856 ns    3145864 ns        217
PlanarMemoryFullRandomAccess<16>/262144/real_time                   62499428 ns   62499590 ns         10
PlanarMemoryFullRandomAccess<16>/2097152/real_time                 329905629 ns  329902670 ns          2
PlanarMemoryFullRandomAccess<16>/16777216/real_time               3401186466 ns 3401170790 ns          1

InterleavedMemorySingleChannelRandomAccess<16>/16/real_time              237 ns        237 ns    2955011
InterleavedMemorySingleChannelRandomAccess<16>/64/real_time             1395 ns       1395 ns     502812
InterleavedMemorySingleChannelRandomAccess<16>/512/real_time           12075 ns      12075 ns      57872
InterleavedMemorySingleChannelRandomAccess<16>/4096/real_time         101954 ns     101952 ns       6872
InterleavedMemorySingleChannelRandomAccess<16>/32768/real_time        968552 ns     968553 ns        723
InterleavedMemorySingleChannelRandomAccess<16>/262144/real_time     19401870 ns   19401917 ns         38
InterleavedMemorySingleChannelRandomAccess<16>/2097152/real_time   192601621 ns  192600203 ns          4
InterleavedMemorySingleChannelRandomAccess<16>/16777216/real_time 2123933077 ns 2123904426 ns          1
PlanarMemorySingleChannelRandomAccess<16>/16/real_time                   238 ns        238 ns    2945630
PlanarMemorySingleChannelRandomAccess<16>/64/real_time                  1390 ns       1390 ns     502401
PlanarMemorySingleChannelRandomAccess<16>/512/real_time                12114 ns      12114 ns      57670
PlanarMemorySingleChannelRandomAccess<16>/4096/real_time               97149 ns      97149 ns       7216
PlanarMemorySingleChannelRandomAccess<16>/32768/real_time             781848 ns     781849 ns        895
PlanarMemorySingleChannelRandomAccess<16>/262144/real_time           7214843 ns    7214856 ns         98
PlanarMemorySingleChannelRandomAccess<16>/2097152/real_time        102491890 ns  102489609 ns          7
PlanarMemorySingleChannelRandomAccess<16>/16777216/real_time      1117445707 ns 1117440223 ns          1
*/

}  // namespace isaac

BENCHMARK_MAIN();

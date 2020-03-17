/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "benchmark/benchmark.h"
#include "cuda_runtime.h"
#include "engine/core/image/image.hpp"
#include "engine/core/tensor/tensor.hpp"
#include "engine/gems/image/conversions.hpp"
#include "engine/gems/image/io.hpp"
#include "engine/gems/image/utils.hpp"

namespace isaac {

void NormalizeCpu(benchmark::State& state) {
  Image3ub image;
  LoadPng("engine/gems/image/data/left.png", image);

  Tensor3f result(3, image.rows(), image.cols());

  for (auto _ : state) {
    ImageToNormalizedTensor(image, result, ImageToTensorIndexOrder::k201,
                            ImageToTensorNormalization::kUnit);
  }
}

void NormalizeGpu(benchmark::State& state) {
  Image3ub image;
  LoadPng("engine/gems/image/data/left.png", image);
  CudaImage3ub cuda_image(image.dimensions());
  Copy(image, cuda_image);

  CudaTensor3f cuda_result(3, image.rows(), image.cols());

  for (auto _ : state) {
    ImageToNormalizedTensor(cuda_image, cuda_result, ImageToTensorIndexOrder::k201,
                            ImageToTensorNormalization::kUnit);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(NormalizeCpu);
BENCHMARK(NormalizeGpu);


// Results as of March 25, 2019 for the left.png which is 1920 x 1080
//
// PC with TitanV and Intel® Core™ i7-6800K CPU @ 3.40GHz × 12 , 32GB Ram
// ----------------------------------------------------
// Benchmark             Time           CPU Iterations
// ----------------------------------------------------
// NormalizeCpu    6607580 ns    6607223 ns        101
// NormalizeGpu      72361 ns      72358 ns       9773
//
// Nano
// ----------------------------------------------------
// Benchmark             Time           CPU Iterations
// ----------------------------------------------------
// NormalizeCpu   35595965 ns   35422153 ns         20
// NormalizeGpu    4026282 ns      63622 ns       1000
//
// Xavier
// ----------------------------------------------------
// Benchmark             Time           CPU Iterations
// ----------------------------------------------------
// NormalizeCpu   27847023 ns   27562344 ns         25
// NormalizeGpu     775838 ns      80262 ns       8778


// Results as of March 25, 2019 for the stairs.png which is 960 x 540
//
// PC with TitanV and Intel® Core™ i7-6800K CPU @ 3.40GHz × 12 , 32GB Ram
// ----------------------------------------------------
// Benchmark             Time           CPU Iterations
// ----------------------------------------------------
// NormalizeCpu    1511264 ns    1511210 ns        463
// NormalizeGpu      22490 ns      22490 ns      30295

// Nano
// ----------------------------------------------------
// Benchmark             Time           CPU Iterations
// ----------------------------------------------------
// NormalizeCpu    8920564 ns    8879110 ns         79
// NormalizeGpu    1045215 ns      60945 ns      11299
//
// Xavier
// ----------------------------------------------------
// Benchmark             Time           CPU Iterations
// ----------------------------------------------------
// NormalizeCpu    7004904 ns    6927820 ns         99
// NormalizeGpu     355761 ns      72271 ns       9316

}  // namespace isaac

BENCHMARK_MAIN();

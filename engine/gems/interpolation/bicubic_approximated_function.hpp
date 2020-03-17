/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include "engine/core/math/types.hpp"
#include "engine/core/tensor/tensor.hpp"
#include "engine/gems/interpolation/cubic.hpp"
#include "engine/gems/interpolation/utils.hpp"

namespace isaac {

// Helper class to approximate a bivariate function with a 2d map of the function values
//  K: type of the co-domain (function output)
//  F: type of a function/object that accept the syntax: obj(x, y) -> K
// TODO(ben): change the function to accept other input type
template <typename K, typename F = std::function<K(size_t, size_t)>>
class BicubicApproximatedFunction {
 public:
  // Constructor
  //  f: The function to approximate (need to be callable with any (row, col) in [0,rows[x[0,cols[)
  //  rows: the number of rows of the domain
  //  cols: the number of columns of the domain
  BicubicApproximatedFunction(F f, size_t rows, size_t cols);

  // Gets the approximate function value at the given value. This will use bicubic interpolation
  // inside cells, and bicubic continuation outside of the domain.
  K operator()(K x, K y);

  // Gets the approximate function value at the given value. This will use bicubic interpolation
  // inside cells, and bicubic continuation outside of the domain.
  K get(K x, K y);

  // Returns the gradient of the interpolated value at a given position.
  Vector2<K> gradient(K x, K y);

  // Returns the hessian (second derivatives) of the interpolated value at a given position.
  Matrix2<K> hessian(K x, K y);

 private:
  // Populates the constrains for a given position
  void populateConstraints(size_t row, size_t col, K& gx, K& gy, K& gxy);

  // Return the coefficient matrix. If not already computed it computes it first.
  const Matrix4<K>& coefficients(size_t row, size_t col);

  // Transform from the world frame to the cache
  Vector2<size_t> cache_T_world(K x, K, K& dx, K& dy) const;

  const F f_;
  std::vector<bool> cached_;
  Tensor<Matrix4<K>, 2> coefficients_;
};

// -------------------------------------------------------------------------------------------------

// Implementation of the BicubicApproximatedFunction
template <typename K, typename F>
BicubicApproximatedFunction<K, F>::BicubicApproximatedFunction(F f, size_t rows, size_t cols)
    : f_(std::move(f)) {
  ASSERT(rows > 1 && cols > 1, "Invalid boundary (Size of the cache: %zdx%zd", rows, cols);
  coefficients_.resize(rows - 1, cols - 1);
  cached_.resize(coefficients_.element_count(), false);
}

template <typename K, typename F>
Vector2<size_t> BicubicApproximatedFunction<K, F>::cache_T_world(K x, K y, K& dx, K& dy) const {
  Vector2<size_t> cell;
  cell.x() = x < K(0) ? 0 : std::min(static_cast<size_t>(x), coefficients_.dimensions()[0] - 1);
  cell.y() = y < K(0) ? 0 : std::min(static_cast<size_t>(y), coefficients_.dimensions()[1] - 1);
  dx = x - static_cast<K>(cell.x());
  dy = y - static_cast<K>(cell.y());
  return cell;
}

template <typename K, typename F>
K BicubicApproximatedFunction<K, F>::operator()(K x, K y) {
  return get(x, y);
}

template <typename K, typename F>
K BicubicApproximatedFunction<K, F>::get(K x, K y) {
  K dx, dy;
  const Vector2<size_t> cell = cache_T_world(x, y, dx, dy);
  return BicubicInterpolationEvaluation(dx, dy, coefficients(cell.x(), cell.y()));
}

template <typename K, typename F>
Vector2<K> BicubicApproximatedFunction<K, F>::gradient(K x, K y) {
  K dx, dy;
  const Vector2<size_t> cell = cache_T_world(x, y, dx, dy);
  return BicubicInterpolationGradient(dx, dy, coefficients(cell.x(), cell.y()));
}

template <typename K, typename F>
Matrix2<K> BicubicApproximatedFunction<K, F>::hessian(K x, K y) {
  K dx, dy;
  const Vector2<size_t> cell = cache_T_world(x, y, dx, dy);
  return BicubicInterpolationHessian(dx, dy, coefficients(cell.x(), cell.y()));
}

template <typename K, typename F>
void BicubicApproximatedFunction<K, F>::populateConstraints(size_t row, size_t col, K& gx, K& gy,
                                                            K& gxy) {
  // Compute the gradient along the X axis
  const size_t idx_b = row > 0 ? row - 1 : row;
  const size_t idx_a = row < coefficients_.dimensions()[0] ? row + 1 : row;
  const K dx_inv = K(1) / static_cast<K>(idx_a - idx_b);
  gx = (f_(idx_a, col) - f_(idx_b, col)) * dx_inv;

  // Compute the gradient along the Y axis
  const size_t idy_b = col > 0 ? col - 1 : col;
  const size_t idy_a = col < coefficients_.dimensions()[1] ? col + 1 : col;
  const K dy_inv = K(1) / static_cast<K>(idy_a - idy_b);
  gy = (f_(row, idy_a) - f_(row, idy_b)) * dy_inv;

  // Compute the second derivative along the X and Y axis
  gxy =
      (f_(idx_a, idy_a) + f_(idx_b, idy_b) - f_(idx_a, idy_b) - f_(idx_b, idy_a)) * dx_inv * dy_inv;
}

template <typename K, typename F>
const Matrix4<K>& BicubicApproximatedFunction<K, F>::coefficients(size_t row, size_t col) {
  const size_t index = row * coefficients_.dimensions()[1] + col;
  if (!cached_[index]) {
    Matrix4<K> mat;
    // See BicubicCoefficients for details:
    //       |  P(0, 0)    P(0, 1)    Py(0, 0)    Py(0, 1)   |
    // mat = |  P(1, 0)    P(1, 1)    Py(1, 0)    Py(1, 1)   |
    //       |  Px(0, 0)   Px(0, 1)   Pxy(0, 0)   Pxy(0, 1)  |
    //       |  Px(1, 0)   Px(1, 1)   Pxy(1, 0)   Pxy(1, 1)  |
    for (int drow = 0; drow < 2; drow++) {
      for (int dcol = 0; dcol < 2; dcol++) {
        mat(drow, dcol) = f_(row + drow, col + dcol);
        populateConstraints(row + drow, col + dcol, mat(2 + drow, dcol), mat(drow, 2 + dcol),
                            mat(2 + drow, 2 + dcol));
      }
    }
    coefficients_(row, col) = BicubicCoefficients(mat);
    cached_[index] = true;
  }
  return coefficients_(row, col);
}

}  // namespace isaac

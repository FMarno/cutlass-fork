/***************************************************************************************************
* Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

// Standard Library includes
#include <cstdlib>
#include <cmath>
#include <type_traits>
#include <cstdint>
#include <vector>

#include <oneapi/mkl/rng/device.hpp>

// Cutlass includes
#include "cutlass/cutlass.h"
#include "cutlass/complex.h"
#include "cutlass/util/reference/device/tensor_foreach.h"


namespace cutlass {
namespace reference {
namespace device {

namespace detail {
template <typename Element>
struct RandomGaussianFunc {

  using FloatType = typename std::conditional<(sizeof(Element) > 4), double, float>::type;
  using IntType = typename std::conditional<(sizeof(Element) > 4), int64_t, int>::type;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    uint64_t seed;
    FloatType mean;
    FloatType stddev;
    int int_scale;
    FloatType float_scale_up;
    FloatType float_scale_down;
    int exclude_zero;           ///< If non-negative, excludes zeros

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      uint64_t seed_ = 0,
      Element mean_ = 0,
      Element stddev_ = 1,
      int int_scale_ = -1,
      int exclude_zero_ = -1
    ):
      seed(seed_),
      mean(static_cast<FloatType>(mean_)),
      stddev(static_cast<FloatType>(stddev_)),
      int_scale(int_scale_),
      exclude_zero(exclude_zero_) {

      float_scale_up = FloatType(IntType(1) << int_scale); // scale up to clamp low order bits
      float_scale_down = FloatType(1) / FloatType(IntType(1) << int_scale);
    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  /// RNG state object
  oneapi::mkl::rng::device::gaussian<FloatType> distribution;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  RandomGaussianFunc(Params const &params):
    params(params),
    distribution(static_cast<FloatType>(params.mean), static_cast<FloatType>(params.stddev)) {}

  /// Compute random value and update RNG state
  Element operator()() {
    oneapi::mkl::rng::device::philox4x32x10<> generator(params.seed,
      ThreadIdxX() + BlockIdxX() * BlockDimX());
    FloatType rnd = oneapi::mkl::rng::device::generate(distribution, generator);

    Element result;
    if (params.int_scale >= 0) {
      rnd = FloatType(std::llround(rnd * params.float_scale_up));
      result = Element(rnd * params.float_scale_down);
    }
    else {
      result = Element(rnd);
    }

    if (params.exclude_zero >=0 && result == Element(0.0)) {
      if (rnd > FloatType(0)) {
        rnd += FloatType(1);
      } else {
        rnd -= FloatType(1);
      }
      result = Element(rnd);
    }

    return result;
  }
};


template <typename Real>
struct RandomGaussianFunc<complex<Real>> {

  using Element = complex<Real>;
  using FloatType = typename std::conditional<(sizeof(Real) > 4), double, float>::type;
  using IntType = typename std::conditional<(sizeof(Real) > 4), int64_t, int>::type;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    uint64_t seed;
    FloatType mean;
    FloatType stddev;
    int int_scale;
    FloatType float_scale_up;
    FloatType float_scale_down;
    int exclude_zero;           ///< If non-negative, excludes zeros

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      uint64_t seed_ = 0,
      Real mean_ = 0,
      Real stddev_ = 1,
      int int_scale_ = -1,
      int exclude_zero_ = -1
    ):
      seed(seed_),
      mean(static_cast<FloatType>(mean_)),
      stddev(static_cast<FloatType>(stddev_)),
      int_scale(int_scale_),
      exclude_zero(exclude_zero_) {

      float_scale_up = FloatType(IntType(1) << int_scale);
      float_scale_down = FloatType(1) / FloatType(IntType(1) << int_scale);
    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  /// RNG state object
  oneapi::mkl::rng::device::gaussian<FloatType> distribution;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  RandomGaussianFunc(Params const &params):
    params(params),
    distribution(static_cast<FloatType>(params.mean), static_cast<FloatType>(params.stddev)) {}

  /// Compute random value and update RNG state
  Element operator()() {
    oneapi::mkl::rng::device::philox4x32x10<> generator(params.seed,
      ThreadIdxX() + BlockIdxX() * BlockDimX());
    FloatType rnd_r = oneapi::mkl::rng::device::generate(distribution, generator);
    FloatType rnd_i = oneapi::mkl::rng::device::generate(distribution, generator);

    Element result;
    if (params.int_scale >= 0) {
      rnd_r = FloatType(std::llround(rnd_r * params.float_scale_up));
      rnd_i = FloatType(std::llround(rnd_i * params.float_scale_up));

      result = {
        Real(rnd_r * params.float_scale_down),
        Real(rnd_i * params.float_scale_down)
      };
    }
    else {
      result = Element(Real(rnd_r), Real(rnd_i));
    }

    if (params.exclude_zero >= 0 &&
        result.real() == Real(0.0) &&
        result.imag() == Real(0.0)) {

      if (rnd_r > FloatType(0)) {
        rnd_r += FloatType(1);
      } else {
        rnd_r -= FloatType(1);
      }
      result = Element(Real(rnd_r), Real(rnd_i));
    }

    return result;
  }
};

/// Computes a random Gaussian distribution
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillRandomGaussianFunc {

  /// View type
  using TensorView = TensorView<Element, Layout>;

  /// Scalar type
  typedef typename TensorView::Element T;

  /// Coordinate in tensor's index space
  typedef typename TensorView::TensorCoord TensorCoord;

  using RandomFunc = RandomGaussianFunc<Element>;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    TensorView view;
    typename RandomFunc::Params random;

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      TensorView view_ = TensorView(),
      typename RandomFunc::Params random_ = typename RandomFunc::Params()
    ):
      view(view_), random(random_) {

    }
  };

  //
  // Data members
  //

  Params params;
  RandomFunc random;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  TensorFillRandomGaussianFunc(Params const &params): params(params), random(params.random) {

  }

  /// Compute random value and update RNG state
  void operator()(TensorCoord const &coord) {

    params.view.at(coord) = random();
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a Gaussian distribution.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillRandomGaussian(
  TensorView<Element, Layout> view,       ///< destination tensor
  uint64_t seed,                          ///< seed for RNG
  typename RealType<Element>::Type mean = Element(0),   ///< Gaussian distribution's mean
  typename RealType<Element>::Type stddev = Element(1), ///< Gaussian distribution's standard deviation
  int bits = -1,                          ///< If non-negative, specifies number of fractional bits that
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.
  int exclude_zero = -1) {                ///< If non-negative, excludes zeros from tensor init

  using RandomFunc = detail::RandomGaussianFunc<Element>;
  using Func = detail::TensorFillRandomGaussianFunc<Element, Layout>;
  using Params = typename Func::Params;

  TensorForEach<Func, Layout::kRank, Params>(
    view.extent(),
    Params(view, typename RandomFunc::Params(seed, mean, stddev, bits, exclude_zero)),
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a Gaussian distribution.
template <typename Element>               ///< Element type
void BlockFillRandomGaussian(
  Element *ptr,
  size_t capacity,
  uint64_t seed,                              ///< seed for RNG
  typename RealType<Element>::Type mean,      ///< Gaussian distribution's mean
  typename RealType<Element>::Type stddev,    ///< Gaussian distribution's standard deviation
  int bits = -1) {                            ///< If non-negative, specifies number of fractional bits that
                                              ///  are not truncated to zero. Permits reducing precision of
                                              ///  data.

  using RandomFunc = detail::RandomGaussianFunc<Element>;

  typename RandomFunc::Params params(seed, mean, stddev, bits);

  BlockForEach<Element, RandomFunc>(ptr, capacity, params);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Computes a random Uniform distribution
template <typename Element>                ///< Element type
struct RandomUniformFunc {

  using FloatType = typename std::conditional<
    (sizeof(Element) > 4),
    double,
    float>::type;

  using IntType = typename std::conditional<
    (sizeof(Element) > 4),
    int64_t,
    int>::type;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    uint64_t seed;
    FloatType max;
    FloatType min;
    int int_scale;
    FloatType float_scale_up;
    FloatType float_scale_down;

    /// Default ctor
    Params() { }

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      uint64_t seed_ = 0,
      Element max_ = 1,
      Element min_ = 0,
      int int_scale_ = -1
    ):
      seed(seed_),
      max(static_cast<FloatType>(max_)),
      min(static_cast<FloatType>(min_)),
      int_scale(int_scale_) {

      float_scale_up = FloatType(IntType(2) << int_scale); // scale up to clamp low order bits
      float_scale_down = FloatType(1) / FloatType(IntType(2) << int_scale);
    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;
  oneapi::mkl::rng::device::uniform<FloatType> distribution;

  //
  // Methods
  //

  explicit RandomUniformFunc(Params const &params):
      params(params),
      distribution(static_cast<FloatType>(params.min), static_cast<FloatType>(params.max)){}

  /// Compute random value and update RNG state
  Element operator()() {
    oneapi::mkl::rng::device::philox4x32x10<> generator(params.seed,
      ThreadIdxX() + BlockIdxX() * BlockDimX());
    FloatType rnd = oneapi::mkl::rng::device::generate(distribution, generator);

    // Random values are cast to integer after scaling by a power of two to facilitate error
    // testing
    Element result;

    if (params.int_scale >= 0) {
      rnd = FloatType(IntType(sycl::round(rnd * params.float_scale_up)));
      result = Element(IntType(rnd * params.float_scale_down));
    }
    else {
      result = Element(rnd);
    }

    return result;
  }
};

/// Computes a random uniform distribution
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillRandomUniformFunc {

  /// View type
  using TensorView = TensorView<Element, Layout>;

  /// Scalar type
  typedef typename TensorView::Element T;

  /// Coordinate in tensor's index space
  typedef typename TensorView::TensorCoord TensorCoord;

  using RandomFunc = RandomUniformFunc<Element>;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    TensorView view;
    typename RandomFunc::Params random;

    /// Default ctor
    Params() { }

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      TensorView view_ = TensorView(),
      typename RandomFunc::Params random_ = RandomFunc::Params()
    ):
      view(view_), random(random_) {

    }
  };

  //
  // Data members
  //

  Params params;
  RandomFunc random;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  TensorFillRandomUniformFunc(Params const &params): params(params), random(params.random) {
  }

  /// Compute random value and update RNG state
  void operator()(TensorCoord const &coord) {

    params.view.at(coord) = random();
  }
};


} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a uniform random distribution.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillRandomUniform(
  TensorView<Element, Layout> view,       ///< destination tensor
  uint64_t seed,                          ///< seed for RNG
  typename RealType<Element>::Type max = Element(1), ///< upper bound of distribution
  typename RealType<Element>::Type min = Element(0), ///< lower bound for distribution
  int bits = -1) {                        ///< If non-negative, specifies number of fractional bits that
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.

  using RandomFunc = detail::RandomUniformFunc<Element>;
  using Func = detail::TensorFillRandomUniformFunc<Element, Layout>;
  using Params = typename Func::Params;

  // TODO does bits line up with int-scale
  // seems ok in the nvidia impl
  typename RandomFunc::Params random(seed, max, min, bits);

  TensorForEach<Func, Layout::kRank, Params>(
    view.extent(),
    Params(view, random)
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Functor to fill a tensor with zeros off the diagonal and a uniform value on the diagonal.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillDiagonalFunc {

  /// View type
  using TensorView = TensorView<Element, Layout>;

  /// Scalar type
  typedef typename TensorView::Element T;

  /// Coordinate in tensor's index space
  typedef typename TensorView::TensorCoord TensorCoord;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    TensorView view;
    Element diag;
    Element other;

    /// Default ctor
    Params() { }

    //
    // Methods
    //

    Params(
      TensorView view_ = TensorView(),
      Element diag_ = Element(1),
      Element other_ = Element(0)
    ):
      view(view_), diag(diag_), other(other_) {

    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  TensorFillDiagonalFunc(Params const &params): params(params) {

  }

  /// Updates the tensor
  void operator()(TensorCoord const &coord) {

    bool is_diag = true;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < Layout::kRank; ++i) {
      if (coord[i] != coord[i - 1]) {
        is_diag = false;
        break;
      }
    }

    params.view.at(coord) = (is_diag ? params.diag : params.other);
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a uniform random distribution.
template <typename Element>
void BlockFillRandomUniform(
  Element *ptr,
  size_t capacity,
  uint64_t seed,                          ///< seed for RNG
  typename RealType<Element>::Type max,   ///< upper bound of distribution
  typename RealType<Element>::Type min,   ///< lower bound for distribution
  int bits = -1                           ///< If non-negative, specifies number of fractional bits that
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.
  ) {

  using RandomFunc = detail::RandomUniformFunc<Element>;

  typename RandomFunc::Params params(seed, max, min, bits);
  BlockForEach<Element, RandomFunc>(ptr, capacity, params);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor everywhere with a unique value for its diagonal.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillDiagonal(
  TensorView<Element, Layout> view,       ///< destination tensor
  Element diag = Element(1),              ///< value to write in the diagonal
  Element other = Element(0)) {           ///< value to write off the diagonal

  typedef detail::TensorFillDiagonalFunc<Element, Layout> Func;
  typedef typename Func::Params Params;

  TensorForEach<Func, Layout::kRank, Params>(
    view.extent(),
    Params(view, diag, other)
  );
}

/// Fills a tensor with a uniform value

template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFill(
  TensorView<Element, Layout> view,         ///< destination tensor
  Element val = Element(0)) {               ///< value to uniformly fill it with

  TensorFillDiagonal(view, val, val);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Computes a random Gaussian distribution
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillLinearFunc {

  /// View type
  using TensorView = TensorView<Element, Layout>;

  /// Scalar type
  typedef typename TensorView::Element T;

  /// Coordinate in tensor's index space
  typedef typename TensorView::TensorCoord TensorCoord;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    TensorView view;
    Array<Element, Layout::kRank> v;
    Element s;

    /// Default ctor
    Params() { }

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      TensorView view_,      ///< destination tensor
      Array<Element, Layout::kRank> const & v_,
      Element s_ = Element(0)
    ):
      view(view_), v(v_), s(s_) { }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  TensorFillLinearFunc(Params const &params): params(params) {

  }
  TensorFillLinearFunc(TensorFillLinearFunc const &) = default;

  /// Compute random value and update RNG state
  void operator()(TensorCoord const &coord) {

    Element sum = params.s;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Layout::kRank; ++i) {
      if constexpr (is_complex<Element>::value) {
        if constexpr (sizeof_bits<Element>::value <= 32) {
          sum = Element(static_cast<complex<float>>(sum) +
                  static_cast<complex<float>>(params.v[i]) * static_cast<complex<float>>(coord[i]));
        }
      }
      else if constexpr (sizeof_bits<Element>::value <= 32) {
        if constexpr (std::numeric_limits<Element>::is_integer) {
          sum = Element(static_cast<int32_t>(sum) +
                  static_cast<int32_t>(params.v[i]) * static_cast<int32_t>(coord[i]));
        }
        else {
          sum = Element(static_cast<float>(sum) +
                  static_cast<float>(params.v[i]) * static_cast<float>(coord[i]));
        }
      }
      else {
        sum += params.v[i] * coord[i];
      }
    }

    params.view.at(coord) = sum;
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills tensor with a linear combination of its coordinate and another vector
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillLinear(
  TensorView<Element, Layout> view,      ///< destination tensor
  Array<Element, Layout::kRank> const & v,
  Element s = Element(0)) {

  using Func = detail::TensorFillLinearFunc<Element, Layout>;
  using Params = typename Func::Params;

  TensorForEach<Func, Layout::kRank, Params>(
    view.extent(),
    Params(view, v, s),
    /*grid_size*/0, /*block_size*/0
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a block of data with sequential elements
template <
  typename Element
>
void BlockFillSequential(
  Element *ptr,
  int64_t capacity,
  Element v = Element(1),
  Element s = Element(0)) {

  using Layout = layout::PackedVectorLayout;
  Layout::TensorCoord size(static_cast<Layout::Index>(capacity)); // -Wconversion
  Layout layout = Layout::packed(size);
  TensorView<Element, Layout> view(ptr, layout, size);

  Array<Element, Layout::kRank> c{};
  c[0] = v;

  TensorFillLinear(view, c, s);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace reference
} // namespace cutlass

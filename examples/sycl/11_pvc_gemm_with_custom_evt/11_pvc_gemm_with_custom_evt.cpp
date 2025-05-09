/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
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
/*! \file
    \brief CUTLASS Intel PVC Gemm with a custom EVT epilogue.

    Epilogue form a tree structure, which utilizes a visitor pattern to implement a range of
    algorithms. There is an excellent article covering the topic here:
    https://research.colfax-intl.com/epilogue_visitor_tree/. In order to use the custom epilogue
    with the epilogue collective builder, a FusionOpInfo class defined, which is a bit of a hack
    depending on knowledge of the internal implementation of the collective builder.

    This particular EVT will do matrix multiplication without scaling or accumulation i.e. D = AB.

    To build & run this example (from your build dir):

      $ ninja 11_pvc_gemm_with_custom_evt
      $ ./examples/sycl/11_pvc_gemm_with_custom_evt/11_pvc_gemm_with_custom_evt

    Call with `--help` for information about available options
*/

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/kernel_hardware_info.h"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "cutlass/array.h"
#include "cutlass/coord.h" // needed?
#include "cutlass/tensor_view.h" // needed?

#include "helper.h"
#include "sycl_common.hpp"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, l, iterations;

  Options():
    help(false),
    error(false),
    m(5120), n(4096), k(4096), l(1), iterations(100)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "PVC GEMM Example\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --iterations=<int>          Iterations\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class Gemm
>
struct ExampleRunner {

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  //
  // Methods
  //

  bool verify(const ProblemShapeType& problem_size) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          ElementCompute(1.),
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          ElementCompute(0.),
          ref_D, // place-holder value
          ref_D,
          ElementCompute(0),
          L,     // batch_count
          M * K, // batch_stride_A
          K * N, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
        );

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::device::BlockCompareEqual(
      block_ref_D.get(), block_D.get(), block_D.size());

    return passed;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(static_cast<std::size_t>(M) * K * L);
    block_B.reset(static_cast<std::size_t>(K) * N * L);
    block_D.reset(static_cast<std::size_t>(M) * N * L);
    block_ref_D.reset(static_cast<std::size_t>(M) * N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{}, nullptr, stride_D, block_D.get(), stride_D},
      hw_info
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run());

    syclcompat::wait();

    // Verify that the result is correct
    bool passed = verify(problem_size);
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

    if(!passed) return cutlass::Status::kErrorInternal;

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        gemm_op.run();
      }
      syclcompat::wait();

      float cute_time = timer.seconds() / options.iterations;
      double tflops = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      printf("Cutlass GEMM Performance:     [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);
    }

    return cutlass::Status::kSuccess;
  }

};

// D = acc
template<
  class ElementOutput_
>
struct Acc : cutlass::epilogue::fusion::FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementOutput;
};


template <typename T>
struct Identity {
  static const bool kIsHeavy = false;

  CUTLASS_HOST_DEVICE
  T operator()(T value) const {
    return value;
  }
};

template <typename T, int N>
struct Identity<cutlass::Array<T, N> > {
  CUTLASS_HOST_DEVICE
  cutlass::Array<T, N> operator()(cutlass::Array<T, N> value) const {
    return value;
  }
};
// TODO can I remove the Sm90Compute
template< class ElementOutput>
using XeAcc =
  cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<Identity, ElementOutput, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::fusion::Sm90AccFetch>;

// D = acc
template <
  class ElementOutput_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct cutlass::epilogue::fusion::FusionCallbacks<
    cutlass::epilogue::IntelXeXMX16,
    Acc<ElementOutput_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : XeAcc<ElementOutput_> {
  using Impl = XeAcc<ElementOutput_>;
  using Operation = Acc<ElementOutput_>;
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementOutput;

  struct Arguments {

    // Conversion to the args expected by the visitor implementation
    // to_underlying_arguments will implicitly call this
    operator typename Impl::Arguments() const {
      return {};
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};
namespace cutlass::epilogue::collective::detail {
  template <class ElementD>
  struct FusionOpInfo<Acc<ElementD>> {
      constexpr static bool HasBuilder = true;

      template <
        class DispatchPolicy,
        class TileShape_MNK,
        class EpilogueTile,
        class>
      using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
        DispatchPolicy,
        Acc<ElementD>,
        TileShape_MNK,
        EpilogueTile
      >;
  };
}


int main(int argc, const char** argv)
{
  //
  // Parse options
  //

  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
  // information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  bool passed;

  // The code section below describes datatype for input, output matrices and computation between
  // elements in input matrices.
  using ElementAccumulator = bfloat16_t;     // <- data type of accumulator
  using ElementComputeEpilogue = bfloat16_t; // <- data type of epilogue operations
  using ElementInputA = bfloat16_t;     // <- data type of elements in input matrix A
  using ElementInputB = bfloat16_t;     // <- data type of elements in input matrix B
  using ElementOutput = bfloat16_t;          // <- data type of elements in output matrix D

  constexpr int AlignmentA = sizeof(ElementInputA);
  constexpr int AlignmentB = sizeof(ElementInputB);
  constexpr int AlignmentC = sizeof(ElementAccumulator);
  constexpr int AlignmentD = sizeof(ElementOutput);

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  // Workgroup-level tile
  using TileShape = Shape<_256, _256, _32>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
    ElementInputA, LayoutA, AlignmentA,
    ElementInputB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, Shape<_1, _1, _1>,                 // the ClusterShape is always <1,1,1> on IntelXe
    cutlass::gemm::collective::StageCountAuto,    // let the builder select the number of pipeline stages (i.e. prefetch iters)
    cutlass::gemm::collective::KernelScheduleAuto // let the builder select the mainloop schedule
  >::CollectiveOp;

  // Define a Linear Combination, Elementwise Activation (LinCombEltAct) epilogue with ReLU activation
  using EpilogueOp = Acc<ElementOutput>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
    TileShape, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementComputeEpilogue,
    ElementAccumulator,
    ElementAccumulator, LayoutC, AlignmentC,
    ElementOutput,      LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    EpilogueOp
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  Shape<int, int, int, int>,
  CollectiveMainloop,
  CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  ExampleRunner<Gemm> runner;

  CUTLASS_CHECK(runner.run(options, hw_info));

  return 0;
}

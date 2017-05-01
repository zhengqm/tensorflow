
# How tensorflow implements and executes an operation

## Important folders

+ tensorflow/core/ops
+ tensorflow/core/kernels

## Details

### core/ops

Register operations to global registry with a builder design patter, examples:

```
REGISTER_OP("MatMul")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {half, float, double, int32, complex64, complex128}")
    .SetShapeFn(shape_inference::MatMulShape)
    .Doc(R"doc(
Multiply the matrix "a" by the matrix "b".

The inputs must be two-dimensional matrices and the inner dimension of
"a" (after being transposed if transpose_a is true) must match the
outer dimension of "b" (after being transposed if transposed_b is
true).

*Note*: The default kernel implementation for MatMul on GPUs uses
cublas.

transpose_a: If true, "a" is transposed before multiplication.
transpose_b: If true, "b" is transposed before multiplication.
)doc");
```

REGISTER_OP is a macro that instantiates a builder and the following chain calls specify the the input/output/attributes
of the opperation.

### core/kernels

The matrix multiplication is implemented in `matmul_op.cc`:

```
template <typename Device, typename T, bool USE_CUBLAS>
class MatMulOp : public OpKernel {
 public:
  explicit MatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    // ... Pre-check
    LaunchMatMul<Device, T, USE_CUBLAS>::launch(ctx, this, a, b, dim_pair, out);
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
};
```

Depending on the avaliable devices, `LaunchMatMul::launch` may run on CPU or GPU or other devices:

On GPU:
``` C++
template <typename T>
struct LaunchMatMul<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx, OpKernel* kernel, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      Tensor* out) {
    perftools::gputools::blas::Transpose trans[] = {
        perftools::gputools::blas::Transpose::kNoTranspose,
        perftools::gputools::blas::Transpose::kTranspose};
    
    // More preparation ...

    if (LaunchBlasGemv<T>::IsSupported() && n == 1) {
      // This is a matrix*vector multiply so use GEMV to compute A * b.
      // Here we are multiplying in the natural order, so we have to flip
      // the transposition flag to compensate for the tensor being stored
      // row-major.
      LaunchBlasGemv<T>::Compute(ctx, stream, !transpose_a, transpose_a ? m : k,
                                 transpose_a ? k : m, a_ptr, b_ptr, &c_ptr);
    } else {
      bool blas_launch_status =
          stream
              ->ThenBlasGemm(blas_transpose_b, blas_transpose_a, n, m, k, 1.0f,
                             b_ptr, transpose_b ? k : n, a_ptr,
                             transpose_a ? m : k, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        ctx->SetStatus(errors::Internal(
            "Blas GEMM launch failed : a.shape=(", a.dim_size(0), ", ",
            a.dim_size(1), "), b.shape=(", b.dim_size(0), ", ", b.dim_size(1),
            "), m=", m, ", n=", n, ", k=", k));
      }
    }
  }
};
```

On CPU:

```
template <typename T>
struct MatMulFunctor<CPUDevice, T> {
  void operator()(
      const CPUDevice& d, typename MatMulTypes<T>::out_type out,
      typename MatMulTypes<T>::in_type in0,
      typename MatMulTypes<T>::in_type in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair) {
    MatMul<CPUDevice>(d, out, in0, in1, dim_pair);
  }
};
```
using the Eigen library. The MatMul template is defined in `matmul_op.h`.
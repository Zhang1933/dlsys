# Homework 4 extra

Public repository and stub/testing code for Homework 4-etra of 10-714.

# Transformer

* Penn Treebank dataset

# Reference:
* [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
* [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)


```cpp

__global__ void reduce_ws_add(float *a, float *out,size_t reduce_size,size_t size){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    size_t offset = i * reduce_size;
    scalar_t reduce_sum = 0;
    for (size_t j = 0; j < reduce_size; j++) {
      reduce_sum += a[offset + j];
    }
    out[i] = reduce_sum;
  }
  
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  reduce_ws_add<<<dim.grid, dim.block>>>(a.ptr, out->ptr,reduce_size,out->size);
  cudaCheckErrors("kernel");
  /// END SOLUTION
}

```

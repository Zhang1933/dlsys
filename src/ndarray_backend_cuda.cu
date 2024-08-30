#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides
// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__device__ size_t gid2strides_idx(size_t gid, CudaVec shape,CudaVec strides, size_t offset){
    size_t pos=offset;
    for(int i=strides.size-1;i>=0;i--){
      pos+=(gid%shape.data[i])*strides.data[i];
      gid/=shape.data[i];
    }
    return pos;
}


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if(gid<size){
    out[gid]=a[gid2strides_idx(gid,shape,strides,offset)];
  }
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if(gid<size){
    out[gid2strides_idx(gid,shape,strides,offset)]=a[gid];
  }
  /// END SOLUTION
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  cudaCheckErrors("kernel");
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(const scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if(gid<size){
    out[gid2strides_idx(gid,shape,strides,offset)]=val;
  }
  /// END SOLUTION
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
    CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr,size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  cudaCheckErrors("kernel");
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */



template <typename F>
__global__ void EwiseOptKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size, F f) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] =f(a[gid] ,b[gid]);
}

template <typename F>
__global__ void ScalarEwiseOptKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size,F f) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = f(a[gid] , val);
}

template <typename F>
__global__ void  SingleEwiseOptKernel(const scalar_t* a,scalar_t* out,size_t size, F f) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; 
  if (gid < size) out[gid] =f(a[gid]);
}

__device__  scalar_t MUL_F(const scalar_t a,const scalar_t b){
  return a*b;
}
__device__  scalar_t DIV_F(const scalar_t a,const scalar_t b){
  return a/b;
}
__device__  scalar_t Power_F(const scalar_t a,const scalar_t b){
  return powf(a,b);
}
__device__  scalar_t Max_F(const scalar_t a,const scalar_t b){
  return fmax(a,b);
}
__device__  scalar_t Eq_F(const scalar_t a,const scalar_t b){
  return a==b;
}
__device__  scalar_t Ge_F(const scalar_t a,const scalar_t b){
  return a>=b; 
}
__device__  scalar_t Log_F(const scalar_t a){
  return logf(a);
}
__device__  scalar_t Exp_F(const scalar_t a){
  return expf(a);
}
__device__  scalar_t Tann_F(const scalar_t a){
  return tanhf(a);
}
__device__  scalar_t ADD_F(const scalar_t a,const scalar_t b){
  return a+b;
}

using func_two_op = scalar_t (*) (scalar_t, scalar_t);  
using func_one_op = scalar_t (*) (scalar_t);  
__device__ func_two_op dev_func_ops[] = { MUL_F, DIV_F ,Power_F,Max_F,Eq_F,Ge_F,(func_two_op)Log_F, (func_two_op)Exp_F ,(func_two_op)Tann_F,ADD_F}; //0-8 cast all function pointer to func_two_op type for index

enum opType{
  e_MUL,e_DIV,e_Power,e_Max,e_Eq,e_Ge,e_Log,e_Exp,e_Tann,e_Add
};

void getFuncHostPtr(int opType, func_two_op &h_pointFunction){
    if (opType < 0 || opType >9 ){
        throw std::range_error("Operations type: 0-9");
    }
    cudaMemcpyFromSymbol(&h_pointFunction, dev_func_ops, sizeof(func_two_op), opType*sizeof(func_two_op));
    cudaCheckErrors("cudaMemcpyFromSymbol");
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Mul together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);

  func_two_op h_pointFunction;
  getFuncHostPtr(e_MUL,h_pointFunction);

  EwiseOptKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size,h_pointFunction);
}
void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  func_two_op h_pointFunction;
  getFuncHostPtr(e_MUL,h_pointFunction);

  ScalarEwiseOptKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size,h_pointFunction);
  cudaCheckErrors("Kernel");
}
void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Div together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);

  func_two_op h_pointFunction;
  getFuncHostPtr(e_DIV,h_pointFunction);

  EwiseOptKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size,h_pointFunction);
  cudaCheckErrors("Kernel");
}
void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  func_two_op h_pointFunction;
  getFuncHostPtr(e_DIV,h_pointFunction);

  ScalarEwiseOptKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size,h_pointFunction);
  cudaCheckErrors("Kernel");
}
void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  func_two_op h_pointFunction;
  getFuncHostPtr(e_Power,h_pointFunction);

  ScalarEwiseOptKernel<<<dim.grid, dim.block>>>(a.ptr,val, out->ptr, out->size,h_pointFunction);
  cudaCheckErrors("Kernel");
}
void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  func_two_op h_pointFunction;
  getFuncHostPtr(e_Max,h_pointFunction);

  EwiseOptKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size,h_pointFunction);
  cudaCheckErrors("Kernel");
}
void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  func_two_op h_pointFunction;
  getFuncHostPtr(e_Max,h_pointFunction);

  ScalarEwiseOptKernel<<<dim.grid, dim.block>>>(a.ptr,val, out->ptr, out->size,h_pointFunction);
  cudaCheckErrors("Kernel");
}
void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  func_two_op h_pointFunction;
  getFuncHostPtr(e_Eq,h_pointFunction);

  EwiseOptKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size,h_pointFunction);
  cudaCheckErrors("Kernel");
}
void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  func_two_op h_pointFunction;
  getFuncHostPtr(e_Eq,h_pointFunction);

  ScalarEwiseOptKernel<<<dim.grid, dim.block>>>(a.ptr,val, out->ptr, out->size,h_pointFunction);
  cudaCheckErrors("Kernel");
}
void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  func_two_op h_pointFunction;
  getFuncHostPtr(e_Ge,h_pointFunction);

  EwiseOptKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size,h_pointFunction);
  cudaCheckErrors("Kernel");
}
void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  func_two_op h_pointFunction;
  getFuncHostPtr(e_Ge,h_pointFunction);

  ScalarEwiseOptKernel<<<dim.grid, dim.block>>>(a.ptr,val, out->ptr, out->size,h_pointFunction);
  cudaCheckErrors("Kernel");
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  func_two_op h_pointFunction;
  getFuncHostPtr(e_Log,h_pointFunction);

  SingleEwiseOptKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size,(func_one_op)h_pointFunction);
  cudaCheckErrors("Kernel");
}
void EwiseExp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  func_two_op h_pointFunction;
  getFuncHostPtr(e_Exp,h_pointFunction);

  SingleEwiseOptKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size,(func_one_op)h_pointFunction);
  cudaCheckErrors("Kernel");
}
void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);

  func_two_op h_pointFunction;
  getFuncHostPtr(e_Tann,h_pointFunction);

  SingleEwiseOptKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size,(func_one_op)h_pointFunction);
  cudaCheckErrors("Kernel");
}
////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE=32;

template <const int BM,const int BN,const int BK,const int RowoffsetAs,const int RowoffsetBs>
__device__ void LoadSharedMEM(scalar_t* As,scalar_t* Bs,const int  RowLAs,const int ColLAs,const int RowLBs,const int ColLBs,int M,int N,int K,scalar_t* A,scalar_t* B,const int starposArow,const int starposAcol,const int starposBrow,const int starposBcol){
  for(int i=0;i<BM;i+=RowoffsetAs){
    if(starposArow+(RowLAs+i)<M&&starposAcol+ColLAs<K){
        As[(ColLAs)*BM+RowLAs+i]=(A+(RowLAs+i)*K+ColLAs)[0];
    }
    else{
      As[(ColLAs)*BM+RowLAs+i]=0;
    }
  }
  for(int i=0;i<BK;i+=RowoffsetBs){
    if(starposBrow+(RowLBs+i)<K&&starposBcol+ColLBs<N){
      (Bs+(RowLBs+i)*BN+ColLBs)[0]=(B+(RowLBs+i)*N+ColLBs)[0];
    }
    else{
      (Bs+(RowLBs+i)*BN+ColLBs)[0]=0;
    }
  }
}

template <const int WN,const int WM,const int WNsuboffset,const int WMsuboffset,const int WNITER,const int WMITER,const int BM,const int BK,const int BN,const int TN,const int TM>
__device__ void CalcThreadRes(scalar_t* As,scalar_t* Bs,scalar_t*regN,scalar_t* regM,scalar_t* threadRes,const int warpRow,const int warpCol,const int threadInWarpRow,const int threadInWarpCol){
    for(int doidx=0;doidx<BK;doidx++){
      //load regM,regN
      for(int i=0;i<WMITER;i++){
        for(int j=0;j<TM;j++){
          regM[i*TM+j]=As[(doidx)*BM+warpRow*WM+i*WMsuboffset+threadInWarpRow*TM+j];
        }
      }
      for(int i=0;i<WNITER;i++){
        for(int j=0;j<TN;j++){
          regN[i*TN+j]=Bs[(doidx)*BN+warpCol*WN+i*WNsuboffset+threadInWarpCol*TN+j];
        }
      }
      for(int i=0;i<WMITER;i++){
        for(int j=0;j<WNITER;j++){
          for(int k=0;k<TM;k++){
            for(int l=0;l<TN;l++){
              threadRes[(i*TM+k)*(TN*WNITER)+j*TN+l]+=regM[i*TM+k]*regN[j*TN+l];
            }
          }
        }
      }
    }
}

template<const int BK,const int BM,const int BN,const int WM,const int WN,const int TM,const int TN,const int WNITER,const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
matmul_warptiling(scalar_t* A,scalar_t* B,scalar_t* C,uint32_t M, uint32_t N, uint32_t K) {
    const uint Rowblock=blockIdx.y;
    const uint ColBlock=blockIdx.x;

    // load shared memory
    __shared__ scalar_t As[BM*BK];
    __shared__ scalar_t Bs[BN*BK];
    const uint RowLAs=threadIdx.x/(BK);
    const uint ColLAs=threadIdx.x%(BK);
    constexpr int RowoffsetAs=NUM_THREADS/BK;
    const uint RowLBs=threadIdx.x/(BN);
    const uint ColLBs=threadIdx.x%(BN);
    constexpr int RowoffsetBs=NUM_THREADS/BN;

    // warptiling calculate
    const uint warpId=threadIdx.x/WARPSIZE;
    const uint warpRow=warpId/(BN/WN);
    const uint warpCol=warpId%(BN/WN);
    constexpr uint WMITER=WM*WN/(WARPSIZE*TM*TN*WNITER);
    constexpr uint WNsuboffset=WN/WNITER;
    constexpr uint WMsuboffset=WM/WMITER;
    scalar_t regN[WNITER*TN]={0.0};
    scalar_t regM[WMITER*TM]={0.0};
    scalar_t threadRes[WMITER*WNITER*TM*TN]={0.0};
    const uint threadInWarpId=threadIdx.x%warpSize;
    const uint threadInWarpRow=threadInWarpId/(WNsuboffset/TN);
    const uint threadInWarpCol=threadInWarpId%(WNsuboffset/TN);
    
    A+=Rowblock*BM*K;
    B+=ColBlock*BN;
    // C+=(Rowblock*BM+warpRow*WM)*N+ColBlock*BN+warpCol*WN;
    int starposArow=Rowblock*BM;
    int starposAcol=0;
    int starposBrow=0;
    int starposBcol=ColBlock*BN;
    const int starposCRow=(Rowblock*BM+warpRow*WM);
    const int starposCCol=ColBlock*BN+warpCol*WN;

    for(uint idx=0;idx<K; idx+=BK){

      LoadSharedMEM<BM,BN,BK,RowoffsetAs,RowoffsetBs>(As,Bs,RowLAs,ColLAs,RowLBs,ColLBs,M,N,K,A,B,starposArow,starposAcol,starposBrow,starposBcol);
      __syncthreads();
      CalcThreadRes<WN,WM,WNsuboffset,WMsuboffset,WNITER,WMITER,BM,BK,BN,TN,TM>(As,Bs,regN,regM,threadRes,warpRow,warpCol,threadInWarpRow,threadInWarpCol);
      A+=BK;
      B+=BK*N;
      starposAcol+=BK;
      starposBrow+=BK;
      __syncthreads();
    }
    // load resualt to C
    for(int i=0;i<WMITER;i++){
        for(int j=0;j<WNITER;j++){
          int C_interim_pos_row = starposCRow+(i*WMsuboffset);
          int C_interim_pos_col= starposCCol+j*WNsuboffset;
          for(int k=0;k<TM;k++){
            for(int l=0;l<TN;l++){
              int nowPos_row=C_interim_pos_row+(threadInWarpRow*TM+k);
              int nowPos_col=C_interim_pos_col+threadInWarpCol*TN+l;
              if(nowPos_row<M&&nowPos_col<N){
                C[nowPos_row*N+nowPos_col]=threadRes[(i*TM+k)*(TN*WNITER)+j*TN+l];
              }
            }
          }
        }
    }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
    // TODO: parameter tuning
    const uint Matmul_NUM_THREADS=128;
    const uint BK=16;
    const uint BM=64;
    const uint BN=64;
    const uint WM=32;
    const uint WN=32;
    const uint TM=4;
    const uint TN=2;
    const uint WNITER=4;

    dim3 blockDim(Matmul_NUM_THREADS);
    dim3 gridDim(CEIL_DIV(P,BN),CEIL_DIV(M,BM));
    // sgemmWarptiling<BM,BN,BK,WM,WN,WNITER,TM,TN,Matmul_NUM_THREADS>
    // <<<gridDim,blockDim>>>(M,P,N,1.0,a.ptr,b.ptr,0,out->ptr);
    matmul_warptiling<BK,BM,BN,WM,WN,TM,TN,WNITER,Matmul_NUM_THREADS>
    <<<gridDim,blockDim>>>(a.ptr,b.ptr,out->ptr,M,P,N);
    cudaCheckErrors("kernel");
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ void reduce_ws(float *a, float *out,size_t reduce_size,T f,const float init_val=0){
  __shared__ float sdata[32];
  int tid=threadIdx.x;
  float val=init_val;
  unsigned FULL_MASK=0xFFFFFFFFU;
  int lane=threadIdx.x%warpSize;
  int warpID=threadIdx.x/warpSize;

  int temp_tid=tid;
  while(temp_tid<reduce_size){
    val=f(val,*(a+blockIdx.x*reduce_size+temp_tid));
    temp_tid+=blockDim.x;
  }
  for(int offset=warpSize/2;offset>0;offset>>=1)
    val=f(val,__shfl_down_sync(FULL_MASK, val, offset));
  if(lane==0){
    sdata[warpID]=val;
  }
  __syncthreads();
  if(warpID==0){
    val=(tid<(blockDim.x+warpSize-1)/warpSize)?sdata[lane]:init_val;
    for(int offset=warpSize/2;offset>0;offset>>=1){
      val=f(val,__shfl_down_sync(FULL_MASK,val,offset));
    }
    if(tid==0){
      out[blockIdx.x]=val;
    }
  }
}

CudaDims CudaDimReduce(size_t out_size,size_t re_ducesize) {
  CudaDims dim;
  size_t block_size = (size_t)BASE_THREAD_NUM;
  block_size=(block_size + 31) & (~31);
  size_t num_blocks = out_size;
  dim.block = dim3(block_size, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaDimReduce(out->size,reduce_size);
  func_two_op h_pointFunction;
  getFuncHostPtr(e_Max,h_pointFunction);
  reduce_ws<<<dim.grid, dim.block>>>(a.ptr, out->ptr,reduce_size,h_pointFunction,-std::numeric_limits<float>::infinity());
  cudaCheckErrors("kernel");
  /// END SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t size) {
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
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    cudaCheckErrors("memcpy");
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}


#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


#define BLOCK_SIZE 256
#define TILE_WIDTH 16
#define CUDA_MAX_NUM_THREADS 1024
//__constant__ float maskCache[];

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    /* Your code here! */
    // declaring threads

    int W_grid = (W_out-1)/(TILE_WIDTH) + 1;
    int n,m,h,w,c,p,q;
    float acc = 0;

    n = blockIdx.x;
    m = blockIdx.y;
    h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;

    if( n < B && m < M && h < H_out && w < W_out){
        for( c = 0; c < C; c++){
            for( p = 0; p < K; p++){
                for( q = 0; q < K; q++){
                    acc += x4d(n,c,h+p,w+q) * k4d(m,c,p,q);
                }
            }
        }
    y4d(n,m,h,w) = acc;
    }

    #undef y4d
    #undef x4d
    #undef k4d
}

__global__ void forward_kernel_shared_convolution(float *y, const float *x, const float *Mask, const int B, const int M, const int C, const int H, const int W, const int K) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = (W_out - 1)/TILE_WIDTH + 1;
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) Mask[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    /* Your code here! */
    // declaring threads

    extern __shared__  float shmem[];
    const int X_TILE_WIDTH = TILE_WIDTH + K - 1;

    int n, m, h0, w0, h_base, w_base, h, w;
    int c, i, j, p, q;
    
    
    float * X_shared = &shmem[0];
    float * W_shared = &shmem[X_TILE_WIDTH * X_TILE_WIDTH];
    float acc = 0.00f;

    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.y;
    w0 = threadIdx.x;
    h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH;

    h = h_base + h0;
    w = w_base + w0;
    for( c = 0; c < C; c++){
        if((h0 < K) && (w0 < K))
            W_shared[ h0 * K + w0] = k4d(m,c,h0,w0);
        __syncthreads();

        for( i = h; i < h_base + X_TILE_WIDTH; i += TILE_WIDTH){
            for( j = w; j < w_base + X_TILE_WIDTH; j += TILE_WIDTH){

                if(i - h_base < X_TILE_WIDTH && j - w_base < X_TILE_WIDTH)
                    X_shared[ (i - h_base) * X_TILE_WIDTH + (j - w_base) ] = x4d(n,c,i,j);
            }
        }
        __syncthreads();

        for( p = 0; p < K; p++){
            for(q = 0; q < K; q++){
                if(h0 + p < X_TILE_WIDTH && w0 + q < X_TILE_WIDTH)
                   acc += X_shared[(h0 + p)* X_TILE_WIDTH + (w0 + q) ] * W_shared[q + p * K];
            }
        }

        __syncthreads();
    }
    if(n < B && m < M && h < H_out && w < W_out)
        y4d(n,m,h,w) = acc;

    #undef y4d
    #undef x4d
    #undef k4d
}





/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/

/****************************************************** SHARED MEMORY CONVOLUTION ****************
template<>
void forward<gpu, float>( mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    
    // inputs = y, x, w
    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...

    int B = x.shape_[0];
    int M = y.shape_[1];
    int C = x.shape_[1];
    int H = x.shape_[2];
    int W = x.shape_[3];
    int K = w.shape_[3];

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    int W_grid = (W_out - 1)/TILE_WIDTH + 1;
    int H_grid = (H_out - 1)/TILE_WIDTH + 1;
    int Z = H_grid *W_grid;

    std::cout << "B " << B << "\n";
    std::cout << "M " << M << "\n";
    std::cout << "C " << C << "\n";
    std::cout << "K " << K << "\n";

    // Set the kernel dimensions
    size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1 ) + K * K);

    dim3 gridDim(B,M,Z);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);

    // Call the kernel
    forward_kernel_shared_convolution<<<gridDim, blockDim, shmem_size, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}
**************************************************************************************************/
__global__ void gemm_kernel(float *Y, float *X_unrolled, const float *W_dash, int H_unroll, int M, int W_unroll, int K,int n, int H_out, int W_out) {

    __shared__ float subTileW[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileX[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    //What row and column should we work on!?
    
    int Row = by*TILE_WIDTH+ty;
    int Col = bx*TILE_WIDTH+tx;
    
    float PValue = 0;
  
    //Loop over the tiles for A and B
    //Assumes TILE_WIDTH / Width of A and Width of B
    int numWColumns = K * K;  //W_unroll; // C*K*K
    int numWRows = M; 
    int numXColumns = 24 * 24;
    int numXRows = K * K;
    int numYColumns = 24 * 24;
    int numYRows = M;
  
    for(int m = 0; m < (TILE_WIDTH + numWColumns -1)/TILE_WIDTH; m++){
      //Load A and B tiles into shared memory
      if(m*TILE_WIDTH + tx < numWColumns && Row < numWRows){
        subTileW[ty][tx] = W_dash[Row*numWColumns+ m*TILE_WIDTH+tx];
      }
      else{
        subTileW[ty][tx] = 0.0f;
      }
      if(m*TILE_WIDTH + ty < numXRows && Col < numXColumns){
        subTileX[ty][tx] = X_unrolled[(m*TILE_WIDTH+ty)*numXColumns+ Col];
      }
      else{
        subTileX[ty][tx] = 0.0f;
      }
      __syncthreads();
      for(int k = 0; k < TILE_WIDTH; ++k){
        PValue += subTileW[ty][k] * subTileX[k][tx];
      }
      __syncthreads();    
    }
    if(Row < numYRows && Col < numYColumns)
      Y[Row*numYColumns+Col] = PValue;

}


__global__ void unroll_kernel(int C, int H, int W, int K,int N, const float* X, float* X_unroll)
{
    #define x4d(i3,i2,i1,i0) X[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]

    int c, s, t, n , p, q;  
    int h_out, w_out, h_unroll,w_unroll;// , w_base;
    int H_out, W_out, W_unroll;

    t = blockIdx.x * CUDA_MAX_NUM_THREADS + threadIdx.x;
    
    n = blockIdx.x;

    H_out = H - K + 1;
    W_out = W - K + 1;
    W_unroll = H_out * W_out;
    #define x2d(i1,i0) X_unroll[(i1)*(W_unroll) + i0]

    

    if (t < C * W_unroll) 
    {
        c = t / W_unroll;
        s = t % W_unroll;
        h_out = s / W_out;
        w_out = s % W_out;

        h_unroll = h_out * W_out + w_out;
        /* W_base is always 0 because there is only 1 input feature thus 0 * k * k = 0 */
        //w_base = c * K * K;
        //printf("w_base : %d ",w_base);
        for(p = 0; p < K; p++){
            for(q = 0; q < K; q++) 
            {
                w_unroll = p * K + q;//w_unroll = w_base + p * K + q;
                x2d(h_unroll,w_unroll) = x4d(n, c, h_out + p, w_out + q);
            }
        }
    }
    #undef x4d
    #undef x2d
}
/*
void unroll( int C, int H, int W, int K, int n, const float *x, float *x_unrolled,cudaStream_t s){
    
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int num_threads = C * H_out * W_out;
    int num_blocks = (num_threads - 1) / CUDA_MAX_NUM_THREADS + 1;
    dim3 gridDim(num_blocks,1,1);
    dim3 blockDim(CUDA_MAX_NUM_THREADS,1,1);


    unroll_kernel<<<gridDim,blockDim,0,s>>>(C,H,W,K,n,x, x_unrolled);

}
*/

/* GEMM HOST CODE */ /*
void gemm(float *Y, float *X_unrolled, const float *W_dash, int H_unroll, int M, int W_unroll, int K,int n, int H_out,int W_out,  cudaStream_t s){
    int numYColumns = H_unroll;
    int numYRows = M;
    dim3 dimGrid(numYColumns/TILE_WIDTH, numYRows/TILE_WIDTH, 1);
    if(numYColumns%TILE_WIDTH){
      dimGrid.x++;
    }
    if(numYRows%TILE_WIDTH)
      dimGrid.y++;
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH, 1);

    gemm_kernel<<<dimGrid, dimBlock,0,s>>>(Y, X_unrolled, W_dash, H_unroll, M, W_unroll, K,n,H_out,W_out);
}
*/
template<>
void forward<gpu, float>( mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    // inputs = y, x, w
    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...

    const int N = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H - K + 1;        // 24
    const int W_out = W - K + 1;        // 24

    const int W_unroll = C*K*K;         // 1 x 5 x 5 = 25

    const int H_unroll = W_out * H_out; // 24 * 24

    float * DX_unrolled;
    /* params for unrolled */
    const int num_threads_unroll = C * H_out * W_out;
    const int num_blocks_unroll = (num_threads_unroll - 1) / CUDA_MAX_NUM_THREADS + 1;

    cudaMalloc( (void**) &DX_unrolled, W_unroll * H_unroll * sizeof(float));

 
    dim3 gridDim_unroll(num_blocks_unroll,1,1);
    dim3 blockDim_unroll(CUDA_MAX_NUM_THREADS,1,1);

    /* params for gemm */
    int numYColumns = H_unroll;
    int numYRows = M;
    dim3 dimGrid_gemm( (numYColumns - 1)/TILE_WIDTH + 1, (numYRows - 1)/TILE_WIDTH + 1, 1);
    dim3 dimBlock_gemm(TILE_WIDTH,TILE_WIDTH, 1);

    std::cout << "Batch size " << N << "\n";        // 10,000
    std::cout << "Output Features " << M << "\n";   // 50
    std::cout << "input features  " << C << "\n";   // 1
    std::cout << "Mask Width " << K << "\n";        // 5
   


    for(int n = 0; n < N; n++){
        //std::cout << "N :"<< n << "\n";

        //std::cout << "unroll\n";
        //unroll(C,H,W,K,n,x.dptr_,DX_unrolled,s);
        unroll_kernel<<<gridDim_unroll,blockDim_unroll,0,s>>>(C,H,W,K,n,x.dptr_, DX_unrolled);
        gemm_kernel<<<dimGrid_gemm, dimBlock_gemm,0,s>>>(y.dptr_, DX_unrolled, w.dptr_, H_unroll, M, W_unroll, K,n,H_out,W_out);
        //std::cout << "GEMM\n";
        //gemm(y.dptr_, DX_unrolled, w.dptr_, H_unroll,M, W_unroll, K,n,H_out,W_out,s);
        //std::cout << "DONE";

    }

    cudaFree(DX_unrolled);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    

}
/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif
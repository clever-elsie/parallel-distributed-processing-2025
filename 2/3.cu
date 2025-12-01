#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <string_view>
#include <iostream>

__global__ void MatrixAdd_col(float*A,float*B,float*C){
  const int width=gridDim.x*blockDim.x;
  int y=blockIdx.x*blockDim.x+threadIdx.x;
  for(int x=0;x<width;++x){
    const int index=width*y+x;
    C[index]=A[index]+B[index];
  }
}

__global__ void MatrixAdd_row(float*A,float*B,float*C){
  const int height=gridDim.x*blockDim.x;
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  for(int y=0;y<height;++y){
    const int index=y*height+x;
    C[index]=A[index]+B[index];
  }
}

int which(int argc, char**argv){
  if(argc!=2) exit(1);
  if(std::string_view("before")==argv[1]) return 0;
  else if(std::string_view("after")==argv[1]) return 1;
  exit(1);
}

int main(int argc, char**argv){
  const int type=which(argc, argv);
  constexpr int n=10'000*10'000;
  dim3 block(10), thread(1000);
  float *m[3], *dm[3];
  for(int i=0;i<3;++i){
    m[i]=new float[n];
    cudaMalloc(&dm[i], n*sizeof(float));
    if(i<2){
      for(int j=0;j<n;++j)
        m[i][j]=1;
      cudaMemcpy(dm[i], m[i], n*sizeof(float), cudaMemcpyHostToDevice);
    }
  }
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  if(type==0) MatrixAdd_col<<<block, thread>>>(dm[0], dm[1], dm[2]);
  else if(type==1) MatrixAdd_row<<<block, thread>>>(dm[0], dm[1], dm[2]);
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaMemcpy(m[2], dm[2], n*sizeof(float), cudaMemcpyDeviceToHost);
  float time;
  cudaEventElapsedTime(&time, start, end);
  std::cout<<"Time: "<<time<<" ms"<<std::endl;
  for(int i=0;i<3;++i){
    cudaFree(dm[i]);
    delete[] m[i];
  }
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  return 0;
}
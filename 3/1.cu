#include <iostream>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include "0.cuh"
#define MSIZE 2048

using namespace std;


template<size_t N>
__global__ void MatrixTranspose(int matrix[][N]){
  __shared__ int s_matrix[2][32][33];
  const int X=blockIdx.x*blockDim.x, Y=blockIdx.y*blockDim.y;
  const int x=threadIdx.x, y=threadIdx.y;
  if(X<Y) return; // 下三角成分はスキップ
  if(X==Y){ // 対角成分
    s_matrix[0][x][y]=matrix[Y+y][X+x];
    __syncthreads();
    matrix[Y+y][X+x]=s_matrix[0][y][x];
  }else{ // 上三角成分
    s_matrix[0][x][y]=matrix[Y+y][X+x];
    s_matrix[1][x][y]=matrix[X+y][Y+x];
    __syncthreads();
    matrix[Y+y][X+x]=s_matrix[1][y][x];
    matrix[X+y][Y+x]=s_matrix[0][y][x];
  }
}

int main(){
  vector<int> matrix(MSIZE*MSIZE);
  cuda_memory<int> g_matrix(MSIZE*MSIZE);
  for(int i=0;i<MSIZE;++i)
    for(int j=0;j<MSIZE;++j)
      matrix[i*MSIZE+j]=i*MSIZE+j;

  g_matrix.push(matrix);
  dim3 block(MSIZE/32, MSIZE/32), thread(32, 32);
  elapsed_time et;
  cout<<"Time: "<<et({block, thread}, MatrixTranspose<MSIZE>, (int(*)[MSIZE])(int*)g_matrix)<<" ms"<<endl;
  g_matrix.pop(matrix);
  
  bool is_ok=true;
  for(int i=0;i<MSIZE&&is_ok;++i)
    for(int j=0;j<MSIZE;++j)
      if(matrix[i*MSIZE+j]!=j*MSIZE+i){
        is_ok=false;
        break;
      }
  cout<<(is_ok?"Valid":"Invalid")<<endl;

  return 0;
}
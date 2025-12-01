#ifndef HELPER_CUH
#define HELPER_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <vector>
#include <future>

template<class T>
class cuda_memory{
  T*data;
  size_t size;
public:
  cuda_memory(size_t size):size(size){ cudaMalloc(&data, size*sizeof(T)); }
  ~cuda_memory(){ cudaFree(data); }
  operator T*() {return data;}
  size_t bytes() const { return size*sizeof(T); }
  void push(const std::vector<T>&v){
    cudaMemcpy(data, v.data(), bytes(), cudaMemcpyHostToDevice);
  }
  void push(const T&v){
    if(size!=1) throw std::runtime_error("size!=1");
    cudaMemcpy(data, &v, sizeof(T), cudaMemcpyHostToDevice);
  }
  void set(const T&v){
    cudaMemset(data, v, bytes());
  }
  void pop(std::vector<T>&v){
    if(v.size()<size) v.resize(size);
    cudaMemcpy(v.data(), data, bytes(), cudaMemcpyDeviceToHost);
  }
  void pop(T&v){
    if(size!=1) throw std::runtime_error("size!=1");
    cudaMemcpy(&v, data, sizeof(T), cudaMemcpyDeviceToHost);
  }
  T* ptr(){ return data; }
};

class elapsed_time{
  cudaEvent_t start, end;
public:
  struct cuda_call_args{
    dim3 grid, block;
    size_t shared_memory;
    cudaStream_t stream;
    cuda_call_args(dim3 grid, dim3 block):grid(grid), block(block), shared_memory(0), stream(cudaStreamDefault){}
    cuda_call_args(dim3 grid, dim3 block, size_t shared_memory, cudaStream_t stream):grid(grid), block(block), shared_memory(shared_memory), stream(stream){}
  };
  elapsed_time(){ cudaEventCreate(&start); cudaEventCreate(&end); }
  ~elapsed_time(){ cudaEventDestroy(start); cudaEventDestroy(end); }
  template<class F, class... Args>
  float operator()(cuda_call_args cargs, F f, Args... args){
    cudaEventRecord(start);
    f<<<cargs.grid, cargs.block, cargs.shared_memory, cargs.stream>>>(args...);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    return time;
  }
  template<class F, class... Args>
  std::future<float> get_future(cuda_call_args cargs, F f, Args... args){
    std::promise<float> p;
    std::future<float> r=p.get_future();
    std::thread([](elapsed_time& obj,std::promise<float> p, cuda_call_args cargs, F f, Args... args){
      p.set_value(obj(cargs, f, args...));
    }, std::ref(*this), std::move(p), cargs, f, args...).detach();
    return r;
  }
};
#endif
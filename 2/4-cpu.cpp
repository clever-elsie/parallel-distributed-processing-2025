#include <iostream>
#include <array>
#include <chrono>
#include <random>
#include <memory>
using namespace std;
using namespace std::chrono;

constexpr int n=1024;
void Copy(float*A,float*B){
  for(int i=0;i<n*n;++i)
    B[i]=A[i];
}

void Scala(float*A,float*B,float scala){
  for(int i=0;i<n*n;++i)
    B[i]=A[i]*scala;
}

void Add(float*A,float*B,float*C){
  for(int i=0;i<n*n;++i)
    C[i]=A[i]+B[i];
}

void Triad(float*A,float*B,float*C,float scala){
  for(int i=0;i<n*n;++i)
    C[i]=A[i]*scala+B[i];
}

template<class F, class... Args>
void check(const char* name, F f, Args... args){
  auto start=high_resolution_clock::now();
  f(args...);
  auto end=high_resolution_clock::now();
  cout<<name<<": "<<duration_cast<microseconds>(end-start).count()<<" us"<<endl;
}

int main(){
  unique_ptr<float[]> a(new float[n*n]), b(new float[n*n]), c(new float[n*n]);
  uniform_real_distribution<float> dist(0, 1);
  default_random_engine eng(random_device{}());
  for(int i=0;i<n*n;++i){
    a[i]=dist(eng);
    b[i]=dist(eng);
    c[i]=dist(eng);
  }
  check("Copy", Copy, a.get(), b.get());
  check("Scala", Scala, a.get(), b.get(), 1.6f);
  check("Add", Add, a.get(), b.get(), c.get());
  check("Triad", Triad, a.get(), b.get(), c.get(), 2.0f);
}

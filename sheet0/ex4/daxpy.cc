#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "time_experiment.hh"

using NUMBER=double;

const int P = 8;
const int N=32;//*1024*1024;     // problem size
std::vector<NUMBER> x(N,2.0); // first vector
std::vector<NUMBER> y(N,1.0); // second vector
double alpha = 0.3;
std::vector<NUMBER> sums(P);  // sum of each thread
std::vector<int> flags(P,0);   // flags[i]==1 signals that thread i provides the result
std::vector<std::mutex> ms(P); // mutexes
std::vector<std::condition_variable> cvs(P); // and condition variables

// scalar product with lock-free sumation
void f (int rank)
{
  for (int i=(N*rank)/P; i<(N*(rank+1))/P; ++i){
    y[i] += alpha * x[i];
  }
}

int main ()
{
  std::vector<std::thread> threads;
  for (int rank=0; rank<P; ++rank)
    threads.push_back(std::thread{f,rank});
  for (int rank=0; rank<P; ++rank)
    threads[rank].join();
  
  std::cout << "y = " << std::endl;
  for (auto item : y)
    std::cout << item <<" ";
}

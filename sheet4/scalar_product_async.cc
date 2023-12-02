#include <iostream>
#include <vector>
#include <thread>
#include <future>

#include "time_experiment.hh"

using NUMBER = double;

const int N = 32 * 1024 * 1024; // problem size
std::vector<NUMBER> x(N, 1.0);  // first vector
std::vector<NUMBER> y(N, 1.0);  // second vector
NUMBER sum = 0.0;               // result

// the scalarproduct
NUMBER f(int idx_start, int idx_end) // for simplicity arguments and result are global
{
    int sum = 0;
    for (int i = idx_start; i < idx_end; ++i)
        sum += x[i] * y[i];
    return sum;
}

// package an experiment as a functor
class Experiment
{
    int n;

public:
    // construct an experiment
    Experiment(int n_) : n(n_) {}
    // run an experiment; can be called several times
    void run() const
    {
        sum = 0;
        int num_threads = 14;
        int idx = 0;
        int step = n/num_threads;
        std::vector<std::future<NUMBER>> sum_vec;
        for(int i=0; i < num_threads-1; ++i){
            sum_vec.push_back(std::async(f,idx, idx + step));
            idx += step;
        }
        sum_vec.push_back(std::async(f,idx, n));

        for(int i=0; i < num_threads; ++i){
            sum += sum_vec[i].get();
        }
        // std::cout<<"scalar product of "<<n<<" is "<< sum<<std::endl;
    }
    // report number of operations
    double operations() const { return 2.0 * n; }
};

int main()
{
    std::cout << N * sizeof(NUMBER) / 1024 / 1024 << " MByte per vector" << std::endl;
    std::vector<int> sizes = {256, 1024, 4096, 16384, 65536, 262144, 1048576, 4 * 1048576, 16 * 1048576, 32 * 1048576};
    for (auto i : sizes)
    {
        Experiment e(i);
        auto d = time_experiment(e);
        double flops = d.first * e.operations() / d.second * 1e6 / 1e9;
        std::cout << "n=" << i << " took " << d.second << " us for " << d.first << " repetitions"
                  << " " << flops << " Gflops/s"
                  << " " << flops * sizeof(NUMBER) << " GByte/s" << std::endl;
    }
    return 0;
}

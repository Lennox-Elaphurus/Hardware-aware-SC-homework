#include <iostream>
#include <cmath>
#include <vector>
#define VCL_NAMESPACE VCL
#include "../../src/vcl/vectorclass.h"
#include "../../src/time_experiment.hh"
#include <fstream>

using VecType = VCL::Vec4d; // Vec4d

#define LEN_VEC 4           // number of elements in vector, corresponding to VecType
#define ElEMENT_TYPE double // element type of the vector, corresponding to VecType

using std::cout, std::endl;

class Function
{
public:
    virtual double eval(double x) = 0;
};

class F1 : public Function
{
public:
    double eval(double x) override
    {
        return pow(x, 3) - 2 * x * x + 3 * x - 1; // need to convert 3 to double, otherwise compiler confuses regular pow with vcl pow
    }
};

class F2 : public Function
{
public:
    double eval(double x) override
    {
        double sum = 0;
        for (int i = 0; i <= 15; ++i)
        {
            sum = sum + pow(x, i);
        }
        return sum;
    }
};

double int_midpoint(double a, double b, int n, Function *fPtr)
{
    double h = (b - a) / n;
    double sum = 0;
    double x_mid = 0;

    for (int i = 0; i < n; ++i)
    {
        x_mid = a + (i + 0.5) * h;
        sum += fPtr->eval(x_mid);
    }
    return sum * h;
}

class Function_vec
{
public:
    virtual VecType eval(VecType x) = 0;
    virtual double eval_single(double x) = 0;
    ~Function_vec() { delete fPtr; }

protected:
    Function *fPtr;
};

class F1_vec : public Function_vec
{
public:
    F1_vec()
    {
        fPtr = new F1;
    }
    VecType eval(VecType x) override
    {
        return VCL::pow_const(x, 3) - 2 * VCL::square(x) + 3 * x - 1;
    }

    double eval_single(double x) override { return fPtr->eval(x); }
};

class F2_vec : public Function_vec
{
public:
    F2_vec()
    {
        fPtr = new F2;
    }
    VecType eval(VecType x) override
    {
        VecType sum(0); // initialize all numbers with 0
        for (int i = 0; i <= 15; ++i)
        {
            sum = sum + VCL::pow(x, i);
        }
        return sum;
    }

    double eval_single(double x) override { return fPtr->eval(x); }
};

// function overload, differentiate by Function_vec or Function
double int_midpoint(double a, double b, int n, Function_vec *fVecPtr)
{
    double h = (b - a) / n;
    int turns = n / LEN_VEC;
    double sum = 0;
    VecType sum_vec(0);
    VecType x_mid_vec(0);

    ElEMENT_TYPE x_mid_arr[LEN_VEC] = {0}; // initialize all elements to 0
    double x_mid = 0;
    for (int i = 0; i < turns; ++i)
    {
        for (int j = 0; j < LEN_VEC; ++j)
        {
            x_mid_arr[j] = a + (i * LEN_VEC + j + 0.5) * h;
        }

        x_mid_vec.load(x_mid_arr);

        sum_vec += fVecPtr->eval(x_mid_vec);
    }
    sum = VCL::horizontal_add(sum_vec);

    for (int i = turns * LEN_VEC; i < n; ++i)
    {
        x_mid = a + (i + 0.5) * h;
        sum += fVecPtr->eval_single(x_mid);
    }
    return sum * h;
}

template <typename FunctionType>
// package an experiment as a functor
class Experiment
{
public:
    // construct an experiment
    Experiment(int num_intervals_, FunctionType *fPtr_) : num_intervals(num_intervals_), fPtr(fPtr_) {}
    // run an experiment; can be called several times
    void run() const { int_midpoint(0, 1000, num_intervals, fPtr); }
    // report number of operations
    double operations() const { return 0; } // calculate outside, not here because F can be different
private:
    int num_intervals;
    FunctionType *fPtr;
};

template <typename VectorType>
bool writeMat(VectorType vec, std::string filename)
{
    std::ofstream outFile(filename);
    if (outFile.is_open())
    {
        for (auto line : vec)
        {
            for (auto element : line)
            {
                outFile << element << " ";
            }
            outFile << endl;
        }
        outFile.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing." << std::endl;
        return false;
    }

    return true;
}
int main(void)
{
    F1 f1;
    F1_vec f1_vec;
    F2 f2;
    F2_vec f2_vec;
    // cout << int_midpoint(0, 1000, 50, f1) << endl;
    // cout << int_midpoint(0, 1000, 50, f1_vec) << endl;
    // cout << int_midpoint(1, 10, 100, f1) << endl;
    // cout << int_midpoint(1, 10, 100, f1_vec) << endl;

    // cout << int_midpoint(1, 10, 50, f2) << endl;
    // cout << int_midpoint(1, 10, 50, f2_vec) << endl;
    // cout << int_midpoint(1, 10, 100, f2) << endl;
    // cout << int_midpoint(1, 10, 100, f2_vec) << endl;
    std::vector<std::vector<double>> time_mat;
    std::vector<std::vector<double>> flops_mat;
    for (int i = 0; i < 25; ++i)
    {
        int n = pow(2, i);
        cout<<i<<" \t"<<n<<endl;
        int ops_f1 = n * 8 + 3 + (n - 1);
        int ops_f2 = n * 122 + 3 + (n - 1);
        Experiment e_f1(n, &f1);
        Experiment e_f1_vec(n, &f1_vec);
        Experiment e_f2(n, &f2);
        Experiment e_f2_vec(n, &f1);

        auto d1 = time_experiment(e_f1);
        auto d1_vec = time_experiment(e_f1_vec);
        auto d2 = time_experiment(e_f2);
        auto d2_vec = time_experiment(e_f2_vec);

        std::vector<double> time_list = {double(1.0 * d1.second / d1.first),
                                         double(1.0 * d1_vec.second / d1_vec.first),
                                         double(1.0 * d2.second / d2.first),
                                         double(1.0 * d2_vec.second / d2_vec.first)}; //us
        time_mat.push_back(time_list);

        std::vector<double> flops_list = {
            d1.first * ops_f1 / d1.second * 1e6 / 1e9,
            d1_vec.first * ops_f1 / d1_vec.second * 1e6 / 1e9,
            d2.first * ops_f2 / d2.second * 1e6 / 1e9,
            d2_vec.first * ops_f2 / d2_vec.second * 1e6 / 1e9,
        };
        flops_mat.push_back(flops_list);
    }

    writeMat(time_mat, "time_mat.txt");
    writeMat(flops_mat, "flops_mat.txt");
}
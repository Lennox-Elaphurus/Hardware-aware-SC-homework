#include <iostream>
#include <cmath>
#define VCL_NAMESPACE VCL
#include "../../vcl/vectorclass.h"

using VecType = VCL::Vec4d;
using VCL::pow;

#define LEN_VEC 4 // when changes, also need to change GEN_MULTI_PARA(LEN_VEC,v)

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
        return pow(x, double(3)) - 2 * x * x + 3 * x - 1; // need to convert 3 to double, otherwise compiler confuses regular pow with vcl pow
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
            sum = sum + pow(x, double(i));
        }
        return sum;
    }
};

double int_midpoint(double a, double b, int n, Function &fPtr)
{
    double h = (b - a) / n;
    double sum = 0;
    double x_mid = 0;

    for (int i = 0; i < n; ++i)
    {
        x_mid = a + (i + 0.5) * h;
        sum += fPtr.eval(x_mid);
    }
    return sum * h;
}

class Function_vec
{
public:
    virtual VecType eval(VecType x) = 0;
};

class F1_vec : public Function_vec
{
public:
    VecType eval(VecType x) override
    {
        return pow<int>(x, 3) - 2 * x * x + 3 * x - 1;
    }
};

class F2_vec : public Function_vec
{
public:
    VecType eval(VecType x) override
    {
        VecType exponent;
        VecType sum(0); // initialize all numbers with 0
        for (int i = 0; i <= 15; ++i)
        {
            sum = sum + pow<int>(x, i);
        }
        return sum;
    }
};

double int_midpoint_vec(double a, double b, int n, Function_vec &fPtr)
{
    double h = (b - a) / n;
    int turns = n / LEN_VEC;
    int residual = n % LEN_VEC;
    double sum = 0;
    VecType sum_vec(0);
    VecType x_mid_vec(0);
    double x_mid[LEN_VEC] = {0}; // initialize all elements to 0
    for (int i = 0; i < turns; ++i)
    {
        for (int j = 0; j < LEN_VEC; ++j)
        {
            x_mid[j] = a + (i * LEN_VEC + j + 0.5) * h;
        }

        x_mid_vec.load(x_mid);

        sum_vec += fPtr.eval(x_mid_vec);
    }

    for (int i = 0; i < LEN_VEC; ++i)
    {
        sum += sum_vec[i];
    }
    return sum * h;
}

int main(void)
{
    F1 f1;
    F1_vec f1_vec;
    F2 f2;
    cout << int_midpoint(1, 10, 100, f1) << endl;
    cout << int_midpoint_vec(1, 10, 100, f1_vec) << endl;
}
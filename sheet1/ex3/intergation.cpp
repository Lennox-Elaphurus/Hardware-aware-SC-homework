#include <iostream>
#include <cmath>
#define VCL_NAMESPACE VCL
#include "../../vcl/vectorclass.h"

using VecType = VCL::Vec4d;

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
        VecType exponent;
        VecType sum(0); // initialize all numbers with 0
        for (int i = 0; i <= 15; ++i)
        {
            sum = sum + VCL::pow(x, i);
        }
        return sum;
    }

    double eval_single(double x) override { return fPtr->eval(x); }
};

double int_midpoint_vec(double a, double b, int n, Function_vec &fVecPtr)
{
    double h = (b - a) / n;
    int turns = n / LEN_VEC;
    double sum = 0;
    VecType sum_vec(0);
    VecType x_mid_vec(0);
    double x_mid_arr[LEN_VEC] = {0}; // initialize all elements to 0
    double x_mid = 0;
    for (int i = 0; i < turns; ++i)
    {
        for (int j = 0; j < LEN_VEC; ++j)
        {
            x_mid_arr[j] = a + (i * LEN_VEC + j + 0.5) * h;
        }

        x_mid_vec.load(x_mid_arr);

        sum_vec += fVecPtr.eval(x_mid_vec);
    }
    sum = VCL::horizontal_add(sum_vec);

    for (int i = turns * LEN_VEC; i < n; ++i)
    {
        x_mid = a + (i + 0.5) * h;
        sum += fVecPtr.eval_single(x_mid);
    }
    return sum * h;
}

int main(void)
{
    F1 f1;
    F1_vec f1_vec;
    F2 f2;
    F2_vec f2_vec;
    cout << int_midpoint(1, 10, 50, f1) << endl;
    cout << int_midpoint_vec(1, 10, 50, f1_vec) << endl;
    cout << int_midpoint(1, 10, 100, f1) << endl;
    cout << int_midpoint_vec(1, 10, 100, f1_vec) << endl;

    cout << int_midpoint(1, 10, 50, f2) << endl;
    cout << int_midpoint_vec(1, 10, 50, f2_vec) << endl;
    cout << int_midpoint(1, 10, 100, f2) << endl;
    cout << int_midpoint_vec(1, 10, 100, f2_vec) << endl;
}
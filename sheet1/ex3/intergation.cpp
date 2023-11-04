#include <iostream>
#include <cmath>
#include "../../vcl/vectorclass.h"

#define GEN_MULTI_PARA(i,v) GEN_MULTI_PARA_##i(v)
#define GEN_MULTI_PARA_1(v) v
#define GEN_MULTI_PARA_4(v) v,v,v,v
#define GEN_MULTI_PARA_8(v) v,v,v,v,v,v,v,v
#define GEN_MULTI_PARA_16(v) v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v

using VecType  = Vec4d;
#define LEN_VEC 4 //when changes, also need to change GEN_MULTI_PARA(LEN_VEC,v)

using std::cout, std::endl;

template <typename DataType>
class Function
{
public:
    virtual DataType eval(DataType x) = 0;
};

class F1 : public Function<double>
{
public:
    double eval(double x) override
    {
        return pow(x, double(3)) - 2 * x * x + 3 * x - 1; // need to convert 3 to double, otherwise compiler confuses regular pow with vcl pow
    }
};

class F1_vec : public Function<VecType>
{
public:
    VecType eval(VecType x) override
    {
        // VecType exponent(GEN_MULTI_PARA(4,3)); // GEN_MULTI_PARA(4,3) means (3,3,3,3)
        return pow<int>(x, 3) - 2 * x * x + 3 * x - 1;
    }
};

class F2 : public Function<double>
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


class F2_vec : public Function<VecType>
{
public:
    VecType eval(VecType x) override
    {
        VecType exponent;
        VecType sum(GEN_MULTI_PARA(4,0)); // GEN_MULTI_PARA(4,0) means 4 zeros
        for (int i = 0; i <= 15; ++i)
        {
            // exponent = (GEN_MULTI_PARA(4,i));
            sum = sum + pow<int>(x, i);
        }
        return sum;
    }
};

template <typename DataType>
double int_midpoint(double a, double b, int n, Function<DataType> &fPtr)
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

template <typename DataType>
double int_midpoint_vec(double a, double b, int n, Function<DataType> &fPtr)
{
    double h = (b - a) / n;
    int turns = n / LEN_VEC;
    int residual = n % LEN_VEC;
    double sum = 0;
    VecType sum_vec(GEN_MULTI_PARA(4,0));
    VecType x_mid(GEN_MULTI_PARA(4,0));

    for (int i = 0; i < turns; ++i)
    {
        x_mid = a + (i + 0.5) * h;
        sum += fPtr.eval(x_mid);
    }
    return sum * h;
}

int main(void)
{
    F1_vec f1;
    F2 f2;
    auto result1 = int_midpoint(1, 3, 10, f1);
    for (int i = 0;i<result1.size();++i){
        cout << result1[i]<< endl;
    }
}
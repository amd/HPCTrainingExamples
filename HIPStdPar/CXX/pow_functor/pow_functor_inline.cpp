// This is from a question submitted by Jony Castagna from
// a problem encountered with OpenFOAM. Jony gave permission
// to use this as an example of handling similar cases
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <execution>

// ------------------------------------------------------------------
// This inline wrapper mimics OpenFOAM's pattern that breaks clang.
// It hides the builtin pow() behind another inline function.
// ------------------------------------------------------------------
inline double myPow(double base, double expon)
{
    // Calling ::pow() here prevents clang from recognizing the builtin.
    return ::pow(base, expon);
}

// ------------------------------------------------------------------
// A binary functor that uses myPow() in std::transform.
// ------------------------------------------------------------------
template<class T1, class T2, class R>
struct PowBinaryFunctionFunctor
{
    __host__ __device__
    R operator()(const T1& a, const T2& b) const
    {
        return myPow(a, b);
    }
};

// ------------------------------------------------------------------
// Main: launch parallel transform using std::execution::par_unseq
// ------------------------------------------------------------------
int main()
{
    std::vector<double> base = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> expo = {2.0, 3.0, 4.0, 0.5};
    std::vector<double> result(base.size(), 0.0);

    PowBinaryFunctionFunctor<double,double,double> functor;

    std::transform(std::execution::par_unseq,
                   base.begin(), base.end(),
                   expo.begin(),
                   result.begin(),
                   functor);

    for (auto v : result)
        std::cout << v << " ";
    std::cout << "\n";
}

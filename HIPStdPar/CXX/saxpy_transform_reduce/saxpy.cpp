#include <vector>
#include <algorithm>
#include <execution>

using namespace std;

int main(int argc, char *argv[])
{
   vector<double> x(1024, 1);

   double result = transform_reduce(
      execution::par_unseq, x.begin(), x.end(), 0.0, plus<>(), [&](const double& x) {
         return 5.0*x;
      }
   );

   printf("Finished Run: Result %lf\n",lf);
}

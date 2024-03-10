#include <vector>
#include <algorithm>
#include <execution>

using namespace std;

int main(int argc, char *argv[])
{
   vector<double> x(1024, 1);

   for_each(execution::par_unseq, x.begin(), x.end(), [&](const double& x) {
       x *= 5.0;
   });

   printf("Finished Run\n");
}

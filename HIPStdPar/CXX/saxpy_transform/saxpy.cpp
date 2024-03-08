#include <vector>
#include <algorithm>
#include <execution>

using namespace std;

int main(int argc, char *argv[])
{
   vector<double> x(1024, 1);

   transform(execution::par_unseq, x.begin(), x.end(), x.begin(), [&](double& x) {
       return 5.0*x;
   });

   printf("Finished Run\n");
}

#include <vector>
#include <execution>

using namespace std;

int main(int argc, char *argv[])
{
   vector<double> x(1024, 1);

   for_each(execution::par_unseq, x.begin(), x.end(), [&](double& x) {
       x *= 5.0;
   });

}

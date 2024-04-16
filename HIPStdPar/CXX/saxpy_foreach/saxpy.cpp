#include <vector>
#include <algorithm>
#include <execution>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
   vector<double> x(1024, 1);

   for_each(
      execution::par_unseq, x.begin(), x.end(), [](double& elem_x) {
         elem_x *= 5.0;
      }
   );

   std::cout << "Finished Run - x[10]: " << x[10] << std::endl;
}

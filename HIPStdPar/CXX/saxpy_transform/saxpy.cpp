#include <vector>
#include <algorithm>
#include <execution>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
   vector<double> x(1024, 1);

   transform(
      execution::par_unseq, x.begin(), x.end(), x.begin(), [](double elem_x) {
         return 5.0*elem_x;
      }
   );

   std::cout << "Finished Run - x[10]: " << x[10] << std::endl;
}

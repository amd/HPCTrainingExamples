#include <vector>
#include <execution>

using namespace std;

int main(int argc, char *argv[])
{
   vector<double> x(1024, 1);

   for_each(execution::par_unseq, begin(x), end(x), [&x](int i) {
       x[i] *= 5.0;
   });

}

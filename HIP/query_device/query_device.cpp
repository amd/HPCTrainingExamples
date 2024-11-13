#include "hip/hip_runtime.h"

bool GetDeviceMemoryInfo() {

  size_t free = 0;
  size_t total = 0;
  hipError_t res = hipMemGetInfo(&free, &total);
  if (res != hipSuccess) {
    printf("ERROR: failed to query device memory info \n");
    return false;
  }
  else{
      printf("SUCCESS: was able to query device memory info \n");
     return true;
  }
}

//////////////////////////////////////////////////////

int main(int argc, char *argv[]){

    GetDeviceMemoryInfo();	

    return 0;
}

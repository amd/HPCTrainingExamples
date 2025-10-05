
## Complete the square elements kernel

README.md in `HPCTrainingExamples/HIP/03_complete_square_elements` of the Training Exercises repository.

In this exercise, there is a host array and a device array. The host array is initialized in a loop so each element is given the value of the iteration from 0 to N-1. Then the host array is copied to the device array, and the GPU kernel simply squares each element of the array. Then the results are sent back from the device array to the host array.

However, the kernel is not complete. So you must complete the kernel by adding in the line where the value is squared, and make sure to guard for going out of the array bounds. Look for the TODO.

To compile and run:
```
$ make

$ sbatch -A <account-name> submit.sh
```
where `account-name` is your account name for the system (may be required for certain systems). A job file titled `<name-of-exercise>-%J.out` will be produced, where `%J` is the job id number of your run. To check your program output, simply run:
```
cat <name-of-exercise>-%J.out
```


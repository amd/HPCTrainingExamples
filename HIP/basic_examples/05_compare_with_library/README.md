
## Complete the matrix multiply kernel

README.md in `HPCTrainingExamples/HIP/05_compare_with_library` of the Training Exercises repository.

In this exercise, we will use the `matrix_multiply` kernel we completed in `04_complete_the_kernel` and compare its performance against the hipBLAS version of DGEMM. 

You will not need to make any code changes. Instead, you will simply compile the code and submit the job. This will run the code under the `rocprof` profiling tool and parse the results. 

To compile and run:
```
$ make

$ sbatch -A <account-name> submit.sh
```
where `account-name` is your account name for the system (may be required for certain systems). A job file titled `<name-of-exercise>-%J.out` will be produced, where `%J` is the job id number of your run. To check your program output, simply run:
```
cat <name-of-exercise>-%J.out
```

To view the resulting profile, run the python script:
```
./parse_output.py
```

It should be clear from the performance difference that using existing libraries is typically the right choice instead of re-inventing the (slower) wheel.

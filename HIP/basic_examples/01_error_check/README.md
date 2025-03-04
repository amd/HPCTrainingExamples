# Find the error

Compile and run the vector addition program and use the error from the error-checking macro to decide how to fix the problem.

To compile and run:
```
$ make

$ sbatch -A <account-name> submit.sh
```
where `account-name` is your account name for the system (may be required for certain systems). A job file titled `<name-of-exercise>-%J.out` will be produced, where `%J` is the job id number of your run. To check your program output, simply run:
```
cat <name-of-exercise>-%J.out
```



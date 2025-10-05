
## Complete the matrix multiply kernel

README.md in `HPCTrainingExamples/HIP//04_complete_matrix_multiply` of the Training Exercises repository.

In this exercise, a matrix multiply is performed on the GPU. In the code, the indices `row_index` and `col_index` iterate through the arrays in row-major (across the first row, then across the second row, etc.) and column-major (down the first column, then down the second column, etc.) order, respectively.

Look at the matrix multiply kernel and decide which of these two indices should define the elements of arrays A and B. Look for the TODO.

To compile and run:
```
$ make

$ sbatch -A <account-name> submit.sh
```
where `account-name` is your account name for the system (may be required for certain systems). A job file titled `<name-of-exercise>-%J.out` will be produced, where `%J` is the job id number of your run. To check your program output, simply run:
```
cat <name-of-exercise>-%J.out
```

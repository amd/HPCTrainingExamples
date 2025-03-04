# HIP/basic_examples Documentation

To connect to Frontier, a username will be provided. Once you have your assigned username, run the following to connect to the Frontier login nodes:
```
ssh <username>@frontier.olcf.ornl.gov
```
You will be prompted for a password. The password will be: 2325<RSA-TOKEN>, where the RSA token is provided and must be entered promptly. The token will be regenerated every ~15 seconds and must be entered during that time limit. If your token expires, you must use the newly generated token.

Once you are on Frontier, you can start going through the exercises. Before you start the exercises, you will need to load the ROCm environment module. To do this, simply run the following command:
```
module load rocm
```

Use the following command to verify that the ROCm environment module has been properly loaded:
```
module list
```

## Table of Contents

1. `01_error_check`
2. `02_add_d2h_data_transfer`
3. `03_complete_square_elements`
4. `04_complete_matrix_multiply`
5. `05_compare_with_library`
6. `06_hipify_pingpong`
7. `07_matrix_multiply_shared`

Please refer to the individual directories for documentation specific to each exercise.

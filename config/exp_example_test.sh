#!/bin/bash

# Set the CONFIGS_LOCATION environment variable if needed,
# else the default location(cwd) is used.
#export CONFIGS_LOCATION=/path/to/configs

# Execute the Python script, please use correct conda env.
python exp_example_test.py model=SAGE
#python exp_example_test.py -m +param=1,2,3  # Example of multirun

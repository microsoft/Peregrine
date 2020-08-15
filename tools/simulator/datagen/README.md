This directory contains the scripts for generating synthetic datasets that have similar distributions of feature values as an input dataset.

There are three main components for datagen and simulation:
1. *extract_inputs.py*: This determines the distributions of feature values in the input dataset.
2. *simulate_dataset.py*: This generates a synthetic dataset with similar distributions as those determined in the above step.
3. *validate.py*: This validates the distributions of the generated dataset with those of the input dataset.

The *datagen.sh* script is the top-level script that generates the synthetic datasets. By default, the script will simulate and validate a new dataset using distributions provided in the ../distributions directory. For a new reference dataset, uncomment the section corresponding to extract_inputs and set the values for the paths as needed.

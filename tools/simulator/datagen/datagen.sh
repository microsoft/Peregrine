#!/usr/bin/bash


##############################################################################
# Uncomment this section and set dist_dir as needed in the next section if
# a new reference dataset is used. 
#
# ref_dataset_csv=
# inputs_dir=
# dist_dir=
# max_jobs=1000
# S=10
# mkdir -p $inputs_dir
# mkdir -p $dist_dir
#
# echo "Extracting"
# python extract_inputs.py $ref_dataset_csv $inputs_dir $dist_dir $max_jobs $S
##############################################################################

##############################################################################
# The following settings assume that we are simulating datasets using the
# distributions provided. Any one of the traces Trace1 ... Trace5 can be used.
# mode can be training or testing.
#
# Example invocations: 
#       ./datagen.sh Trace3 training 1000
#       ./datagen.sh Trace1 testing 100
#
tr=$1
mode=$2
size_per_query=$3
dist_dir=../distributions/${tr}/${mode}_distributions
gen_dir=../distributions/${tr}/
sim_dir=${gen_dir}/${mode}_sim

mkdir -p $sim_dir
echo "Simulating"
python simulate_dataset.py $dist_dir $sim_dir ${gen_dir}/${mode}.csv $size_per_query
##############################################################################

echo "Validating"
python validate.py $dist_dir $sim_dir > ${gen_dir}/${mode}_KL.txt


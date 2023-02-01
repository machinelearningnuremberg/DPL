#!/bin/bash

export benchmark=$1

if [ "$benchmark" == "lcbench" ]
then
  file="lcbench_dataset_names.txt"
elif [ "$benchmark" == "taskset" ]
then
  file="taskset_dataset_names.txt"
elif [ "$benchmark" == "nasbench201" ]
then
  file="nasbench_dataset_names.txt"
else
  file="pd1_dataset_names.txt"
fi

export dir="$2"

if ! [ -e "$file" ] ; then     # spaces inside square brackets
    echo "$0: $file does not exist" >&2  # error message includes $0 and goes to stderr
    exit 1                   # exit code is non-zero for error
fi

NAMES=$(<$file)
for NAME in $NAMES
do
   export dataset=$(echo $NAME)
   msub -V -t 1-10 experiment_array.moab
done

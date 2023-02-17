#!/usr/bin/bash
if [ -z "$1" ]
  then
    dataset=Fed-TCGA-BRCA
  else
    dataset=$1
fi

if [ -z "$2" ]
  then
    MAX_RUNS=12
  else
    MAX_RUNS=$2
fi

if [ -z "$3" ]
  then
    OUTPUT_FOLDER=.
  else
    OUTPUT_FOLDER=$3
fi

if [ -z "$4" ]
  then
    TIMEOUT=360000000000
  else
    TIMEOUT=$4
fi

# Perform validation runs on all parameters defined in common.py
benchopt run --max-runs $MAX_RUNS -s FederatedAveraging -d $dataset --output $OUTPUT_FOLDER --timeout $TIMEOUT
benchopt run --max-runs $MAX_RUNS -s Cyclic -d $dataset --output $OUTPUT_FOLDER --timeout $TIMEOUT
benchopt run --max-runs $MAX_RUNS -s FedProx -d $dataset --output $OUTPUT_FOLDER --timeout $TIMEOUT
benchopt run --max-runs $MAX_RUNS -s Scaffold -d $dataset --output $OUTPUT_FOLDER --timeout $TIMEOUT
benchopt run --max-runs $MAX_RUNS -s FedAdam -d $dataset --output $OUTPUT_FOLDER --timeout $TIMEOUT
benchopt run --max-runs $MAX_RUNS -s FedAdagrad -d $dataset --output $OUTPUT_FOLDER --timeout $TIMEOUT
benchopt run --max-runs $MAX_RUNS -s FedYogi -d $dataset --output $OUTPUT_FOLDER --timeout $TIMEOUT

# Extract best hyperparameters for each strategy using final objective_value
python write_config_from_validation_results.py -o $OUTPUT_FOLDER -d $dataset



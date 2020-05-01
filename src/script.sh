#!/usr/bin/env bash

function runMain() {
    mode=$1
    estimator=$2
    fp=$3

    python3 main.py --mode ${mode} --estimator ${estimator} > ${fp}
}


# binary
runMain "binary" "logistic" "../binary_logistic.txt"
runMain "binary" "sgd" "../binary_sgd.txt"
runMain "binary" "xgb" "../binary_xgb.txt"
runMain "binary" "rf" "../binary_rf.txt"
runMain "binary" "vc" "../binary_vc.txt"

# multi
runMain "multi" "logistic" "../multi_logistic.txt"
runMain "multi" "sgd" "../multi_sgd.txt"
runMain "multi" "xgb" "../multi_xgb.txt"
runMain "multi" "rf" "../multi_rf.txt"
runMain "multi" "vc" "../multi_vc.txt"

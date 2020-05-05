#!/usr/bin/env bash

function runMain() {
    mode=$1
    estimator=$2
    fp=$3
    print_line=$4

    # run python script with given name of estimator
    python3 main.py --mode ${mode} --estimator ${estimator} > ${fp}

    # print finish message
    echo ${print_line}
}


# binary
runMain "binary" "logistic" "../output/binary_logistic.txt" "Finish binary logistic"
runMain "binary" "sgd" "../output/binary_sgd.txt" "Finish binary Stochastic Gradient Descent"
runMain "binary" "rf" "../output/binary_rf.txt" "Finish binary RandomForest"
runMain "binary" "vc" "../output/binary_vc.txt" "Finish binary Voting Classification"
runMain "binary" "xgb" "../output/binary_xgb.txt" "Finish binary XGBoost"

# multi
runMain "multi" "logistic" "../output/multi_logistic.txt" "Finish multi logistic"
runMain "multi" "sgd" "../output/multi_sgd.txt" "Finish multi Stochastic Gradient Descent"
runMain "multi" "rf" "../output/multi_rf.txt" "Finish multi RandomForest"
runMain "multi" "vc" "../output/multi_vc.txt" "Finish multi Voting Classification"
runMain "multi" "xgb" "../output/multi_xgb.txt" "Finish multi XGBoost"

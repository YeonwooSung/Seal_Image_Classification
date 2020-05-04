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
runMain "binary" "logistic" "../binary_logistic.txt" "Finish binary logistic"
runMain "binary" "sgd" "../binary_sgd.txt" "Finish binary Stochastic Gradient Descent"
runMain "binary" "rf" "../binary_rf.txt" "Finish binary RandomForest"
runMain "binary" "vc" "../binary_vc.txt" "Finish binary Voting Classification"
runMain "binary" "xgb" "../binary_xgb.txt" "Finish binary XGBoost"

# multi
runMain "multi" "logistic" "../multi_logistic.txt" "Finish multi logistic"
runMain "multi" "sgd" "../multi_sgd.txt" "Finish multi Stochastic Gradient Descent"
runMain "multi" "rf" "../multi_rf.txt" "Finish multi RandomForest"
runMain "multi" "vc" "../multi_vc.txt" "Finish multi Voting Classification"
runMain "multi" "xgb" "../multi_xgb.txt" "Finish multi XGBoost"

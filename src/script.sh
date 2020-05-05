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
runMain "binary" "logistic" "../result/output/binary_logistic.txt" "Finish binary logistic"
runMain "binary" "sgd" "../result/output/binary_sgd.txt" "Finish binary Stochastic Gradient Descent"
runMain "binary" "rf" "../result/output/binary_rf.txt" "Finish binary RandomForest"
runMain "binary" "vc" "../result/output/binary_vc.txt" "Finish binary Voting Classification"
runMain "binary" "xgb" "../result/output/binary_xgb.txt" "Finish binary XGBoost"

# multi
runMain "multi" "logistic" "../result/output/multi_logistic.txt" "Finish multi logistic"
runMain "multi" "sgd" "../result/output/multi_sgd.txt" "Finish multi Stochastic Gradient Descent"
runMain "multi" "rf" "../result/output/multi_rf.txt" "Finish multi RandomForest"
runMain "multi" "vc" "../result/output/multi_vc.txt" "Finish multi Voting Classification"
runMain "multi" "xgb" "../result/output/multi_xgb.txt" "Finish multi XGBoost"

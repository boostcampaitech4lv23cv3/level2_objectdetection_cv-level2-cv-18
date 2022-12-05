#!/usr/bin/env bash

WORK_DIR=$1
EPOCH_NAME=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

pipenv run python eval.py --work_dir="$1" --epoch_name="$2"
pipenv run python confusion_matrix.py --work_dir="$1" --eval_name="$2_eval.csv"

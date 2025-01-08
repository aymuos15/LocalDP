#!/bin/bash

# datasets=("breastmnist" "pneumoniamnist" "retinamnist" "bloodmnist" "organcmnist" "octmnist")
datasets=("breastmnist" "pneumoniamnist" "retinamnist" "bloodmnist")

for dataset in "${datasets[@]}"; do
   python main.py "$dataset"
done
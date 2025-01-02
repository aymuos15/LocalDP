#!/bin/bash

datasets=("breastmnist" "retinamnist")

for dataset in "${datasets[@]}"; do
   python main.py "$dataset"
done
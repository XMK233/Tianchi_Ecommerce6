#!/bin/bash
set -e
# export PYTHONPATH=$PYTHONPATH:/mnt/d/forCoding_code/Tianchi_Ecommerce6/plan_1
# cd /mnt/d/forCoding_code/Tianchi_Ecommerce6/plan_1

echo "Starting Training..."
python train.py

echo "\n\n\nStarting Prediction..."
python predict.py

echo "\n\n\nStarting Evaluation..."
python evaluate_f1.py

echo "Done!"

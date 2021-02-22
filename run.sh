#!/bin/sh

for embedding in 16 32 64 128 256; do
  for window_test in 1 2 3 4; do
    for attention_heads in 1 2 3 4; do
      echo 'EMBEDDING' $embedding 'WINDOW' $window_test 'HEADS' $attention_heads
      python src/hive/task/link_prediction/dynamic/train.py --data_path=/media/santaris/HighSpeedStorage/Datasets/datasets/analytics/predictive/stream/meta_learn/adidas/complete_interactions --dataset_file=aascy6qzqmw3uworzp3xzwf3e5we5w7qpn3apgpp3ih5lwmxx6uq_all_interactions.csv --emb_dim=$embedding --window=$window_test --attention_heads=$attention_heads
    done
  done
done
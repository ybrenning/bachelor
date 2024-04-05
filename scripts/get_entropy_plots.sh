#!/bin/bash

datasets=("mr" "ag-news" "trec")
query_strategies=("gc" "cb")

# Loop through each string in the list
for qs in "${query_strategies[@]}"
do
    for d in "${datasets[@]}"
    do
        scp -r yb63tadu@ssh.webis.de:/mnt/ceph/storage/data-tmp/current/yb63tadu/entropy-plot-$qs-$d-transformer.npz data
    done
done

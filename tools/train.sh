#!/bin/bash

script=$1
shift

hostname=`(hostname)`
if [[ $hostname == ccc* ]];
then
    let "NODE_RANK=($LSF_PM_XMACHID - 1)"
    MASTER=$LSF_FROM_HOST
elif [[ $hostname == c699* ]];
then
    MASTER=$MASTER_HOSTNAME
    NODE_RANK=$PMIX_RANK
    NGPUS=6
fi

echo python -m torch.distributed.launch --master_addr=$MASTER --master_port=10596 --nnodes=$NODES --node_rank=$NODE_RANK --nproc_per_node=$NGPUS $script "$@"
python -m torch.distributed.launch --master_addr=$MASTER --master_port=10596 --nnodes=$NODES --node_rank=$NODE_RANK --nproc_per_node=$NGPUS $script "$@"



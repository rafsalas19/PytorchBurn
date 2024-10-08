#!/bin/bash

# Azure Market Place Image specific. Exclude if not using Azure Market Place Image
source /opt/hpcx-v2.18-gcc-mlnx_ofed-ubuntu22.04-cuda12-x86_64/hpcx-init.sh 
hpcx_load

mpirun -np 2 -H <host1>:1 <host2>:1 --map-by ppr:1:node -bind-to numa \
-mca coll_hcoll_enable 0 \
-x UCX_TLS=tcp \
-x UCX_NET_DEVICES=eth0 \
-x CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 /mnt/anfvol/rafael/PytorchBurn/pyburn.py -p 2 -t 3 -g 0 1 2 3 4 5 6 7
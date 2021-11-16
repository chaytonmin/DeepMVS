#!/bin/bash
source env.sh

# -----------for tanks------------------

testlist=./lists/tp_list.txt
outdir=./outputs_tnt_1101
test_dataset=tanks
python fusion.py --testpath=$TP_TESTING \
                     --testlist=$testlist \
                     --outdir=$outdir \
                     --test_dataset=$test_dataset  &

# -----------for dtu------------------

# testlist=./lists/dtu/test.txt
# outdir=./outputs_1108
# test_dataset=dtu
# python fusion.py --testpath=$DTU_TESTING \
#                     --testlist=$testlist \
#                     --outdir=$outdir \
#                     --test_dataset=$test_dataset \
#!/bin/bash

### yago2-sample
#make clean reset
#IMPORT="recursive=False" D=yago2-sample N=nonrecursive DEPTH=0 make -Be phase
#make clean reset
#IMPORT="recursive=True" D=yago2-sample N=recursive DEPTH=0 make -Be phase
#make clean reset
#IMPORT="recursive=True" D=yago2-sample N=recursive DEPTH=1 make -Be phase
#make clean reset
#IMPORT="recursive=True" D=yago2-sample N=recursive DEPTH=2 make -Be phase
#make clean reset
#IMPORT="recursive=True" D=yago2-sample N=recursive DEPTH=3 PARALLEL=10 make -Be phase

### yago2-core
#make clean reset
#IMPORT="recursive=True" D=yago2-core N=recursive DEPTH=1 make -Be phase

#make clean reset
#IMPORT="recursive=False" D=yago2-core N=nonrecursive DEPTH=0 make -Be phase
#make clean reset
#IMPORT="recursive=False" D=yago2-core N=nonrecursive DEPTH=1 make -Be phase
#make clean reset
#IMPORT="recursive=True" D=yago2-core N=recursive DEPTH=0 make -Be phase

## failed
#make clean reset
#IMPORT="recursive=True" D=yago2-core N=recursive DEPTH=2 make -Be phase
#make clean reset
#IMPORT="recursive=True" D=yago2-core N=recursive DEPTH=2 PARALLEL=10 make -e phase

#IMPORT="recursive=True" D=yago2-core N=recursive_5k DEPTH=2 PARALLEL=10 make -e phase
#IMPORT="recursive=True" D=yago2-core N=recursive_5k DEPTH=1 PARALLEL=10 make -e phase
#IMPORT="recursive=True" D=yago2-core SET_M="M=1000;" N=recursive_1k DEPTH=2 PARALLEL=5 EPOCHS=10 make -e phase
#IMPORT="recursive=True" D=yago2-core SET_M="M=1000;" N=recursive_1k DEPTH=1 P#ARALLEL=10 make -e phase

#IMPORT="recursive=True" D=yago2-core SET_M="M=1000;" N=recursive_1k_kbeval DEPTH=2 PARALLEL=5 EPOCHS=10 make -e phase

#IMPORT="recursive=False" D=yago2-core SET_M="M=1000;" N=nonrecursive_1k_kbeval DEPTH=0 PARALLEL=10 EPOCHS=30 make -e phase
#IMPORT="recursive=False" D=yago2-core SET_M="M=2000;" N=nonrecursive_2k_kbeval DEPTH=0 PARALLEL=10 EPOCHS=30 make -e phase
#IMPORT="recursive=True" D=yago2-core SET_M="M=1000;" N=recursive_1k_kbeval DEPTH=1 PARALLEL=10 EPOCHS=30 make -e phase
#IMPORT="recursive=True" D=yago2-core SET_M="M=1000;" N=recursive_1k_kbeval DEPTH=2 PARALLEL=5 EPOCHS=30 RATE=0.0001 make -e phase

#DRYRUN="n";

## primary experimental set
#IMPORT="recursive=False" D=yago2-core SET_M="M=1000;" N=nonrecursive_1k_kbeval DEPTH=0 PARALLEL=30 EPOCHS=30 make -e${DRYRUN} phase
#IMPORT="recursive=True" D=yago2-core SET_M="M=1000;" N=recursive_1k_kbeval DEPTH=1 PARALLEL=30 EPOCHS=30 make -e${DRYRUN} phase
#IMPORT="recursive=True" D=yago2-core SET_M="M=1000;" N=recursive_1k_kbeval DEPTH=2 PARALLEL=5 EPOCHS=30 RATE=0.0001 make -e${DRYRUN} phase


## proppr untrained inference
rm proppr.settings
#IMPORT="recursive=False" D=yago2-core N=nonrecursive_1k_kbeval DEPTH=0 PARALLEL=30 EPOCHS=30 make -e${DRYRUN} proppr
proppr set --prover idpr:maxTreeDepth=5
IMPORT="recursive=True" D=yago2-core N=recursive_1k_kbeval DEPTH=2 PARALLEL=5 EPOCHS=30 make -e${DRYRUN} proppr

## minibatch experiment
#for b in 10 5 2 1; do
#    IMPORT="recursive=False" D=yago2-core N=nonrecursive_mb DEPTH=0 PARALLEL=30 EPOCHS=30 BATCH=$b make -e${DRYRUN} minibatch
#done



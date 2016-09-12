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
IMPORT="recursive=True" D=yago2-core SET_M="M=1000;" N=recursive_1k_kbeval DEPTH=1 PARALLEL=10 EPOCHS=30 make -e phase
IMPORT="recursive=True" D=yago2-core SET_M="M=1000;" N=recursive_1k_kbeval DEPTH=2 PARALLEL=5 EPOCHS=30 RATE=0.0001 make -e phase



# primary experimental set
# for r in True False; do
#     n="recursive";
#     depths="1 2";
#     if [ "$r" = "False" ]; then
# 	n="nonrecursive";
# 	depths="0";
#     fi
#     for d in $depths; do
# 	IMPORT="recursive=$r" D=yago2-core N="${n}_1k_kbeval" DEPTH="$d" PARALLEL=5 EPOCHS=10 make -e phase
#     done
# done



## proppr untrained inference
#IMPORT="recursive=False" D=yago2-core N=nonrecursive DEPTH=0 make -e proppr
#IMPORT="recursive=True" D=yago2-core N=recursive DEPTH=0 make -e proppr


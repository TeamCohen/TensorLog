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
IMPORT="recursive=True" D=yago2-core N=recursive DEPTH=2 PARALLEL=10 make -e phase



## proppr untrained inference
#IMPORT="recursive=False" D=yago2-core N=nonrecursive DEPTH=0 make -e proppr
#IMPORT="recursive=True" D=yago2-core N=recursive DEPTH=0 make -e proppr


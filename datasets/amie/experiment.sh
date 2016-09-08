#!/bin/bash

#make clean reset
#IMPORT="recursive=False" D=yago2-sample-nonrecursive DEPTH=0 make -Be phase
#make clean reset
#IMPORT="recursive=True" D=yago2-sample-recursive DEPTH=0 make -Be phase
#make clean reset
#IMPORT="recursive=True" D=yago2-sample-recursive DEPTH=1 make -Be phase
#make clean reset
#IMPORT="recursive=True" D=yago2-sample-recursive DEPTH=2 make -Be phase
make clean reset
IMPORT="recursive=True" D=yago2-sample-recursive DEPTH=3 PARALLEL=10 make -Be phase

expt: expt.log

expt.log:
	python supervised-expt.py cora > expt.log
	python supervised-expt.py citeseer >> expt.log
	python supervised-expt.py pubmed >> expt.log

%.eval:
	python -m expt \
	    --db 'inputs/$*.db|inputs/$*-corpus.cfacts' --prog inputs/$*-textcat.ppr --proppr \
	    --train 'inputs/$*-train.dset|inputs/$*-train.exam' \
	    --test 'inputs/$*-test.dset|inputs/$*-test.exam' \
	    +++ \
	    --learner plearn.ParallelFixedRateGDLearner --learnerOpts "\"{'epochs':60,'parallel':55,'rate':20}\"" \
	    --savedModel tmp-cache/$*.model

sr8:
	python supervised-expt.py r8 20

profile:
	python -m cProfile -s cumtime supervised-expt.py r8 4

check: actual.txt
	diff -y actual.txt expected.txt || true

actual.txt: expt.log
	echo actual results on `date`: cora, citeseer, pubmed > actual.txt
	grep 'training.*done' expt.log >> actual.txt
	grep 'eval.*on test' expt.log >> actual.txt

clean:
	rm -rf tmp-cache/* *~ bin/*~ expt.log

#
# import the original inputs, by default from /afs/cs/project/proppr-1
#

import:
	python bin/import.py cora
	python bin/import.py citeseer
	python bin/import.py pubmed
	python bin/import.py r8 ~/shared-home/data/proppr/r8-inputs/

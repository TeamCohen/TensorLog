expt: expt.log

expt.log:
	python supervised-expt.py cora > expt.log
	python supervised-expt.py citeseer >> expt.log
	python supervised-expt.py pubmed >> expt.log

check: actual.txt
	diff -y actual.txt expected.txt || true

actual.txt: expt.log
	echo actual results on `date`: cora, citeseer, pubmed > actual.txt
	grep 'training.*done' expt.log >> actual.txt
	grep 'eval.*on test' expt.log >> actual.txt

clean:
	rm -rf tmp-cache/* inputs/* *~ bin/*~ expt.log

#
# import the original inputs from /afs/cs/project/proppr-1
#

import:
	python bin/import.py cora
	python bin/import.py citeseer
	python bin/import.py pubmed
	python bin/import.py r8 ~/shared-home/data/proppr/r8-inputs/

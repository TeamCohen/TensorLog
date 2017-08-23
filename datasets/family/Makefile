#TODO: fix for new API

VPATH= raw

expt.log: 
	python expt.py > expt.log

setup:
	mkdir -p temp-cache

actual.txt: expt.log
	echo \# actual result on `date` > actual.txt
	for f in tmp-cache/*.solutions.txt; do \
	echo file $$f >> actual.txt ; \
	proppr eval tmp-cache/kinship-test.examples $$f --metric auc --defaultNeg >> actual.txt ; \
	done

check: actual.txt
	diff -y $< expected.txt || true

unittest:
	python testexpt.py

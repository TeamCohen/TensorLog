expt.log:
	python expt.py acc 16 > expt.log

actual.txt: expt.log
	echo \# actual result on `date` > actual.txt
	for f in tmp-cache/*.solutions.txt; do \
	echo file $$f >> actual.txt ; \
	proppr eval tmp-cache/test.examples $$f --metric auc --defaultNeg >> actual.txt ; \
	done
#	tail -13 expt.log  >> actual.txt

check: actual.txt
	diff -y actual.txt expected.txt || true

setup:
	mkdir -p inputs tmp-cache

unittest:
	python testexpt.py 

test:
	python expt.py time 16
	python expt.py time 64
	python expt.py acc

clean:
	rm -f *.pyc *.cfacts *.exam *~ visualize.png expt.log actual.txt inputs/* tmp-cache/*



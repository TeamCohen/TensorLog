expt.log: setup
	python grid-expt.py acc 16 > expt.log

actual.txt: expt.log
	echo experiment on `date` > actual.txt
	tail -13 expt.log  >> actual.txt

check: actual.txt
	diff -y actual.txt expected.txt || true

setup:
	mkdir -p inputs tmp-cache

unittest:
	python testexpt.py 

test:
	python grid-expt.py time 16
	python grid-expt.py time 64
	python grid-expt.py acc

clean:
	rm -f *.pyc *.cfacts *.exam *~ visualize.png expt.log actual.txt inputs/* tmp-cache/*



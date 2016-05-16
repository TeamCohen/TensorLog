short-tests:
	python testtensorlog.py

long-tests: short-tests
	python benchmark.py

wnet-test:
	(export PYTHONPATH=`pwd`; cd ../datasets/wordnet/; python wnet-expt.py)

snap:
	tar -czf backup/snap`date +%m.%d.%H.%M.%S`.tgz *.py Makefile
	ls -alh backup

wc:
	wc `ls *.py | grep -v pyparsing | grep -v wam | grep -v test | grep -v try | grep -v benchmark`

wc-all:
	wc `ls *.py | grep -v pyparsing`

clean:
	rm *.pyc *.py~

short-tests:
	python testtensorlog.py

long-tests: short-tests benchmark-test wnet-test

benchmark-test:
	python benchmark.py

wnet-test:
	(export PYTHONPATH=`pwd`; cd ../datasets/wordnet/; python wnet-expt.py; \
	 proppr eval hypernym-test.examples hypernym-test.solutions.txt --metric map)

snap:
	tar -czf backup/snap`date +%m.%d.%H.%M.%S`.tgz *.py Makefile
	ls -alh backup

wc:
	wc `ls *.py | grep -v pyparsing | grep -v wam | grep -v test | grep -v try | grep -v benchmark`

wc-all:
	wc `ls *.py | grep -v pyparsing`

clean:
	rm *.pyc *.py~

# unit tests that take a few secs to run - all of these are about a minute all told

short-tests:
	python -m unittest testtensorlog
	python -m unittest testxcomp
	python -m unittest testtf

all-automatic-tests: short-tests grid-test cora-test wikimovies-test simple-comline-test smokers-test fb-benchmark-test

# tests that take a few minutes to run
# run 'make prepare-for-long-tests' in a fresh install before you run these
long-tests: benchmark-test wnet-test cora-test grid-test

prepare-for-long-tests:
	(export PYTHONPATH=`pwd`; cd ../datasets/wordnet; make setup)
	(export PYTHONPATH=`pwd`; cd ../datasets/cora; make setup)

#
# individual longer tests, which should take about 1min each
#

grid-test:
	(cd ../datasets/grid/; make clean; make check; make clean; make unittest)

cora-test:
	(cd ../datasets/cora/; make clean; make check; make clean; make unittest)

wikimovies-test:
	(cd ../datasets/wikimovies/; make clean; make unittest)

smokers-test:
	(cd ../datasets/smokers/; make clean; make unittest)

simple-comline-test:
	python simple.py experiment --db test-data/textcattoy.cfacts --prog test-data/textcat.ppr --train test-data/toytrain.exam  --test test-data/toytest.exam

fb-benchmark-test:
	(cd ../datasets/fb15k-speed/; make clean; make unittest)

# not converted yet....

wnet-test:
	(cd ../datasets/wordnet/; make clean; make; make check)

# debug-test needs to be interactive
debug-test:
	python expt.py --prog test-data/textcat.ppr --db test-data/textcattoy.cfacts \
		--trainData test-data/toytrain.exam --testData test-data/toytest.exam --proppr +++ --savedModel expt-model.db
	python debug.py --prog test-data/textcat.ppr --db expt-model.db --trainData test-data/toytrain.exam --testData test-data/toytest.exam --proppr predict/io

# word count

wc:
	wc `ls *.py | grep -v test | grep -v try`

# cleanup

clean:
	rm -f *.pyc *.py~ *.prof 
	rm -f toy-test.examples toy-test.solutions.txt toy-train.examples 
	rm -rf toy-trained.db

dist:
	tar -cvf ../tensorlog-dist.tar ../LICENSE Makefile *.py test-data/*.*

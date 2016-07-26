# unit tests that take a few secs to run
short-tests:
	python -m unittest testtensorlog

# tests that take a few minutes to run
# run 'make prepare-for-long-tests' first
long-tests: short-tests benchmark-test wnet-test cora-test

prepare-for-long-tests:
	make fb15k-valid.db 
	(export PYTHONPATH=`pwd`; cd ../datasets/wordnet; make setup)
	(export PYTHONPATH=`pwd`; cd ../datasets/cora; make setup)

fb15k-valid.db:
	python matrixdb.py --serialize test/fb15k-valid.cfacts fb15k-valid.db

#
# individual long tests
#

benchmark-test:
	python benchmark.py

wnet-test:
	(export PYTHONPATH=`pwd`; cd ../datasets/wordnet/; make clean; make; make check)

cora-test:
	(export PYTHONPATH=`pwd`; cd ../datasets/cora/; make clean; make; make check)

textcattoy-expt:
	#python -m expt --db test/textcattoy.cfacts --prog test/textcat.ppr --trainData test/toytrain.exam --testData test/toytrain.exam --proppr +++ --learner plearn.ParallelAdaGradLearner --learnerOpts '{"regularizer":learn.L2Regularizer()}'
	python -m expt --db test/textcattoy.cfacts --prog test/textcat.ppr --trainData test/toytrain.exam --testData test/toytrain.exam --proppr

# debug-test needs to be interactive
debug-test:
	python expt.py --prog test/textcat.ppr --db test/textcattoy.cfacts --trainData test/toytrain.exam --testData test/toytest.exam --proppr
	python debug.py --prog test/textcat.ppr --db expt-model.db --trainData test/toytrain.exam --testData test/toytest.exam --proppr predict/io

# word count

wc:
	wc `ls *.py | grep -v pyparsing | grep -v wam | grep -v test | grep -v try | grep -v benchmark`

wc-all:
	wc `ls *.py | grep -v pyparsing`

# cleanup

clean:
	rm -f *.pyc *.py~ *.prof
	rm -f toy-test.examples toy-test.solutions.txt toy-train.examples 
	rm -rf toy-trained.db

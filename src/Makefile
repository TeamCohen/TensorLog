tests:
	python testtensorlog.py

noconstTests:
	python -m unittest testtensorlog.TestSmallProofs.testIf
	python -m unittest testtensorlog.TestSmallProofs.testOr
	python -m unittest testtensorlog.TestSmallProofs.testChain
	python -m unittest testtensorlog.TestSmallProofs.testMid
	python -m unittest testtensorlog.TestSmallProofs.testNest
	python -m unittest testtensorlog.TestSmallProofs.testBack1
	python -m unittest testtensorlog.TestSmallProofs.testRec1
	python -m unittest testtensorlog.TestSmallProofs.testAltChain

sample-usage:
	python tensorlog.py test/fam.cfacts 'p(X,Z):-spouse(X,Y),sister(Y,Z).' 'p(i,o)' 'susan' 

snap:
	tar -czf backup/snap`date +%m.%d.%H.%M.%S`.tgz *.py Makefile
	ls -alh backup

wc:
	wc `ls *.py | grep -v pyparsing | grep -v wam | grep -v test | grep -v try | grep -v benchmark`
wc-all:
	wc `ls *.py | grep -v pyparsing`

clean:
	rm *.pyc

expt: actual.txt

actual.txt:
	echo \# actual result on `date` > actual.txt
	python smokers-expt.py >> actual.txt

check:
	diff -y actual.txt expected.txt || true

clean:
	rm -f *~ *.pyc actual.txt


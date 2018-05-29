args="--duplicateCheck -1 --countFeatures false"
date > proppr-expt.log
echo args $args >> proppr-expt.log
#args="--duplicateCheck -1 --countFeatures false --apr eps=1e-5:alph=0.1:depth=10"
#args="--duplicateCheck -1 --countFeatures false --prover ppr --apr eps=1e-4:alph=0.1:depth=10"
proppr compile grid.ppr
#for n in 10
for n in 10 25 50 100 200;
do
    echo executing for size $n args $args
    python exam2proppr.py < g$n-test.exam > g$n.examples
    python facts2proppr.py < g$n.cfacts > p$n.cfacts
    echo run proppr answer g$n.examples --programFiles grid.wam:p$n.cfacts $args --threads 1 
    proppr answer g$n.examples --programFiles grid.wam:p$n.cfacts $args --threads 1 > g$n.log
    python average-time-in-solutions.py g$n 1 < g$n.log >> proppr-expt.log 
done

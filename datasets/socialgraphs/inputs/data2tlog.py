import sys
import random

if __name__ == "__main__":
  stem = sys.argv[1]
  trainFrac = 2.0/3.0
  factFrac = 1.0/2.0
  seen = set()
  with open(stem+'.cfacts','w') as factFP, open(stem+'-train.exam','w') as trainFP, open(stem+'-test.exam','w') as testFP:
    for line in open(stem+'-ghirl.txt'):
      try:
        (_,rel,src,dst) = line.strip().split(" ")
      except ValueError:
        print 'bad line %r' % line
      if src.isdigit(): src = 'node%03d' % int(src)
      if dst.isdigit(): dst = 'node%03d' % int(dst)
      if rel=='e':
        factFP.write('\t'.join(['friend',src,dst]) + '\n')
      elif rel=='isa':
        if src not in seen:
          seen.add(src)
          r = random.uniform(0.0,1.0)
          if  r > trainFrac:
            examFP = testFP
            examRel = 'inferred_label'
          elif r>factFrac*trainFrac:
            examFP = trainFP
            examRel = 'inferred_label'
          else:
            examFP = factFP
            examRel = 'label'
          examFP.write('\t'.join([examRel,src,dst]) + '\n')

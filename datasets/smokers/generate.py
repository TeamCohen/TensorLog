if __name__=="__main__":
   fp = open("cancer-smokes.cfacts",'w')
   fp2 = open("query-entities.txt",'w')
   for line in open('labels.txt'):
       id,lab = line.strip().split("\t")
       fp2.write(id + '\n')
       if lab=="yAgents":
          fp.write("cancer\t%s\n" % id)
       elif lab=="yAI":
          fp.write("smokes\t%s\n" % id)
       elif lab=="yDB":
          fp.write("smokes\t%s\n" % id)
          fp.write("cancer\t%s\n" % id)



import sys


#prepare input for automatic evaluation of compound translation error rates (CTER)
source_path = sys.argv[1]
compound_path = sys.argv[2]
target_path = sys.argv[3]
out_path = sys.argv[4]

with open(source_path) as sf, open(compound_path) as cf, open(target_path) as tf, open(out_path, 'w') as of:
	for sline, cline, tline in zip(sf, cf, tf):
		line = sline.strip()+'\t'+cline.strip()+'\t'+tline.strip()+'\n'
		of.write(line)


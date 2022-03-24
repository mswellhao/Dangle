import sys
import spacy
from collections import Counter
import re, json
import random


#preprocess meaning representations in cogs : 
#removing indexed variables to enable deeper recursion generalization because it's hard for a neural model to generate a new index at test time.

in_file = sys.argv[1]
out_file = sys.argv[2]


prog1 = re.compile(r"[D;] (\w+) \.|[*D;] (\w+) \(|^(\w+) \.|^(\w+) \(")
prog2 = re.compile(r"\. (\w+) \(")
prog3 = re.compile(r"(\* \w+|\w+) \( (x \_ \d+) \)|(\w+) \. (\w+) \( (x \_ \d+) , (x \_ \d+|[A-Z][a-z]+) \)|(\w+) \. \w+ \. (\w+) \( (x \_ \d+) , (x \_ \d+|[A-Z][a-z]+) \)")

duplicate = 0
rels = {'theme' : 1, 'on' : 2, 'ccomp' : 3, 'agent' : 4, 'beside' : 5, 'in' : 6, 'recipient' : 7, 'xcomp' : 8}
with open(in_file) as tf, open(out_file, "w") as of:
	for line1 in tf:
		if "LAMBDA" in line1:
			of.write(line1)
			continue
		matches = prog3.findall(line1)
		if len(matches) == 0:
			of.write(line1)
			continue
		v2p = {}
		triples = []
		for match in matches:
			if match[0]:
				var = match[1]
				v2p[var] = match[0]
			elif match[2]:
				var = match[4]
				if var not in v2p:
					v2p[var] = match[2]
				triples.append([match[4], match[3], match[5]])
			elif match[6]:
				var = match[8]
				if var not in v2p:
					v2p[var] = match[6]
				triples.append([match[8], match[7], match[9]])
		# print("triples : "+str(triples))
		en = set()
		str_list = []
		for triple in triples:
			if triple[2] in v2p:
				triple[2] = v2p[triple[2]]
			triple[0] = v2p[triple[0]]
			str_list.append(triple[1]+" ( "+triple[0]+" , "+triple[2]+" )")

		out_str = "bos "+" AND ".join(str_list)+"\n"

		# print("en : "+str(en))
		# print("triples : "+str(triples))

		of.write(out_str)




		# predicates = prog1.findall(line2)
		# predicates = ["".join(item) for item in predicates]
		# for pre in predicates:
		# 	find_lemmas = [item for item in lemmas if item == pre]
		# 	if len(find_lemmas) == 0:
		# 		print("0: ")
		# 		print("source : "+str(line1.strip()))
		# 		print("pre : "+str(pre)+"\n")
		# 		stat[0] += 1
		# 	elif len(find_lemmas) == 1:
		# 		stat[1] += 1
		# 	elif len(find_lemmas) == 2:
		# 		# print("2: ")
		# 		# print("source : "+str(line1))
		# 		# print("pre : "+str(pre))
		# 		stat[2] += 1
		# 	elif len(find_lemmas) == 3:
		# 		stat[3] += 1
		# relations = prog2.findall(line2)
		# rels.update(relations)

# print("duplicate : "+str(duplicate))
# print("stat : "+str(stat))
# print("rels : "+str(rels))


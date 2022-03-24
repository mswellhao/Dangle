import sys
import random
from collections import defaultdict, Counter
import os, re

#directory for original splits
in_dir = sys.argv[1]
#directory for ouput splits
out_dir = sys.argv[2]

prog_recurs3 = re.compile(r" (beside|in|on) \w+ \w+ (beside|in|on) \w+ \w+ (beside|in|on) ")
prog_recurs4 = re.compile(r" (beside|in|on) \w+ \w+ (beside|in|on) \w+ \w+ (beside|in|on) \w+ \w+ (beside|in|on) ")
prog_recurs5 = re.compile(r" (beside|in|on) \w+ \w+ (beside|in|on) \w+ \w+ (beside|in|on) \w+ \w+ (beside|in|on) \w+ \w+ (beside|in|on) ")
prog_recurs6 = re.compile(r" (beside|in|on) \w+ \w+ (beside|in|on) \w+ \w+ (beside|in|on) \w+ \w+ (beside|in|on) \w+ \w+ (beside|in|on) \w+ \w+ (beside|in|on) ")


test_predicate_file = os.path.join(in_dir, "test.word-predicate.predicate")
test_word_file = os.path.join(in_dir, "test.word-predicate.word")
test_code_file = os.path.join(in_dir, "test.word-predicate.code")

train_predicate_file = os.path.join(in_dir, "train.word-predicate.predicate")
train_word_file = os.path.join(in_dir, "train.word-predicate.word")
train_code_file = os.path.join(in_dir, "train.word-predicate.code")


with open(test_predicate_file) as pf, open(test_word_file) as wf,  open(test_code_file) as cf:
	original_test_items = [item for item in zip(pf,wf,cf)]

with open(train_predicate_file) as pf, open(train_word_file) as wf,  open(train_code_file) as cf:
	original_train_items = [item for item in zip(pf,wf,cf)]
	# random.shuffle(original_train_items)


d = defaultdict(set)
for item in original_test_items:
	wline = item[1].strip()
	# print(wline)
	words = wline.split()
	matches = prog_recurs3.findall(wline)
	if matches:
		d[3].add(item)
	elif Counter(words)['that'] >= 3:
		d[3].add(item)

	matches = prog_recurs4.findall(wline)
	if matches:
		d[4].add(item)
	elif Counter(words)['that'] >= 4:
		d[4].add(item)

	matches = prog_recurs5.findall(wline)
	if matches:
		d[5].add(item)
	elif Counter(words)['that'] >= 5:
		d[5].add(item)

	matches = prog_recurs6.findall(wline)
	if matches:
		d[6].add(item)
	elif Counter(words)['that'] >= 6:
		d[6].add(item)

	


print([(key, len(d[key])) for key in d])

items_recurse_d = {3: d[3] - d[4], 4: d[3] - d[5], 5: d[3] - d[6]}
for i in range(3,6):
	original_test_num = len(original_test_items)
	original_train_num = len(original_train_items)
	print("original train num : "+str(original_train_num)+" test num : "+str(original_test_num))
	added_num = len(items_recurse_d[i]) 
	test_num = original_test_num - added_num
	added_items = items_recurse_d[i]
	test_items = set(original_test_items) - added_items
	train_items = original_train_items + list(added_items) 

	out_prefix = out_dir+"-recursion"+str(i)
	os.makedirs(out_prefix, exist_ok=True)
	out_train_predicate_file = os.path.join(out_prefix, "train.word-predicate.predicate")
	out_train_word_file = os.path.join(out_prefix, "train.word-predicate.word")
	out_train_code_file = os.path.join(out_prefix, "train.word-predicate.code")

	out_test_predicate_file = os.path.join(out_prefix, "test.word-predicate.predicate")
	out_test_word_file = os.path.join(out_prefix, "test.word-predicate.word")
	out_test_code_file = os.path.join(out_prefix, "test.word-predicate.code")

	with open(out_train_predicate_file, "w") as pf, open(out_train_word_file,"w") as wf, open(out_train_code_file,"w") as cf:
		for item in train_items:
			pf.write(item[0])
			wf.write(item[1])
			cf.write(item[2])

	with open(out_test_predicate_file, "w") as pf, open(out_test_word_file,"w") as wf, open(out_test_code_file,"w") as cf:
		for item in test_items:
			pf.write(item[0])
			wf.write(item[1])
			cf.write(item[2])

	print("recursion "+str(i)+" train num : "+str(len(train_items))+" test num : "+str(len(test_items)))






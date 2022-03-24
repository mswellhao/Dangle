import sys
import re
from collections import defaultdict

in_file = sys.argv[1]
out_file = sys.argv[2]

reverse_dic = {}
reverse_dic['film.film.executive_produced_by'] = 'film.producer.films_executive_produced'
reverse_dic['film.film.directed_by'] = 'film.director.film'
reverse_dic['film.film.written_by'] = 'film.writer.film'
reverse_dic['influence.influence_node.influenced_by'] = 'influence.influence_node.influenced'
reverse_dic['film.film.edited_by'] = 'film.editor.film'
reverse_dic['film.film.produced_by|ns:film.film.production_companies'] = 'film.producer.film|ns:film.production_company.films'
reverse_dic['people.person.employment_history/ns:business.employment_tenure.company'] = 'business.employer.employees/ns:business.employment_tenure.person'
reverse_dic['organization.organization.founders'] = 'organization.organization_founder.organizations_founded'
reverse_dic['film.film.distributors/ns:film.film_film_distributor_relationship.distributor'] = 'film.film_distributor.films_distributed/ns:film.film_film_distributor_relationship.film'
reverse_dic['organization.organization.acquired_by/ns:business.acquisition.acquiring_company']= 'organization.organization.companies_acquired/ns:business.acquisition.company_acquired'


prog1 = re.compile(r"[^ ]{2,} [^ ]{1,} [^ ]{2,} [.}]")
prog2 = re.compile(r"FILTER [^.]+ [.}]")

with open(in_file) as inf, open(out_file, "w") as outf:
	for lnum, line in enumerate(inf):
		predicate_matches = prog1.findall(line)
		d = defaultdict(list)
		for match in predicate_matches:
			items = match.split()
			if items[1] in reverse_dic:
				items[1] = reverse_dic[items[1]]
				items[0], items[2] = items[2], items[0]
			d[(items[0], items[1])].append(items[2])
		d2 = defaultdict(list)
		for key, value in d.items():
			d2[(key[1], tuple(value))].append(key[0])
		d3 = defaultdict(list)
		for key, value in d2.items():
			#(arg2, arg1) -> pred
			d3[(key[1], tuple(value))].append(key[0])

		ostring = []
		flag = False
		for key, value in d3.items():
			arg1 = list(key[1])
			arg1.sort()
			arg2 = list(key[0])
			arg2.sort()
			if len(arg1) > 1:
				flag = True
			arg1_string = "{} ".format(" ".join(arg1))
			arg2_string = " {}".format(" ".join(arg2))

			predicate = list(value)
			predicate.sort()
			pred_string = "{}".format(" ".join(predicate))

			ostring.append(arg1_string + pred_string + arg2_string)
		filter_matches = prog2.findall(line)
		for match in filter_matches:
			ostring.append(match[:-2])
		# mstring = " . ".join([" ".join(match) for match in matches])
		# print("line : "+line.strip())
		# print("mstring : "+mstring+"\n")
		# assert mstring in line 
		ostring = sorted(ostring)
		ostring = " . ".join(ostring)
		newline = line
		for i, match in enumerate(predicate_matches+filter_matches):
			newline = newline.replace(match, "")
		
		newline = newline.strip() + " " + ostring + " }\n"

		# print("new line : "+newline)

		# if flag:
		# 	print("line : "+line.strip())
		# 	print("new line : "+newline)

		outf.write(newline)
		# if lnum == 100:
			# break
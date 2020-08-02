import segment
import snp_ssm
import cnv
import constants as cons
import exceptions_onctopus as eo
import os.path
import lineage
import submarine
import itertools
import numpy as np
import logging
import os
import random
import json

# defined functions in this file
# def parse_vcf_file_for_onctopus(input_file_name, output_file_name, test):
# def write_CPLEX_results_to_result_file(file_name, opt, snp_list, ssm_list, seg_list):
# def get_CNV_lines_from_CPLEX(opt, seg_list, start_index):
# def get_SSM_lines_from_CPLEX(opt, ssm_list, start_index):
# def get_SNP_line_from_CPLEX(opt, snp_list, start_index):
# def read_result_file(file_name):
# def store_results(tag, my_lineage, line_part):
# def get_sublineages_list_from_result_file(line_part):
# def get_CNV_list_from_result_file(line_part):
# def get_SNP_SSM_list_from_result_file(line_part):
# def read_segment_file(file_name):
# def read_snp_ssm_file(file_name, mut_type):
# def raise_if_file_exists(file_name):
# def raise_if_file_not_exists(file_name):
# def write_segment_file(reads_total, seg_start, seg_end, segment_number, mass, file_name, no_test):
# def write_single_muts_file(reads_total, lineages, seg_start, segment_number, file_name_snp, file_name_ssm, no_test):
# def get_sublineages_line(sublins):
# def get_end_of_mutation_line(line, mut, lin, pha, segment_number, seg_start, seg_end):
# def write_simulation_results(seg_start, seg_end, lineages, segment_number, file_name, no_test):
# def write_simulation_info(lineages_file, segment_number, snp_number, ssm_number, cnv_number, 
# def read_lineages_tree(file_name, segment_number):
# def read_mutation_assignments(file_name)
# def str_to_bool(s):
# def str_possibly_to_none(s):
# def check_duplicates(chr_pos_list):
# def create_fixed_segments_all_but_one(result_file, output_prefix, cn_state_num=2, test=False):
# def create_fixed_CNV_data(result_file, cn_state_num):
# def create_fixed_SNPs_data(result_file):
# def create_fixed_SSMs_data(result_file):
# def create_fixed_Z_data(result_file):
# def create_fixed_frequencies_data(result_file):
# def read_fixed_value_file(fixed_file):
# def write_fixed_value_file(data, output_file, row_num, column_num, info, unfixed_segment = -1, unfixed_start = -1, unfixed_stop = -1, file_type=None, test=False):
# def visualize_result_file(result_file, output_file=None, test=False):

def read_parent_vector(my_file):
	lins_parents = []
	first_line = True
	with open(my_file, "r") as f:
		for line in f:
			if first_line:
				first_line = False
				continue
			lins_parents.append(list(map(int,line.rstrip().split("\t"))))
	parents = [p[1] for p in sorted(lins_parents, key=lambda x: x[0])]
	return parents

def read_frequencies(my_file, return_ids=False, ordering_given=True):
	lins_freqs = []
	first_line = True
	with open(my_file, "r") as f:
		for line in f:
			if first_line:
				first_line = False
				continue
			lins = int(line.split("\t")[0])
			freqs = list(map(float, line.rstrip().split("\t")[1:]))
			lins_freqs.append([lins, freqs])
	if ordering_given:
		freqs = [f[1] for f in sorted(lins_freqs, key=lambda x: x[0])]
	else:
		freqs = [f[1] for f in lins_freqs]
	
	if return_ids == False:
		return freqs

	if ordering_given == False:
		return freqs, [str(f[0]) for f in lins_freqs]

	raise eo.MyException("nothing returned here")

def write_new_ssm_phasing(ssm_phasing, output_file, overwrite=False):
	if overwrite == False:
		raise_if_file_exists(output_file)

	with open(output_file, "w") as f:
		f.write("SSM_index\tphase\n")
		for current_ssm in ssm_phasing:
			if current_ssm[1] == cons.A:
				phase = "A"
			elif current_ssm[1] == cons.B:
				phase = "B"
			else:
				phase = 0
			f.write("{0}\t{1}\n".format(current_ssm[0], phase))

def read_cnas(my_file, sorting_id_mapping=None, use_cna_indices=False, lin_num=-1):
	my_cnas = []
	first_line = True
	cna_index_count = 0
	with open(my_file, "r") as f:
		for line in f:
			if first_line:
				first_line = False
				continue
			if use_cna_indices is False:
				seg_index, chromosome, start, end, lineage, phase, change = line.rstrip().split("\t")
			else:
				cna_index, seg_index, lineage, phase, change = line.rstrip().split("\t")
				if int(cna_index) != cna_index_count:
					raise eo.MyException("CNAs have to be sorted in order of their indices, indices must go from "
						"0 to L-1, where L is the total number of CNAs.")
				chromosome = -1
				start = -1
				end = -1
				cna_index_count += 1
			cna = cnv.CNV(int(change), int(seg_index), int(chromosome), int(start), int(end))
			if int(change) < -1:
				raise eo.MyException("Loss cannot be smaller than -1 because of monotonicity restriction.")
			if use_cna_indices:
				cna.index = int(cna_index)
			if phase == "A":
				cna.phase = cons.A
			elif phase == "B":
				cna.phase = cons.B
			else:
				raise eo.MyException("undefined phase for CNA")
			if sorting_id_mapping is None:
				cna.lineage = int(lineage)
			else:
				try:
					cna.lineage = sorting_id_mapping[lineage]
				except KeyError:
					raise eo.MyException("Subclone ID {0} wasn't used in frequency matrix".format(lineage))
			if lin_num != -1:
				if cna.lineage >= lin_num:
					raise eo.MyException("CNA cannot be assign to subclone {0}. Only {1} subclones given.".format(
						cna.lineage, lin_num))

			my_cnas.append(cna)

	return my_cnas

def get_SSM_constraints(userSSM_file=None, my_ssms=None):
	ssm_num = len(my_ssms)

	first_line = True
	with open(userSSM_file, "r") as f:
		for line in f:
			if first_line:
				first_line = False
				continue
			index, phase = line.rstrip().split("\t")
			index = int(index)
			if index >= ssm_num:
				raise eo.MyException("User SSM constraints cannot be used. No SSM with index {0} "
					"exists.".format(index))

			# get user phase
			if phase == "A":
				phase = cons.A
			elif phase == "B":
				phase = cons.B
			else:
				raise eo.MyException("Phase type {0} for SSM {1} is invalid".format(phase, index))

			# SSM has no phasing yet
			if my_ssms[index].phase is None:
				my_ssms[index].phase = phase
			elif my_ssms[index].phase != phase:
				raise eo.MyException("According to CNA impact, SSM {0} must have phase {1}. This contradicts with "
					"user constraint for phase {2}".format(index, my_ssms[index].phase, phase))
			

def create_impact_matrix(impact_file=None, cna_num=-1, ssm_num=-1):
	
	# either no CNAs or SSMs exist, or no impact file is given
	if impact_file is None or cna_num == -1 or ssm_num == -1:
		return None

	# create empty impact matrix
	impact_matrix = np.zeros(ssm_num*cna_num).reshape(ssm_num, cna_num)

	# read impact file and update matrix
	first_line = True
	with open(impact_file, "r") as f:
		for line in f:
			if first_line:
				first_line = False
				continue
			ssm_index, cna_index = map(int, line.rstrip().split("\t"))
			if cna_index >= cna_num:
				raise eo.MyException("CNA index {0} is too high.".format(cna_index))
			if ssm_index >= ssm_num:
				raise eo.MyException("SSM index {0} is too high.".format(ssm_index))
			impact_matrix[ssm_index][cna_index] = 1

	return impact_matrix

def read_ssms(my_file, phasing=True, sorting_id_mapping=None, use_SSM_index=False, lin_num=-1):
	my_ssms = []
	first_line = True
	ssm_index_count = 0
	with open(my_file, "r") as f:
		for line in f:
			if first_line:
				first_line = False
				continue
			if phasing == True:
				seg_index, chromosome, pos, lineage, phase, cna_infl_same_lineage = line.rstrip().split("\t")
			elif use_SSM_index == False:
				seg_index, chromosome, pos, lineage = line.rstrip().split("\t")
			else:
				ssm_index, seg_index, lineage = line.rstrip().split("\t")
				if int(ssm_index) != ssm_index_count:
					raise eo.MyException("SSMs have to be sorted in order of their indices, indices must go from "
						"0 to J-1, where J is the total number of SSMs.")
				ssm_index_count += 1
				chromosome = -1
				(pos) = -1
			ssm = snp_ssm.SSM()
			ssm.chr = int(chromosome)
			ssm.pos = int(pos)
			ssm.seg_index = int(seg_index)
			if use_SSM_index:
				ssm.index = int(ssm_index)
			if phasing == True:
				if cna_infl_same_lineage == "0":
					ssm.infl_cnv_same_lin = False
				elif cna_infl_same_lineage == "1":
					ssm.infl_cnv_same_lin = True
				else:
					raise eo.MyException("undefined state for SSM")
				if phase == "A":
					ssm.phase = cons.A
				elif phase == "B":
					ssm.phase = cons.B
				elif phase == "0":
					ssm.phase = cons.UNPHASED
				else:
					raise eo.MyException("undefined phase for SSM")
			if sorting_id_mapping is None:
				ssm.lineage = int(lineage)
			else:
				try:
					ssm.lineage = int(sorting_id_mapping[lineage])
				except KeyError:
					raise eo.MyException("Subclone ID {0} is used for SSMs but wasn't defined in frequency matrix.".format(
						lineage))
			if lin_num != -1:
				if ssm.lineage >= lin_num:
					raise eo.MyException("SSM cannot be assign to subclone {0}. Only {1} subclones given.".format(
						ssm.lineage, lin_num))

			my_ssms.append(ssm)

	return my_ssms

def read_userZ(my_file, lin_num, sorting_id_mapping=None):
	z_matrix = np.zeros(lin_num*lin_num).reshape(lin_num,lin_num)
	first_line = True
	with open(my_file, "r") as f:
		for line in f:
			if first_line:
				first_line = False
				continue
			k, kp, v = list(map(int, line.rstrip().split("\t")))
			if sorting_id_mapping is not None:
				k = sorting_id_mapping[str(k)]
				kp = sorting_id_mapping[str(kp)]
			if v == 1:
				z_matrix[k][kp] = 1
			elif v == 0:
				z_matrix[k][kp] = -1
			else:
				raise eo.MyException("undefined relationship")

	return z_matrix

def read_userSSM(my_file, lin_num, seg_num):
	tmp_user_ssm = [[False] * lin_num for i in range(3)]
	user_ssm = [tmp_user_ssm for i in range(seg_num)]
	first_line = True
	with open(my_file, "r") as f:
		for line in f:
			if first_line:
				first_line = False
				continue
			seg_index, phase, lineage = line.rstrip().split("\t")
			if phase == "A":
				phase = cons.A
			elif phase == "B":
				phase = cons.B
			else:
				raise eo.MyException("phase undefined")
			user_ssm[int(seg_index)][phase][int(lineage)] = True

	return user_ssm

def print_ssm_phasing(my_lins, output_file, overwrite=False):
	if overwrite == False:
		raise_if_file_exists(output_file)

	my_ssms = []

	for k in range(1, len(my_lins)):
		get_ssm_information(my_lins[k].ssms, my_ssms)
		get_ssm_information(my_lins[k].ssms_a, my_ssms)
		get_ssm_information(my_lins[k].ssms_b, my_ssms)

	my_ssms = sorted(my_ssms, key=lambda x: (x[0], x[1], x[2]))
	
	with open(output_file, "w") as f:
		f.write("seg_index\tchr\tpos\tlineage\tphase\tcna_infl_same_lineage\n")
		for ssm in my_ssms:
			f.write("{0}\n".format("\t".join(list(map(str, ssm)))))
		
def get_ssm_information(ssm_list, info_list):
	for ssm in ssm_list:
		if ssm.phase == cons.A:
			phase = "A"
		elif ssm.phase == cons.B:
			phase = "B"
		else:
			phase = 0
		if ssm.infl_cnv_same_lin:
			cna_infl_same_lineage = 1
		else:
			cna_infl_same_lineage = 0
		info_list.append([ssm.seg_index, ssm.chr, ssm.pos, ssm.lineage, phase, cna_infl_same_lineage])
	return info_list


# parses a VCF file into a file Onctopus can read
# TODO old function, do I really need it?
def parse_vcf_file_for_onctopus(input_file_name, output_file_name, test):
	#open VCF-File as Reader
	vcf_reader= vcf.Reader(open(input_file_name,"r"))
	
	#open output as Writer if file doesn't exist (in a non test case)
	output = None
	if not test:
		raise_if_file_exists(output_file_name)
		output=open(output_file_name,"w")
	else:
		output=open(output_file_name,"w")
		#clear outputfile
		output.truncate()
	
	#write chrom, pos, variance count and reference count in outputfile
	for record in vcf_reader:
		output.write((record.CHROM.split("chr")[1]))
		output.write("\t")
		output.write(str(record.POS))
		output.write("\t")
		#get last field of tumour entry
		sample=record.samples[0].data[1]
		output.write(str(sample[1]))
		output.write("\t")
		output.write(str(sample[0]))
		output.write("\n")

	output.close()

# reads a vcf file
# slightly modified from https://github.com/morrislab/phylowgs/blob/master/
# parser/create_phylowgs_inputs.py
def read_vcf_file(input_file_name):
	vcf_reader = vcf.Reader(filename=input_file_name)
	records = []

	for variant in vcf_reader:
		variant.CHROM = variant.CHROM.upper()
		# Some VCF dialects prepend "chr", some don't. Remove the prefix to
		# standardize.
		if variant.CHROM.startswith('CHR'):
			variant.CHROM = variant.CHROM[3:]
		records.append(variant)
	return records

# variants: vcf object
# output_file_name: path where to write file for Onctopus
# gets vcf object and parses it to new file that is in a format that Onctopus expects
def create_onctopus_SSM_file_from_vcf_record(variants, output_file_name, test=False):
	#open output as Writer if file doesn't exist (in a non test case)
	output = None
	if not test:
		raise_if_file_exists(output_file_name)
		output=open(output_file_name,"w")
	else:
		output=open(output_file_name,"w")
		#clear outputfile
		output.truncate()

	# write sigle variants to file
	for i, variant in enumerate(variants):
		try:
			create_onctopus_SSM_line_from_variant_record(variant, output)
		except eo.ReadCountsUnavailableError as exc:
			logging.info("Variant {0} not added to Onctopus because of following error: {1}".format(
				i, exc))
	
	output.close()

# variant: vcf variant entry
# output: output stream
# parses information from variant entry and writes it to line in format
# that Onctopus expects: "chr \t pos \t var_count \t ref_count"
# modified from https://github.com/morrislab/phylowgs/blob/master/
# parser/create_phylowgs_inputs.py
def create_onctopus_SSM_line_from_variant_record(variant, output):
	if not ('t_alt_count' in variant.INFO and 't_ref_count' in variant.INFO):
		raise eo.ReadCountsUnavailableError("\'t_alt_count\' or \'t_ref_count\' don't exist.")
	# multiple entries for alternative and variant count
	single_entry = False
	try: 
		len(variant.INFO['t_alt_count'])
	except TypeError:
		single_entry = True
	if single_entry == False:
		assert len(variant.INFO['t_alt_count']) == len(variant.INFO['t_ref_count']) == 1
	else:
		assert type(variant.INFO['t_ref_count']) is int

	# get fields
	chromosome = variant.CHROM
	position = variant.POS
	if single_entry == False:
		alt_reads = int(variant.INFO['t_alt_count'][0])
		ref_reads = int(variant.INFO['t_ref_count'][0])
	else: 
		alt_reads = variant.INFO['t_alt_count']
		ref_reads = variant.INFO['t_ref_count']
	
	# Some variants havezero alt and ref reads.
	if alt_reads + ref_reads == 0:
		raise eo.ReadCountsUnavailableError("No reads for variant.")
	
	# write line
	output.write("{0}\t{1}\t{2}\t{3}\n".format(chromosome, position, alt_reads, ref_reads))

# writes an SSM file in the format Onctopus needs as input
def write_onctopus_ssm_file(file_name, ssm_list, test=False):
	
	if not test:
		 raise_if_file_exists(file_name)

	output_file = open(file_name, 'w')

	for ssm in ssm_list:
		output_file.write("{0}\t{1}\t{2}\t{3}\n".format(ssm.chr, ssm.pos,
			ssm.variant_count, ssm.ref_count))
	output_file.close()

# get a list of lineages objects and writes them to a result file
def write_lineages_to_result_file(file_name, my_lineages, test=False):

	if not test:
		raise_if_file_exists(file_name)

	result_file = open(file_name, 'w')
	for i in range(len(my_lineages)):
		result_file.write("@\n")
		result_file.write("LINEAGE: {0}\n".format(i))
		freq_index = my_lineages[i].freq
		result_file.write("FREQUENCY: {0:.25f}\n".format(my_lineages[i].freq))
		result_file.write("SUBLINEAGES: {0}\n".format(";".join(str(s) for s in 
			my_lineages[i].sublins)))
		# normal lineage
		if i == 0:
			# SNPs are written
			snps_line = get_SNP_line_from_lineage(my_lineages[0].snps)
			result_file.write("SNPS: {0}\n".format(snps_line))
			snps_a_line = get_SNP_line_from_lineage(my_lineages[0].snps_a)
			result_file.write("SNPS_A: {0}\n".format(snps_a_line))
			snps_b_line = get_SNP_line_from_lineage(my_lineages[0].snps_b)
			result_file.write("SNPS_B: {0}\n".format(snps_b_line))
			# no SSMs appear in normal lineage
			result_file.write("SSMS: \nSSMS_A: \nSSMS_B: \nCNVS_A: {0}\nCNVS_B: \n".format(
				";".join(get_CNV_lines_from_lineages(my_lineages[cons.NORMAL].cnvs_a))))
		# not the normal lineage
		else:
			# no SNPs appear in sublineages
			result_file.write("SNPS: \nSNPS_A: \nSNPS_B: \n")
			# write mutation of sublineages
			ssms_line = get_SSM_lines_from_lineages(my_lineages[i].ssms)
			result_file.write("SSMS: {0}\n".format(";".join(ssms_line)))
			ssms_a_line = get_SSM_lines_from_lineages(my_lineages[i].ssms_a)
			result_file.write("SSMS_A: {0}\n".format(";".join(ssms_a_line)))
			ssms_b_line = get_SSM_lines_from_lineages(my_lineages[i].ssms_b)
			result_file.write("SSMS_B: {0}\n".format(";".join(ssms_b_line)))
			cnvs_a_line = get_CNV_lines_from_lineages(my_lineages[i].cnvs_a)
			result_file.write("CNVS_A: {0}\n".format(";".join(cnvs_a_line)))
			cnvs_b_line = get_CNV_lines_from_lineages(my_lineages[i].cnvs_b)
			result_file.write("CNVS_B: {0}\n".format(";".join(cnvs_b_line)))

	result_file.close()

# given a list with segments, formats them in a line that is written to a result file
def get_CNV_lines_from_lineages(cnv_list):
	cnvs_line = []

	for i in range(len(cnv_list)):
		state = ""
		if cnv_list[i].change >= 1:
			state = "+{0}".format(str(cnv_list[i].change))
		elif cnv_list[i].change == -1:
			state = "-1"
		elif cnv_list[i].change == 0:
			state = "0"
		cnvs_line.append("{4},{0},{1},{2},{3}".format(cnv_list[i].seg_index, cnv_list[i].chr, cnv_list[i].start,
			cnv_list[i].end, state))
	return cnvs_line

# given a list with with SSMs, formats them in a line that is written to a result file
def get_SSM_lines_from_lineages(ssm_list):
	ssms_line = []

	for i in range(len(ssm_list)):
		current_ssm = ("{0},{1},{2}".format(ssm_list[i].seg_index, 
			ssm_list[i].chr, ssm_list[i].pos))
		# if SSM is influenced by CN gain in same lineage, the phase of the gain is added
		try:
			if ssm_list[i].infl_cnv_same_lin == True:
				current_ssm += ",infl"
		except AttributeError:
			pass
		ssms_line.append(current_ssm)
	return ssms_line

# given a list with with SNPs, formats them in a line that is written to a result file
def get_SNP_line_from_lineage(snp_list):
	snps = []

	for i in range(len(snp_list)):
		snps.append("{0},{1},{2}".format(snp_list[i].seg_index, snp_list[i].chr, 
			snp_list[i].pos))
	return ';'.join(snps)


###########################################################################################
############################  read_result_file  ###########################################

# result file is read
# '@' marks the start  of a new lineage
# each other row starts with an idenifier, according to this identifier the line is split
#	at the delimiter character ';' and according to the type of information in the
#	row (SNP, SSM, CNV) the information is further parsed and saved in corresponding
# 	objects (snp_ssm.SNP_SSM, segment.Segment) that are then put in a list
#	for the actual lineage
# phasing_not_known: if I don't know about the phasing of the SSMs because I used a different method than 
#	Onctopus
def read_result_file(file_name, phasing_not_known=False):
	lineages_list = []
	with open(file_name) as f:
		my_lineage  = None
		lineage_index = 0
		for line in f:
			split_line = line.rstrip("\n").split(":")
			tag = split_line[0]
			#print tag
			if tag.startswith("@"):
				# lineage is stored in list
				if my_lineage is not None:
					lineages_list.append(my_lineage)
					lineage_index += 1
				#print "new lineage"
				my_lineage = lineage.Lineage([], -1, [], [], [], [], [],
					[], [], [])
			else:
				# split_line[1] is information about mutations/frequency/sublins stored in line
				store_results(tag, my_lineage, split_line[1], phasing_not_known, lineage_index)
		# store last lineage in list
		lineages_list.append(my_lineage)

	return lineages_list

# phasing_not_known: if I don't know about the phasing of the SSMs because I used a different method than
#       Onctopus
def store_results(tag, my_lineage, line_part, phasing_not_known, lineage_index):
	line_part=line_part.lstrip()
	if tag == "LINEAGE":
		pass
	elif tag == "FREQUENCY":
		my_lineage.freq = float(line_part)
	elif tag == "SUBLINEAGES":
		my_lineage.sublins=get_sublineages_list_from_result_file(line_part)
	elif tag == "SNPS":
		my_lineage.snps = get_SNP_SSM_list_from_result_file(line_part, my_type=cons.SNP)
	elif tag == "SNPS_A":
		my_lineage.snps_a = get_SNP_SSM_list_from_result_file(line_part, my_type=cons.SNP)
	elif tag == "SNPS_B":
		my_lineage.snps_b = get_SNP_SSM_list_from_result_file(line_part, my_type=cons.SNP)
	elif tag == "SSMS":
		my_lineage.ssms = get_SNP_SSM_list_from_result_file(line_part, my_type=cons.SSM, 
			phasing_not_known=phasing_not_known, phase=cons.UNPHASED, lineage_index=lineage_index)
	elif tag == "SSMS_A":
		my_lineage.ssms_a = get_SNP_SSM_list_from_result_file(line_part, my_type=cons.SSM,
			phasing_not_known=phasing_not_known, phase=cons.A, lineage_index=lineage_index)
	elif tag == "SSMS_B":
		my_lineage.ssms_b = get_SNP_SSM_list_from_result_file(line_part, my_type=cons.SSM,
			phasing_not_known=phasing_not_known, phase=cons.B, lineage_index=lineage_index)
	elif tag == "CNVS_A":
		my_lineage.cnvs_a = get_CNV_list_from_result_file(line_part, cons.A)
	elif tag == "CNVS_B":
		my_lineage.cnvs_b = get_CNV_list_from_result_file(line_part, cons.B)

def write_result_file(lineages, file_name, test=False):
	#test, if output file exists
	if not test:
		raise_if_file_exists(file_name)

	with open(file_name, 'w') as f:
		#clear file in testcase
		if test:
			f.truncate()
		for lin_num, lineage in enumerate(lineages):
			# set all entries to valid type, if they were None
			if lineage.freq == None:
				lineage.freq = 0.0
			if lineage.sublins == None:
				lineage.sublins = []
			if lineage.snps == None:
				lineage.snps = []
			if lineage.snps_a == None:
				lineage.snps_a = []
			if lineage.snps_b == None:
				lineage.snps_b = []
			if lineage.ssms == None:
				lineage.ssms = []
			if lineage.ssms_a == None:
				lineage.ssms_a = []
			if lineage.ssms_b == None:
				lineage.ssms_b = []
			if lineage.cnvs_a == None:
				lineage.cnvs_a = []
			if lineage.cnvs_b == None:
				lineage.cnvs_b = []

			#write file
			f.write("@\n")
			f.write("LINEAGE: {0}\n".format(lin_num))
			f.write("FREQUENCY: {0:.5f}\n".format(lineage.freq))
			#SUBLINEAGES
			f.write("SUBLINEAGES: ")
			# every entry after the first needs ',' as seperator
			for num, sublin in enumerate(lineage.sublins):
				if num == 0:
					f.write(str(sublin))
				else:
					f.write(";{0}".format(sublin))
			# new line
			f.write("\n")
			#SNPS
			f.write("SNPS: ")
			for num, snp in enumerate(lineage.snps):
				# every entry after the first needs ';' as seperator
				if num == 0:
					f.write("{0},{1},{2}".format(snp.seg_index, snp.chr, snp.pos))
				else:
					f.write(";{0},{1},{2}".format(snp.seg_index, snp.chr, snp.pos))
			# new line
			f.write("\n")
			#SNPS_A
			f.write("SNPS_A: ")
			for num, snp in enumerate(lineage.snps_a):
				# every entry after the first needs ';' as seperator
				if num == 0:
					f.write("{0},{1},{2}".format(snp.seg_index, snp.chr, snp.pos))
				else:
					f.write(";{0},{1},{2}".format(snp.seg_index, snp.chr, snp.pos))
			# new line
			f.write("\n")
			#SNPS_B
			f.write("SNPS_B: ")
			for num, snp in enumerate(lineage.snps_b):
				# every entry after the first needs ';' as seperator
				if num == 0:
					f.write("{0},{1},{2}".format(snp.seg_index, snp.chr, snp.pos))
				else:
					f.write(";{0},{1},{2}".format(snp.seg_index, snp.chr, snp.pos))
			# new line
			f.write("\n")
			#SSMS
			f.write("SSMS: ")
			for num, ssm in enumerate(lineage.ssms):
				# every entry after the first needs ';' as seperator
				if num == 0:
					f.write("{0},{1},{2}".format(ssm.seg_index, ssm.chr, ssm.pos))
				else:
					f.write(";{0},{1},{2}".format(ssm.seg_index, ssm.chr, ssm.pos))
			# new line
			f.write("\n")
			#SSMS_A
			f.write("SSMS_A: ")
			for num, ssm in enumerate(lineage.ssms_a):
				# every entry after the first needs ';' as seperator
				if num == 0:
					f.write("{0},{1},{2}".format(ssm.seg_index, ssm.chr, ssm.pos))
				else:
					f.write(";{0},{1},{2}".format(ssm.seg_index, ssm.chr, ssm.pos))
			# new line
			f.write("\n")
			#SSMS_B
			f.write("SSMS_B: ")
			for num, ssm in enumerate(lineage.ssms_b):
				# every entry after the first needs ';' as seperator
				if num == 0:
					f.write("{0},{1},{2}".format(ssm.seg_index, ssm.chr, ssm.pos))
				else:
					f.write(";{0},{1},{2}".format(ssm.seg_index, ssm.chr, ssm.pos))
			# new line
			f.write("\n")
			#CNVS_A
			f.write("CNVS_A: ")
			for num, cnv in enumerate(lineage.cnvs_a):
				# every entry after the first needs ';' as seperator
				if num == 0:
					#+1 change needs extra character +
					if cnv.change == 1:
						f.write("+{0},{1},{2},{3},{4}".format(cnv.change, 
							cnv.seg_index, cnv.chr, cnv.start, cnv.end))
					else:
						f.write("{0},{1},{2},{3},{4}".format(cnv.change, 
							cnv.seg_index, cnv.chr, cnv.start, cnv.end))
				else:
					if cnv.change == 1:
						f.write(";+{0},{1},{2},{3},{4}".format(cnv.change, 
							cnv.seg_index, cnv.chr, cnv.start, cnv.end))
					else:
						f.write(";{0},{1},{2},{3},{4}".format(cnv.change, 
							cnv.seg_index, cnv.chr, cnv.start, cnv.end))
			# new line
			f.write("\n")
			#CNVS_B
			f.write("CNVS_B: ")
			for num, cnv in enumerate(lineage.cnvs_b):
				# every entry after the first needs ';' as seperator
				if num == 0:
					#+1 change needs extra character +
					if cnv.change == 1:
						f.write("+{0},{1},{2},{3},{4}".format(cnv.change, 
							cnv.seg_index, cnv.chr, cnv.start, cnv.end))
					else:
						f.write("{0},{1},{2},{3},{4}".format(cnv.change, 
							cnv.seg_index, cnv.chr, cnv.start, cnv.end))
				else:
					if cnv.change == 1:
						f.write(";+{0},{1},{2},{3},{4}".format(cnv.change, 
							cnv.seg_index, cnv.chr, cnv.start, cnv.end))
					else:
						f.write(";{0},{1},{2},{3},{4}".format(cnv.change, 
							cnv.seg_index, cnv.chr, cnv.start, cnv.end))
			# new line
			f.write("\n")

# write the lineages list to JSON file
def write_result_file_as_JSON(lineages, file_name, test=False):
	#test, if output file exists
	if not test:
		raise_if_file_exists(file_name)

	# create serializable structure
	seri = [my_lin.create_dict() for my_lin in lineages]

	# write to file
	with open(file_name, "w") as f:
		json.dump(seri, f)

# reads a result file in json formate and returns a list of lineages
def read_JSON_result_file(file_name):
	with open(file_name, "r") as f:
		lineages_json = json.load(f)

	my_lineages = []
	for lin in lineages_json:
		ssms = [create_ssm_from_json_dump(current_ssm) for current_ssm in lin["ssms"]]
		ssms_a = [create_ssm_from_json_dump(current_ssm) for current_ssm in lin["ssms_a"]]
		ssms_b = [create_ssm_from_json_dump(current_ssm) for current_ssm in lin["ssms_b"]]
		cnvs_a = [create_cnv_from_json_dump(current_cnv) for current_cnv in lin["cnvs_a"]]
		cnvs_b = [create_cnv_from_json_dump(current_cnv) for current_cnv in lin["cnvs_b"]]
		my_lineages.append(lineage.Lineage(lin["sublins"], lin["freq"], cnvs_a, cnvs_b, [], [], [],
			ssms, ssms_a, ssms_b))
	
	return my_lineages

# given the information of an CNV object in a dictionary, create an CNV object
def create_cnv_from_json_dump(cnv_dict):
	my_cnv = cnv.CNV(cnv_dict["change"], cnv_dict["seg_index"], cnv_dict["chr"], cnv_dict["start"], cnv_dict["end"])
	my_cnv.phase = cnv_dict["phase"]

	return my_cnv

# given the information of an SSM object in a dictionary, create an SSM object
def create_ssm_from_json_dump(ssm_dict):
	my_ssm = snp_ssm.SSM()
	my_ssm.chr = ssm_dict["chr"]
	my_ssm.pos = ssm_dict["pos"]
	my_ssm.variant_count = ssm_dict["variant_count"]
	my_ssm.seg_index = ssm_dict["seg_index"]
	my_ssm.infl_cnv_same_lin = ssm_dict["infl_cnv_same_lin"]
	my_ssm.phase = ssm_dict["phase"]
	my_ssm.lineage = ssm_dict["lineage"]

	return my_ssm


def get_sublineages_list_from_result_file(line_part):
	if line_part == "":
		return []
	sublins=line_part.split(";")
	return list(map(int,sublins))

def get_CNV_list_from_result_file(line_part, phase=-1):
	cnv_list = []
	if line_part == "":
		return cnv_list
	cnvs = line_part.split(";")
	for i in range(len(cnvs)):
		(change, seg_index, chr, start, end) = cnvs[i].split(",")
		cnv_list.append(cnv.CNV(int(change), int(seg_index), int(chr), int(start), int(end)))
		cnv_list[-1].phase = phase
	return cnv_list
		
# phasing_not_known: if I don't know about the phasing of the SSMs because I used a different method than Onctopus
def get_SNP_SSM_list_from_result_file(line_part, my_type=cons.SSM, phasing_not_known=False, phase=None,
	lineage_index=-1):
	snp_list = []
	if line_part == "":
		return snp_list
	snps = line_part.split(";")
	for i in range(len(snps)):
		if my_type == cons.SSM:
			snp_ssm_object = snp_ssm.SSM()
		else:
			snp_ssm_object = snp_ssm.SNP()

		# to store influence of CN gain in same lineage
		infl  = None

		try:
			(seg_index, chr, pos) = snps[i].split(",")
		except ValueError:
			# influence of CN gain in same lineage is also stored in result file
			(seg_index, chr, pos, infl) = snps[i].split(",")

		snp_ssm_object.chr = int(chr)
		snp_ssm_object.pos = int(pos)
		snp_ssm_object.seg_index = int(seg_index)

		# if influence of CN gain in same lineage exists, it is stored
		if infl is not None:
			snp_ssm_object.infl_cnv_same_lin = True
		# if phasing information about the SSM is known, set it
		if phasing_not_known == False:
			snp_ssm_object.phase = phase
		# set lineage index
		snp_ssm_object.lineage = lineage_index

		snp_list.append(snp_ssm_object)
	return snp_list
				
# given a lineage file (input_file) a new lineage file (output_file) is created
# that contains only those mutations that appear on the segment with index seg_index
def create_result_file_for_seg_x(input_file, output_file, seg_index, test=False):
	# read result file
	my_lins = read_result_file(input_file)

	# only use SSMs and CNVs that are in right segment
	for lin in my_lins:
		lin.cnvs_a = get_mutations_for_seg_x(lin.cnvs_a, seg_index)
		lin.cnvs_b = get_mutations_for_seg_x(lin.cnvs_b, seg_index)
		lin.ssms = get_mutations_for_seg_x(lin.ssms, seg_index)
		lin.ssms_a = get_mutations_for_seg_x(lin.ssms_a, seg_index)
		lin.ssms_b = get_mutations_for_seg_x(lin.ssms_b, seg_index)

	# write result file
	write_result_file(my_lins, output_file, test=test)

def get_mutations_for_seg_x(mutations, seg_index):
	new_mutations = []
	for mut in mutations:
		if mut.seg_index == seg_index:
			new_mutations.append(mut)
	return new_mutations

############################  read_result_file  ###########################################
###########################################################################################
	
def read_segment_file(file_name, allele_specific=False):
	segment_list = []
	with open(file_name) as f:
		for line in f:
			if allele_specific:
				(chr, start, end, given_cn_A, standard_error_A, given_cn_B,
					standard_error_B) = (line.rstrip()).split('\t')
				# check that CN of A is bigger equal CN of B, if not, swap
				if given_cn_A < given_cn_B:
					(given_cn_A, standard_error_A, given_cn_B, standard_error_B) = (
						given_cn_B, standard_error_B, given_cn_A, standard_error_A)
					logging.info("CN A needs to be larger or equal than CN B. Swapping "
						"CNs for line: {0}".format(line.rstrip()))
				segment_list.append(segment.Segment_allele_specific(int(chr),
					int(start), int(end), float(given_cn_A),
					float(standard_error_A), float(given_cn_B),
					float(standard_error_B)))
			else:
				(chr, start, end, c, hm) = (line.rstrip()).split('\t') 
				segment_list.append(segment.Segment(int(chr), int(start), 
					int(end), float(c), float(hm)))
	return segment_list

def read_snp_ssm_file(file_name, mut_type):
	mut_list = []
	with open(file_name) as f:
		for line in f:
			(chr, pos, c1, c2) = (line.rstrip()).split('\t')
			mut = snp_ssm.SNP()
			if (mut_type == cons.SSM):
				mut = snp_ssm.SSM()
			mut.set_all_but_seg_index(int(chr), int(pos), int(c1), int(c2))
			mut_list.append(mut)
	return mut_list

def raise_if_file_exists(file_name):
	if os.path.isfile(file_name):
		error = "File {0} does already exist.\n".format(file_name)
		raise(eo.FileExistsException(error))
	return True

def raise_if_file_not_exists(file_name):
	if not os.path.isfile(file_name):
		error = "File {0} does not exist.\n".format(file_name)
		raise(eo.FileDoesNotExistException(error))
	return True

# for the simulation of data
def write_segment_file(reads_total, seg_start, seg_end, segment_number, mass, file_name, no_test):
	chr = 1
	if no_test:
		raise_if_file_exists(file_name)
	with open(file_name, 'w') as f:
		for seg in range(segment_number):
			start = seg_start[seg]
			end = seg_end[seg]
			line = "{0}\t{1}\t{2}\t{3}\t{4}\n".format(chr, start, end, reads_total[seg], mass[seg])
			f.write(line)

# for the simulation of data
def write_segment_file_allele_specific(seg_start, seg_end, segment_number, cn_A, cn_B,
	standard_error_A, standard_error_B, file_name, no_test):
	chr = 1
	if no_test:
		raise_if_file_exists(file_name)
	with open(file_name, 'w') as f:
		for seg in range(segment_number):
			start = seg_start[seg]
			end = seg_end[seg]
			line = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(chr, start, end, cn_A[seg], 
				standard_error_A[seg], cn_B[seg], standard_error_B[seg])
			f.write(line)

# write file from segments
def write_segment_file_allele_specific_from_segments(segments, file_name):
	raise_if_file_exists(file_name)

	with open(file_name, 'w') as f:
		for seg in segments:
			line = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(seg.chr, seg.start, seg.end, seg.given_cn_A,
				seg.standard_error_A, seg.given_cn_B, seg.standard_error_B)
			f.write(line)


#for the simulation of data
# in allele-specific CN case, SNPs are not used
def write_single_muts_file(reads_total, lineages, seg_start, segment_number, file_name_snp, file_name_ssm, 
	no_test, allele_specific=False):
	chr = 1
	if no_test:
		if not allele_specific:
			raise_if_file_exists(file_name_snp)
		raise_if_file_exists(file_name_ssm)

	if not allele_specific:
		f_snp = open(file_name_snp, 'w')
	f_ssm = open(file_name_ssm, 'w')

	for seg in range(segment_number):
		for pha in range(cons.PHASE_NUMBER):
			for lin_index, lin in enumerate(lineages):
				mut = cons.SNP
				if (lin_index != cons.NORMAL):
					mut = cons.SSM
				elif allele_specific:
					continue
				for i in range(len(lin.segments[pha][seg][mut])):
					line = "{0}\t{1}\t{2}\t{3}\n".format(chr, lin.segments[pha][seg][mut][i].pos,
						lin.segments[pha][seg][mut][i].variant_count,
						lin.segments[pha][seg][mut][i].ref_count)
					if (mut == cons.SNP):
						f_snp.write(line)
					else:
						f_ssm.write(line)
	if not allele_specific:
		f_snp.close()
	f_ssm.close()

#for the simulation of data
def get_sublineages_line(sublins):
	line = ["SUBLINEAGES: "]
	if (len(sublins) > 1):
		for i in range(len(sublins) - 1):
			item = "{0};".format(sublins[i])
			line.append(item)
	if (len(sublins) > 0):
		item = "{0}".format(sublins[-1])
		line.append(item)
	line.append("\n")
	return ''.join(line)

# for the simulation of data
# returns end of the line that holds information about the mutations in all segments according
#	 to phases
def get_end_of_mutation_line(line, mut, lin, pha, segment_number, seg_start, seg_end):
	first_mut = True
	item = []
	for seg in range(segment_number):
		mut_count = lin.get_mut_count(pha, seg, mut)
		for i in range(mut_count):
			# first mutation
			if first_mut:
				# 1) mutation is CNV
				if (mut == cons.CNV):
					state = lin.segments[pha][seg][mut][0]
					# 1 stands for chromosome
					item = "{0},{1},1,{2},{3}".format(state, seg, seg_start[seg], 
						seg_end[seg])
				# 3) mutation is SNP or SSM
				else:
					# here, chromosome is gotten from SNP/SSM but it was set to 1 before
					# so CNVs and SNPs/SSMs are on the same chromosome
					item = "{2},{0},{1}".format(lin.segments[pha][seg][mut][i].chr, 
						lin.segments[pha][seg][mut][i].pos, seg)
					# influence of CN gain in same lineage is added
					try:
						if lin.segments[pha][seg][mut][i].infl_cnv_same_lin == True:
							item += ",infl"
					except AttributeError:
						pass
				first_mut = False
			# not the first mutation
			else:
				# 2) mutation is CNV
				if (mut == cons.CNV):
					state = lin.segments[pha][seg][mut][0]
					item = ";{0},{1},1,{2},{3}".format(state, seg, seg_start[seg], 
						seg_end[seg])
				# 4) mutation is SNP or SSM
				else:
					item = ";{2},{0},{1}".format(lin.segments[pha][seg][mut][i].chr,
						lin.segments[pha][seg][mut][i].pos, seg)
					# influence of CN gain in same lineage is added
					try:
						if lin.segments[pha][seg][mut][i].infl_cnv_same_lin == True:
							item += ",infl"
					except AttributeError:
						pass
			line.append(item)
	line.append("\n")
	return ''.join(line)
					
# for the simulation of data
# writes a file that stores the results of the simulation
def write_simulation_results(seg_start, seg_end, lineages, segment_number, file_name, no_test):
	if no_test:
		raise_if_file_exists(file_name)
	
	with open(file_name, 'w') as f:
		for lin_index, lin in enumerate(lineages):
			f.write("@\n")
			f.write("LINEAGE: {0}\n".format(lin_index))
			f.write("FREQUENCY: {0}\n".format(lin.freq))
			f.write(get_sublineages_line(lin.sublin))
			for mut in range(cons.MUTATION_TYPE_NUM):
				mut_name = ""
				if (mut == cons.SNP):
					mut_name = "SNPS_"
				elif (mut == cons.SSM):
					mut_name = "SSMS_"
				else:
					mut_name = "CNVS_"
				for pha in range(cons.PHASE_NUMBER):
					pha_name = ""
					if (pha == cons.A):
						pha_name = "A: "
					else:
						pha_name = "B: "
					line = []
					mut_line = get_end_of_mutation_line(line, mut, lin, pha, 
						segment_number, seg_start, seg_end)
					f.write(''.join([mut_name, pha_name, mut_line]))

# for simulation of data
def write_simulation_info(lineages_file, segment_number, snp_number,
        ssm_number, cnv_number, noise, CN_noise,
	mass, output_segments, output_snps,
        output_ssms, output_results, output_info, no_test, new_version,
        overdispersion, coverage_overdispersion, frequency_overdispersion,
        CNV_assignment, SNP_assignment, SSM_assignment, allele_specific,
        SSM_num_per_unit, clonal_cn_percentage, p1_A_prop, p1_A_B_prop,
	m1_B_prop, m1_A_B_prop, p1_m1_prop, SSM_before_CNV_LH, addSSMsAccoringToFreqs=False,
	clonal_ssm_percentage=-1, CNAs_mult_lin_prop=0):

	if no_test:
		raise_if_file_exists(output_info)

	with open(output_info, 'w') as f:
		f.write('lineages_file: {0}\nsegment_number: {1}\nsnp_number: {2}\nssm_number: {3}\n'.
			format(lineages_file, segment_number, snp_number, ssm_number))
		f.write('cnv_number: {0}\nnoise: {1}\nCN noise: {2}\nmass: {3}\n'
			'output_segments: {4}\noutput_snps: {5}\n'.format(
			cnv_number,  noise, CN_noise, mass, output_segments, 
			output_snps))
		f.write('output_ssms: {0}\noutput_results: {1}\noutput_info: {2}\nno_test: {3}\n'.format(
			output_ssms, output_results, output_info, no_test))
		f.write('new_version: {0}\noverdispersion: {1}\ncoverage_overdispersion: {2}\n'.format(
			new_version, overdispersion, coverage_overdispersion))
		f.write('frequency_overdispersion: {0}\nCNV_assignment: {1}\nSNP_assignment: {2}\n'.format(
			frequency_overdispersion, CNV_assignment, SNP_assignment))
		f.write('SSM_assignment: {0}\nallele_specific: {1}\nSSM_num_per_unit: {2}\n'
			'clonal CN frequency: {3}\nproportion plus 1 A: {4}\n'
			'proportion plus 1 A and B: {5}\nproportion minus 1 B: {6}\n'
			'proportion minus 1 A and B: {7}\nproportion copy-neutral LOH: {8}\n'
			'proportion of two CNAs in different lineages: {10}\n'
			'likelihood SSM before CNV: {9}\n'
			.format(SSM_assignment, allele_specific, SSM_num_per_unit, clonal_cn_percentage,
			p1_A_prop, p1_A_B_prop, m1_B_prop, m1_A_B_prop, p1_m1_prop,
			SSM_before_CNV_LH, CNAs_mult_lin_prop))
		if addSSMsAccoringToFreqs == True:
			f.write("Lineages of SSMs were chosen somewhat according to their frequencies.")
		elif clonal_ssm_percentage != -1:
			f.write("{0} % of SSMs were assigned to the first lineage, the others were assigned "
				"uniformly to the other lineages.".format(clonal_ssm_percentage	))
		else:
			f.write("Lineages of SSMs were chosen uniformly.")

# for simulation of data
# format:
# sublin1,sublin2,...;freq
def read_lineages_tree(file_name, segment_number):
	lin = []

	with open(file_name) as f:
		for line in f:
			(sublins, freq) = (line.rstrip()).split(';')
			if (len(sublins) > 0):
				sublins = list(map(int, sublins.split(',')))
			else:
				sublins = []
			lin.append(lineage_for_data_simulation.Lineage_Simulation(sublins, 
				float(freq), segment_number))
	return lin

# for simulation of data
# format:
# lineage, phase, segment, state
def read_mutation_assignments(file_name):
	assignment = []

	with open(file_name) as f:
		for line in f:
			line_parts = (line.rstrip()).split(',')
			lin = line_parts[0]
			pha = line_parts[1]
			seg = line_parts[2]
			state = None
			# when lines assignes CNVs, it has 4 entries
			if len(line_parts) == 4:
				state = line_parts[3]

			# phases are rewritten in constants
			if pha == "A":
				pha = cons.A
			elif pha == "B":
				pha = cons.B
			else:
				raise ValueError('Phase cannot be written')

			# assignment describes CNVS
			if len(line_parts) == 4:
				assignment.append([int(lin), pha, int(seg), state])
			# assignment describes SNPs/SSMs
			else:
				assignment.append([int(lin), pha, int(seg)])
	
	return assignment
###############################################################################################
###############################################################################################
###############################################################################################
# for simulation
# given a result file, assignment files for CNVs, SSMs, and SNPs are created based on the
# mutation's assignment in the result file
def create_mutation_assignments_from_result_files(result_file_name, cnv_assignment_name, snp_assignment_name,
	ssm_assignment_name, no_test=True):

	raise_if_file_not_exists(result_file_name)

	# check if files exist
	if no_test:
		raise_if_file_exists(cnv_assignment_name)
		raise_if_file_exists(ssm_assignment_name)
		raise_if_file_exists(ssm_assignment_name)

	# open files in which assignments will be written
	cnv_assignment_file = open(cnv_assignment_name, 'w')
	snp_assignment_file = open(snp_assignment_name, 'w')
	ssm_assignment_file = open(ssm_assignment_name, 'w')

	my_lineage = read_result_file(result_file_name)

	# go through all lineages
	for lin_index, lin in enumerate(my_lineage):
		# write CNVs assignments to file, independant of the lineage
		write_cnv_assignment(lin.cnvs_a, cnv_assignment_file, lin_index, "A")
		write_cnv_assignment(lin.cnvs_b, cnv_assignment_file, lin_index, "B")
		# in normal lineage, the SNPs are considered
		if lin_index == cons.NORMAL:
			# write SNP assignments to file
			# currently, unphased SNPs are phased to A in output
			write_snp_ssm_assignment(lin.snps, snp_assignment_file, lin_index, "A")
			write_snp_ssm_assignment(lin.snps_a, snp_assignment_file, lin_index, "A")
			write_snp_ssm_assignment(lin.snps_b, snp_assignment_file, lin_index, "B")
		# in the non-normal lineages, the SSMs are considered
		else:
			# write SSM assignments to file
			# currently, unphased SSMs are phased to A in output
			write_snp_ssm_assignment(lin.ssms, ssm_assignment_file, lin_index, "A")
			write_snp_ssm_assignment(lin.ssms_a, ssm_assignment_file, lin_index, "A")
			write_snp_ssm_assignment(lin.ssms_b, ssm_assignment_file, lin_index, "B")

	cnv_assignment_file.close()
	snp_assignment_file.close()
	ssm_assignment_file.close()

# writes SNPs and SSMs to file given a list of them
def write_snp_ssm_assignment(mut_list, assignment_file, lin_index, phase):
	for mut in mut_list:
		assignment_file.write("{0},{1},{2}\n".format(lin_index, phase, mut.seg_index))


# writes CNV assignments to file given a list with CNV objects
def write_cnv_assignment(cnv_list, cnv_assignment_file, lin_index, phase):
	# for every CNV in list
	for my_cnv in cnv_list:
		# the CNV change must be parsed into a string
		change = "0"
		if my_cnv.change == 1:
			change = "+1"
		elif my_cnv.change == -1:
			change = "-1"
		elif my_cnv.change == 0:
			change = "0"
		else:
			raise oe.myException("Unknown change in create_cnv_assignment.")

		# current assignment is written to file
		cnv_assignment_file.write("{0},{1},{2},{3}\n".format(lin_index, phase,
			my_cnv.seg_index, change))

##### create_mutation_assignments_from_result_files ###########################################
###############################################################################################
###############################################################################################


def str_to_bool(s):
	if (s == 'True'):
		return True
	elif (s == 'False'):
		return False
	else:
		raise ValueError('String cannot be converted to boolean.')

# if string is 'None' type None is returned, otherwise string stays what it is
def str_possibly_to_none(s):
	if s == 'None':
		return None
	return s

# function to read a fixed value file, regardless of data(CNV, SNP, SSM, Z)
def read_fixed_value_file(fixed_file):
	logging.info("Reading fixed file {0}".format(fixed_file))
	fixed_value_list = []
	unfixed_start = -1
	unfixed_stop = -1
	with open(fixed_file) as f:
		for line in f:
			# get unfixed segment info from header
			if "# unfixed_segment" in line:
				line_split = line.rstrip("\n").split(',')
				unfixed_start = int(line_split[1].split(':')[1])
				unfixed_stop = int(line_split[2].split(':')[1])
			# ignore other comment lines
			elif "#" in line:
				pass
			# if line is non-empty
			elif line != '\n':
				if not '*' in line:
					line_list = line.rstrip("\n").split("\t")
					fixed_value_list.extend(line_list)

	# convert all entries to float
	return (list(map(float, fixed_value_list), unfixed_start, unfixed_stop))


# creates a temporary random file name
def create_temp_rand_name(prefix=""):
	return "{0}tmp_{1}".format(prefix, random.random())

# creates fixed files of various types based on the lineages
def create_some_fixed_files(lineages, fixed_cnv_file=None, fixed_ssm_file=None, 
	fixed_z_matrix_file=None, fixed_phi_file=None):

	if fixed_cnv_file:
		submarine.create_fixed_file(lineages, cons.CNV, output_file=fixed_cnv_file)
	if fixed_ssm_file:
		submarine.create_fixed_file(lineages, cons.SSM, output_file=fixed_ssm_file)
	if fixed_z_matrix_file:
		submarine.create_fixed_file(lineages, cons.Z, output_file=fixed_z_matrix_file)
	if fixed_phi_file:
		submarine.create_fixed_file(lineages, cons.FREQ, output_file=fixed_phi_file)

# removes given fixed files
def remove_some_fixed_files(fixed_cnv_file=None, fixed_ssm_file=None, 
	fixed_z_matrix_file=None, fixed_phi_file=None):
	if fixed_cnv_file:
		os.remove(fixed_cnv_file)
	if fixed_ssm_file:
		os.remove(fixed_ssm_file)
	if fixed_z_matrix_file:
		os.remove(fixed_z_matrix_file)
	if fixed_phi_file:
		os.remove(fixed_phi_file)


# function to create a fixed_value file for every data-type, only needs file_type if Z-data is written
def write_fixed_value_file(data, output_file, row_num, column_num, info, unfixed_segment=-1, unfixed_start=-1, unfixed_stop=-1, file_type=None, test=False):
	output = None

	# open output as Writer if file doesn't exist (in a non test case)
	if not test:
		raise_if_file_exists(output_file)
		output = open(output_file, "w")
	else:
		output = open(output_file,"w")
		# clear outputfile in test case
		output.truncate()

	#write info line(header)
	output.write("# {0}\n".format(info))
	# info about unfixed segment
	if unfixed_start > unfixed_stop:
		unfixed_start = -1
		unfixed_stop = -1
	if file_type == cons.CNV or file_type == cons.SNP or file_type == cons.SSM:
		output.write("# unfixed_segment: {0}, unfixed_start: {1}, unfixed_stop: {2}\n".format(
			unfixed_segment, unfixed_start, unfixed_stop))

	# write fixed Z matrix
	if file_type == cons.Z:
		# check if all enries are fixed
		if (unfixed_start != unfixed_stop) and (unfixed_start != -1):
			raise eo.MyException("All entries need to be fixed for Z matrix.")
		start_index = stop_index = 0
		while column_num > 0:
			# line break after each line that isn't the first one
			if stop_index > 0:
				output.write("\n")

			# calculate new parts of data list that should be used
			start_index = stop_index
			stop_index = start_index + column_num
			# take parts of data list
			output.write("\t".join(list(map(str,data[start_index:stop_index]))))
			# decrease number of columns
			column_num -= 1

	# write other fixed file types
	else:
		row_index = 0
		# stepwise write separation character and value or indication that variable should
		# not be fixed
		for entry_index, entry in enumerate(data):
			# check if entry isn't last of line or end of matrix
			if entry_index % column_num != 0:
				output.write("\t")
			# check if entry is last of line but not of matrix
			elif entry_index % (row_num * column_num) != 0:
				output.write("\n")
				row_index += 1
				############### to delete ######################################################
				#if file_type == cons.Z:
				#	# Z-file has one less column to the previous row
				#	column_num -= 1
			# entry is last of matrix
			else:
				if entry_index != 0:
					output.write("\n\n")
					row_index = 0
			# write '*' for values that should stay unfixed
			if row_index >= unfixed_start and row_index <= unfixed_stop:
				output.write('*')
			else:
				output.write(str(entry))

	output.close()

# transformes a resultfile to a specific format that is needed to apply the 2A
# metric of the SMC-Het scoring harness
# format: 1\n1\n2\n3... indices of lineages, where SSMs appear
def resultfile_to_2A_file(input_name, output_name, test=False):
	lineages = read_result_file(input_name)
	# get all ssms and their corresponding lineage
	ssms = []
	for index, lineage in enumerate(lineages):
		for ssm in lineage.ssms:
			ssms.append((ssm, index))
		for ssm in lineage.ssms_a:
			ssms.append((ssm, index))
		for ssm in lineage.ssms_b:
			ssms.append((ssm, index))

	# sort the ssms
	ssms.sort()

	# check if file doesn't exist (in a non test case)
	if not test:
		raise_if_file_exists(output_name)
	# open file(in test case, clear file)
	with open(output_name, 'w') as writer:
		if test:
			writer.truncate()

		# write the lineage number of all ssms in the file
		for (_, index) in ssms:
			writer.write(str(index) + "\n")

# transformes a resultfile to a specific format that is needed to apply the 2A
# metric of the SMC-Het scoring harness
# format: True\nTrue\nTrue... "true" for each SSM
def resultfile_to_pseudo_VCF_for_2A(input_name, output_name, test=False):
	lineages = read_result_file(input_name)
	# count the number of ssms
	ssms = 0
	for lineage in lineages:
		ssms += len(lineage.ssms)
		ssms += len(lineage.ssms_a)
		ssms += len(lineage.ssms_b)

	# create file with entry "True" for every ssm
	# check if file doesn't exist (in a non test case)
	if not test:
		raise_if_file_exists(output_name)
	# open file(in test case, clear file)
	with open(output_name, 'w') as writer:
		if test:
			writer.truncate()

		for _ in range(ssms):
			writer.write("True\n")

# computes the average coverage of a SNP_count file
# format: # CHR \t POS \t Count_A \t Count_C \t Count_G \t Count_T \t Good_depth 
def compute_av_coverage_from_SNP_count(file_name):
	coverage = 0 
	line_count = 0

	with open(file_name) as f:
		for line in f: 
			if line_count > 0:
				coverage_entry = int(line.rstrip("\n").split("\t")[-1])
				coverage += coverage_entry 
			line_count = line_count + 1

	coverage = float(coverage) / (line_count - 1)
	return coverage

# return the SNP index from the input file as dictionary
def read_SNP_index(file_name):
	index = dict()
	with open(file_name, 'r') as f:
		for line in f:
			# tab is seperator in the index file
			split_line = line.rstrip("\n").split("\t")
			# 0 entry is chromosom, 1 entry is position both are the key. 
			# 2 entry is reference base and 3 entry is variance base
			index[(int(split_line[0]), int(split_line[1]))] = (split_line[2].upper(), 
				split_line[3].upper())
	return index

def SNP_count_to_BAF(input_SNPs_file, index, output_BAF_file, test=False):
	# dictionary for base and index_number in inputfile
	mapping_base_to_index = {"A":2, "C":3, "G":4, "T":5}
	mapping_index_to_base = {2:"A", 3:"C", 4:"G", 5:"T"}
	# test if file exist, raise exception on non testcase
	if not test:
		raise_if_file_exists(output_BAF_file)

	# open input file and output file. If test case clear output file
	with open(input_SNPs_file, 'r') as input_file:
		with open(output_BAF_file, 'w') as output_file:
			if test:
				output_file.truncate()
		
			# skip header of input
			input_file.readline()
			# write output header
			output_file.write("\tchrs\tpos\tsample")
			
			# offset for skipped entries
			offset = 0
			# iterate enries in input with start number of 1
			for line_num, line in enumerate(input_file, start = 1):
				# tab is seperator for entries. split line on seperator and convert all 
				# entries to int
				split_line = list(map(int, line.rstrip("\n").split("\t")))
				# get variance and reference from index
				if index.has_key((split_line[0], split_line[1])):
					(reference, variance) = index[(split_line[0], split_line[1])]
				else:
					print("ERROR: Skipping SNP(chr: {0}, pos: {1}) with no entry in indexfile".format(split_line[0], split_line[1], index))
					#increment offset
					offset += 1
					#skip this line
					continue
				
				# iterate over all base_counts and check if they are 0 or reference/variance
				for i in range(2,6):
					# entry not 0
					if split_line[i] != 0:
						# entry is not reference or variance
						if i != mapping_base_to_index[reference] and i != mapping_base_to_index[variance]:
							# variance is 0
							if split_line[mapping_base_to_index[variance]] == 0:
								# change variance
								variance = mapping_index_to_base[i]
								# message the change of variance
								print("SNP(chr: {0}, pos: {1}) has variance_count 0. Set new variance({2}) with count = {3}.").format(split_line[0], split_line[1], variance, split_line[i])
								# terminate loop
								break
							# variance is not 0
							else:
								print("WARNING: SNP(chr: {0}, pos: {1}) has an entry unequal 0, which isn't the variance or reference.".format(split_line[0], split_line[1]))
								#terminate loop
								break

				# try to compute BAF
				try:
					sample = submarine.compute_BAF(split_line[mapping_base_to_index[variance]], split_line[6])
				# catch BAF exceptions, print error message and set sample to 0 
				except eo.BAFComputationException as ex:
					print("BAF computation failed for SNP(chr: {0}, pos: {1}) with message:{2}".format(split_line[0], split_line[1], ex))
					sample = 0

				# write results into file/ SNP-num; chromosome; position; BAF
				output_file.write("\n{0}\t{1}\t{2}\t{3}".format(line_num-offset, split_line[0], split_line[1], sample))

def SNP_count_to_LogR_new(input_SNPs_file, output_LogR_file, avg_coverage=0, normal=True, test=False,
	normal_SNP_file=None):
	my_compute_LogR = None
	if normal:
		my_compute_LogR = submarine.compute_LogR_normal
	else:
		my_compute_LogR = submarine.compute_LogR_tumor
	my_compute_LogR_star = submarine.compute_LogR_star

	# test if file exist, raise exception on non testcase
	if not test:
		raise_if_file_exists(output_LogR_file)
	
	# read input file
	input_lines = []
	with open(input_SNPs_file, 'r') as f:
		input_lines = f.readlines()

	# get coverage from input file
	coverage = [int(line.rstrip().split("\t")[6]) for line in input_lines[1:]]

	# if LogR for tumor file should be computed, normal file also has to be read
	# and coverage is also needed
	normal_lines = []
	normal_coverage = []
	if not normal:
		with open(normal_SNP_file, 'r') as f:
			normal_lines = f.readlines()  
		normal_coverage = [int(line.rstrip().split("\t")[6]) for line in normal_lines[1:]]

	# compute logR
	logR = []
	if normal:
		logR = [my_compute_LogR(cov, avg_coverage) for cov in coverage]
	else:
		logR = [my_compute_LogR(coverage[i], normal_coverage[i]) for i in range(len(coverage))]  
	my_median = submarine.median_list(logR)
	logR_star = [my_compute_LogR_star(logR_value, my_median) for logR_value in logR]

	# write file
	with open(output_LogR_file, "w") as f:
		f.write("\tchrs\tpos\tsample")
		[f.write("\n{0}\t{1}\t{2}\t{3}".format(i+1, input_lines[i+1].split("\t")[0],
			input_lines[i+1].split("\t")[1], logR_star[i])) for i in range(len(input_lines) - 1)]



def SNP_count_to_LogR(input_SNPs_file, avg_coverage, output_LogR_file, test=False):
	# test if file exist, raise exception on non testcase
	if not test:
		raise_if_file_exists(output_LogR_file)

	# open input file and output file. If test case clear output file
	with open(input_SNPs_file, 'r') as input_file:
		with open(output_LogR_file, 'w') as output_file:
			if test:
				output_file.truncate()
		
			# skip header of input
			input_file.readline()
			# write output header
			output_file.write("\tchrs\tpos\tsample")

			# iterate enries in input with start number of 1
			for line_num, line in enumerate(input_file, start = 1):
				# tab is seperator for entries. split line on seperator and convert all 
				# entries to int
				split_line = list(map(int, line.rstrip("\n").split("\t")))
				# compute LogR 
				sample = submarine.compute_LogR(split_line[6], avg_coverage)

				#write results in file/ SNP-num; chromosome; position; LogR
				output_file.write("\n{0}\t{1}\t{2}\t{3}".format(line_num, split_line[0], split_line[1], sample))

def clean_SNP_count_naive(input_SNP_file, index, output_clean_file, test=False):
	# test if file exist, raise exception on non testcase
	if not test:
		raise_if_file_exists(output_clean_file)

	# mapping from index to base in split_line
	mapping_index_to_base = {2:"A", 3:"C", 4:"G", 5:"T"}
	# mapping from base to index in split_line
	mapping_base_to_index = {"A":2, "C":3, "G":4, "T":5}
	#constan index for total in split_line
	total_index = 6

	# open input file and output file. If test case clear output file
	with open(input_SNP_file, 'r') as input_file:
		with open(output_clean_file, 'w') as output_file:
			if test:
				output_file.truncate()

			# skip header of input
			input_file.readline()
			# write output header
			output_file.write("#CHR\tPOS\tCount_A\tCount_C\tCount_G\tCount_T\tGood_depth")

			# count snps, cleaned snps and gt_5
			snps_count = 0
			cleaned_snps_count = 0
			cleaned_snps_gt_5_count = 0

			#Flags for cleaned SNP and SNP_gt5
			cleaned_snps = False
			cleaned_snps_gt_5 = False
			for line in input_file:
				split_line = list(map(int, line.rstrip("\n").split("\t")))
				# check if key is in index
				if index.has_key((split_line[0], split_line[1])):
					#get variance and reference from index
					(reference, variance) = index[(split_line[0], split_line[1])]
				else:
					#error message and skip this entry, it is removed in the new file
					print("ERROR: SNP(chr: {0}, pos: {1}) has no entry in indexfile and is removed.".format(split_line[0], split_line[1]))
					continue
				#set Flags to default
				cleaned_snp = False
				cleaned_snp_gt_5 = False
				# check every count entry
				for i in range(2,total_index):
					#if entry is != 0 and not reference or variance, set to 0 and set flags
					if mapping_index_to_base[i] != reference and mapping_index_to_base[i] != variance and split_line[i] != 0:
						if split_line[i] > 5:
							#set Flag
							cleaned_snp_gt_5 = True
						# set Flag
						cleaned_snp = True
						# set entry to 0
						split_line[i] = 0

				# correct total
				split_line[total_index] = split_line[mapping_base_to_index[reference]] + split_line[mapping_base_to_index[variance]]

				# write snp in file
				output_file.write("\n{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}".format(split_line[0], split_line[1], split_line[2], split_line[3], split_line[4], split_line[5], split_line[6]))

				# increment counter
				snps_count += 1
				if cleaned_snp:
					cleaned_snps_count += 1
				if cleaned_snp_gt_5:
					cleaned_snps_gt_5_count += 1

			#print number of processed snps, cleaned snps and cleaned snps greater than 5
			print("#SNPs: {0}".format(snps_count))
			print("#cleaned SNPs: {0}".format(cleaned_snps_count))
			print("#cleaned SNPs, where cleaned entry was greater 5: {0}".format(cleaned_snps_gt_5_count))

# removes all positions in a SNP count file (generated by Amit) that have a cover-
# age less than a given cutoff, and writes to a new SNP count file
# returns a list (chromsom, position) of the deleted lines
def remove_positions_in_normal_with_low_coverage(normal_file_input, 
	new_normal_file_output, coverage_cutoff, test=False):
	first_line = True
	removed_positions = []

	if not test:
		raise_if_file_exists(new_normal_file_output)

	with open(normal_file_input) as input_file:
		with open(new_normal_file_output, 'w') as output_file:
			for line in input_file:
				if first_line:
					first_line = False
					output_file.write(line)
				else:
					(chro, pos, ca, cc, cg, ct, cov) = (
						list(map(int, line.rstrip().split('\t'))))
					# if coverage of line is too low
					# the current chromosome and position
					# are written to the list
					if cov <= coverage_cutoff:
						removed_positions.append(
							(chro, pos))
					# if the coverage of the line is high 
					# enough, the line is written to the
					# new file
					else:
						output_file.write(line)

	return removed_positions

# removes lines in a SNP count file according to a list (chromsome, position)
# and writes to a new file
def remove_positions_from_list(file_input, file_output, removed_positions,
	test=False):
	first_line = True 
	current_pos = 0

	if not test:	
		raise_if_file_exists(file_output)

	with open(file_input) as input_file:
		with open(file_output, 'w') as output_file:
			for line in input_file:
				if first_line:
					first_line = False
					output_file.write(line)
				else:
					(chro, pos, ca, cc, cg, ct, cov) = (
						list(map(int, line.rstrip().split('\t'))))
					# if current line corresponds to a 
					# position in the list with positions
					# to be removed
					if (current_pos < len(removed_positions) 
						and removed_positions[current_pos] == (chro, pos)):
						current_pos += 1
					# if line doesn't correspond to position
					# that should be removed write to new
					# file
					else:
						output_file.write(line)

# removes positions from SNP count file that have a coverage less than
# the coverage cutoff and that appear not in a row of at least num_in_row
def remove_single_positions_with_low_coverage(input_file_name, 
	output_file_name, coverage_cutoff, num_in_row, test=False):

	first_line = True
	smaller_lines = ""
	tmp_positions = []
	removed_positions = []
	smaller_coverage = False

	if not test:
		raise_if_file_exists(output_file_name)

	with open(input_file_name) as input_file:
		with open(output_file_name, 'w') as output_file:
			for line in input_file:
				if first_line:
					output_file.write(line)
					first_line = False
				else:
					(chro, pos, ca, cc, cg, ct, cov) = (list(map(int,line.rstrip().split('\t'))))
					# if coverage is less than the cutoff save position of SNP and the line
					if cov <= coverage_cutoff:
						smaller_coverage = True
						tmp_positions.append((chro, pos))
						smaller_lines += line
					# if coverge is higher
					else:
						# if coverage was smaller before, check, whether the minimum
						# length of the region is reached, if so print line 
						# if not: save removed positions
						if smaller_coverage:
							if len(tmp_positions) >= num_in_row:
								output_file.write(smaller_lines)
							else:
								removed_positions.extend(tmp_positions)

							smaller_lines = ""
							tmp_positions = []
							smaller_coverage = False

						# write line with high enough coverage
						output_file.write(line)

			# after all lines from input file are used, check other last line was one
			# with a coverage under the threshold
			if smaller_coverage:
				if len(tmp_positions) >= num_in_row:
					output_file.write(smaller_lines)
				else:
					removed_positions.extend(tmp_positions) 

	return removed_positions

# gets the Z-matrix and writes it to a file
def write_matrix_to_file(z_matrix, file_name, test=False):
	if not test:
		raise_if_file_exists(file_name)
	with open(file_name, "w") as f:
		matrix_string = json.dumps(z_matrix)
		f.write(matrix_string)

# reads the Z-matrix from a file and returns it
def read_matrix_from_file(file_name):
	with open(file_name, "r") as f:
		z_matrix = json.load(f)

	return z_matrix

def get_ID_mapping_from_log_file(file_name):
	mapping = {}

	with open(file_name, "r") as f:
		for line in f:
			if "Subclone index to ID mapping" in line:
				mapping_parts = line.rstrip().split(": ")[-1].split(", ")
				for i in mapping_parts:
					subclone_index, my_id = i.split("->")
					try:
						my_id = int(my_id)
					except ValueError:
						pass
					mapping[my_id] = int(subclone_index)

	return mapping
				
def print_status_value(status, value):
	logging.debug("status: {0}".format(status))
	logging.debug("value: {0}".format(value))

def print_llh(llh):
	logging.info("LLH: {0}".format(llh))

def print_mdl(mdl):
	logging.info("MDL: {0}".format(mdl))

def visualize_result_file(result_file, output_file=None, test=False):
	pass

def possible_parents_to_string(ppm, output_file, test=False):
	if test is False:
		raise_if_file_exists(output_file)

	output = []

	for i in range(1,len(ppm)):
		possible_parents = np.where(ppm[i] == 1)[0]
		possible_parents = ["{0}".format(x) for x in possible_parents]
		output.append("{0}:{1}".format(i, ",".join(possible_parents)))

	output = "{0}".format(";".join(output))

	if test == True:
		return output

	with open(output_file, "w") as f:
		f.write(output)

# show the current tree, which children are definite up until last, excluding k
def build_current_tree_definite_children(ppm, last, k):
	tree_string = []
	for i in range(1,last+1):
		if i == k:
			continue
		possible_parents = np.where(ppm[i] == 1)[0]
		if len(possible_parents) == 1:
			parent = possible_parents[0]
			tree_string.append("{0}->{1}".format(parent, i))
	return ",".join(tree_string)



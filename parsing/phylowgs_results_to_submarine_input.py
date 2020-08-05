import sys
from os.path import dirname, abspath, join, realpath, isfile, sep, pardir
sys.path.append(dirname(realpath(__file__)) + sep + pardir)
import json
import numpy as np
from scipy.special import gammaln
from numpy import log
import argparse

# given the summary file of PhyloWGS, find the tree with the best LLH and return it
def get_tree_w_best_llh(summary_file):
	# read json file
	with open(summary_file, "r") as f:
		data = json.load(f)

	llh = -float("inf")
	best_tree_id = -1
	mulitple_best = False
	# iterate through all trees
	for i in data["trees"].keys():
		current_llh = data["trees"][i]["llh"]
		# current LLH is better or equal the previous LLH
		if current_llh >= llh:
			# store whether there are multiple trees with the same LLH
			if current_llh == llh:
				mulitple_best = True
			else:
				mulitple_best = False
			# save current best LLH and its tree
			llh = current_llh
			best_tree_id = i

	# there should only be one best tree
	assert mulitple_best == False

	# return best tree
	return best_tree_id, data["trees"][best_tree_id]

# gets the subclones, frequencies and relationships of a PhyloWGS tree
def get_subclones_freq_relationships(tree):
	# gets the subclonal IDs, also contains information about germline
	subclones = tree["populations"].keys()
	
	# get frequencies
	freqs = {}
	for subclone in subclones:
		freqs[subclone] = tree["populations"][subclone]["cellular_prevalence"]

	# get relationships in Z matrix
	z_matrix = get_relationship_from_structure(len(subclones), tree["structure"])

	return subclones, freqs, z_matrix
	
# given the structure dict from a PhyloWGS tree, computes the relationships between
# subclones in form of a Z matrix
def get_relationship_from_structure(lin_num, tree_structure):
	# create empty Z matrix
	z_matrix = [[-1] * lin_num for x in range(lin_num)]

	# set present relationships accroding to structure
	for anc in tree_structure.keys():
		descendants = tree_structure[anc]
		for des in descendants:
			assert int(anc) < des
			z_matrix[int(anc)][des] = 1

	# set all present relationships
	for kp in range(lin_num-1, 1, -1):
		z_matrix[0][kp] = 1
		for k in range(kp-1, 0, -1):
			if z_matrix[k][kp] == 1:
				for kstar in range(k-1, -1, -1):
					if z_matrix[kstar][k] == 1:
						z_matrix[kstar][kp] = 1

	return z_matrix
	
# write a Submarine frequency file
def write_freq_file(subclones, freqs, filename):
	with open(filename, "w") as f:
		# write header
		f.write("subclone_ID\tfrequencies\n")
		# write information for each subclone
		for subclone in subclones:
			# skip germline
			if subclone == "0":
				continue
			f.write("{0}\t{1}\n".format(subclone, "\t".join(map(str, freqs[subclone]))))

def get_all_SSMs(ssm_file):
	ssms = {}
	first_line = True
	# read SSM file
	with open(ssm_file, "r") as f:
		# skip header
		for line in f:
			if first_line == True:
				first_line = False
				continue

			# read SSM
			my_id = line.split("\t")[0]
			index = int(my_id.split("s")[-1])
			chromosome, position = line.split("\t")[1].split("_")
			a = list(map(int, (line.split("\t")[2].split(","))))
			d = list(map(int, (line.split("\t")[3].split(","))))
			position = int(position)

			# store SSM
			assert my_id not in ssms
			ssms[my_id] = {"chromosome": chromosome, "position": position,
				"a": a, "d": d, "index": index}

	return ssms

# reads the CNAs from file
def get_all_CNAs(cna_file, male):
	cnas = {}
	first_line = True
	seg_index = 0
	cna_index = 0
	# read CNA file
	with open(cna_file, "r") as f:
		# skip header
		for line in f:
			if first_line == True:
				first_line = False
				continue

			# read CNA
			my_id, tmp, tmp2, overlapping_ssms, physical_cnvs = line.rstrip().split("\t")
			overlapping_ssms = [x.split(",")[0] for x in overlapping_ssms.split(";")]
			physical_cnvs = physical_cnvs.split(";")
			# parse the physical CNAs
			physical_cnvs_list = []
			for pc in physical_cnvs:
				chrom, start, end, major_cn, minor_cn, cell_prev = pc.split(",")
				chrom = chrom.split("=")[-1]
				start = int(start.split("=")[-1])
				end = int(end.split("=")[-1])
				major_cn = int(major_cn.split("=")[-1])
				minor_cn = int(minor_cn.split("=")[-1])
				# one allele needs to have a CN change
				assert (major_cn != 1) or (minor_cn != 1)
				# process major and minor CN separately
				# major CN
				if major_cn != 1:
					# change cannot be smaller than -1
					change = major_cn - 1
					assert change >= -1
					# add CNA
					physical_cnvs_list.append({"chromosome": chrom,
						"start": start, "end": end, "CNA_index": cna_index, 
						"seg_index": seg_index, "phase": "A", "change": change})
					cna_index += 1
				# minor CN
				if minor_cn != 1:
					# skip on male sex chromosome
					if (chrom != "X" and chrom != "Y") or male == False:
						# change cannot be smaller than -1
						change = minor_cn - 1
						assert change >= -1
						# add CNA
						physical_cnvs_list.append({"chromosome": chrom,
							"start": start, "end": end, "CNA_index": cna_index, 
							"seg_index": seg_index, "phase": "B", "change": change})
						cna_index += 1
				seg_index += 1
			# add CNA of whole line
			assert my_id not in cnas
			cnas[my_id] = {"physical_cnas": physical_cnvs_list, "overlapping_ssms": overlapping_ssms}

		# add additional segment for SSMs on CN-normal sements
		seg_index += 1
		# get number of different CNAs
		cna_num = cna_index + 1

		return cnas, seg_index, cna_num

# assigns the subclones to the SSMs and CNAs
def get_mutation_assignment(json_tree_file, ssms, cnas):
	with open(json_tree_file, "r") as f:
		assignments = json.load(f)

	# iterate through all subclones
	for subclone in assignments["mut_assignments"].keys():
		# go through all SSMs
		for ssm in assignments["mut_assignments"][subclone]["ssms"]:
			assert "subclone" not in ssms[ssm]
			ssms[ssm]["subclone"] = subclone
		# go through all CNAs
		for cna in assignments["mut_assignments"][subclone]["cnvs"]:
			assert "subclone" not in cnas[cna]
			cnas[cna]["subclone"] = subclone

def assign_seg_index_to_ssms_get_impact_matrix(ssms, cnas, z_matrix, normal_seg_index, cna_num, 
	freqs, male, sequencing_error):
	# create empty impact matrix
	impact_matrix = np.zeros(len(ssms)*cna_num).reshape(len(ssms), cna_num)

	# iterate through CNAs
	for cna_id in cnas:
		cna = cnas[cna_id]
		for ssm_id in cna["overlapping_ssms"]:
			if ssm_id == "":
				continue
			
			ssm = ssms[ssm_id]
			# check whether SSM is contained in physical CNA
			ssm_contained = False
			for i in range(len(cna["physical_cnas"])):
				if (cna["physical_cnas"][i]["start"] <= ssm["position"] and
					cna["physical_cnas"][i]["end"] >= ssm["position"]):
					ssm_contained = True
					break
			assert ssm_contained

			# set seg_index of SSM
			assert "seg_index" not in ssm
			ssm["seg_index"] = cna["physical_cnas"][i]["seg_index"]

			# if current phase is A, check whether the next phyisical CNA appears on the same seg_index
			next_CNA_same_seg_index = False
			if cna["physical_cnas"][i]["phase"] == "A" and len(cna["physical_cnas"]) > i+1:
				if cna["physical_cnas"][i]["seg_index"] == cna["physical_cnas"][i+1]["seg_index"]:
					next_CNA_same_seg_index = True

			set_impact_entry_if_necessary(ssm, cna, i, next_CNA_same_seg_index, z_matrix, freqs, impact_matrix, male, sequencing_error)


	# check which SSMs appear on segments without CN changes
	# iterate through SSMs
	for ssm in ssms:
		if "seg_index" not in ssms[ssm]:
			ssms[ssm]["seg_index"] = normal_seg_index

	return impact_matrix

# checks whether the CN of an SSM should be influenced by a CNA
def set_impact_entry_if_necessary(ssm, cna, physical_cna_index, next_CNA_same_seg_index, z_matrix, freqs, impact_matrix, male, sequencing_error):
	
	# influence is only possible if SSM is in same subclone as CNA or an ancestral one
	if ssm["subclone"] != cna["subclone"] and z_matrix[int(ssm["subclone"])][int(cna["subclone"])] != 1:
		return

	# get major and minor change
	change_major = 0
	change_minor = 0
	# change of CNA
	change_cna = 0
	# change is on A
	if cna["physical_cnas"][physical_cna_index]["phase"] == "A":
		change_major = cna["physical_cnas"][physical_cna_index]["change"]
		change_cna = change_major
		# check whether change is also on B
		if next_CNA_same_seg_index:
			assert cna["physical_cnas"][physical_cna_index]["start"] == cna["physical_cnas"][physical_cna_index+1]["start"]
			assert cna["physical_cnas"][physical_cna_index+1]["phase"] == "B"
			change_minor = cna["physical_cnas"][physical_cna_index+1]["change"]
	# change is on B
	elif cna["physical_cnas"][physical_cna_index]["phase"] == "B":
		change_minor = cna["physical_cnas"][physical_cna_index]["change"]
		change_cna = change_minor
	# change must be on A or B
	else:
		raise
	# one allele must contain a CNC
	assert (change_major != 0) or (change_minor != 0)

	# compute LLH of SSM being influenced or not by CNC
	sample_size = len(freqs["0"])
	k = ssm["subclone"]
	kp = cna["subclone"]
	# LLH if SSM is influenced
	llh_ssm_influenced_by_cna = get_llh_for_ssm(ssm, freqs, sample_size, k, kp, True, 
		change_cna, male, change_major, change_minor, sequencing_error)
	# LLH if SSM is not influenced
	# SSM is always influenced on male sex chromosomes
	llh_ssm_not_influenced_by_cna = -float("inf")
	if (cna["physical_cnas"][physical_cna_index]["chromosome"] != "X" and 
		cna["physical_cnas"][physical_cna_index]["chromosome"] != "Y") or male == False:
		llh_ssm_not_influenced_by_cna = get_llh_for_ssm(ssm, freqs, sample_size, k, kp, False, 
			change_cna, male, change_major, change_minor, sequencing_error)
	# if another CNC is on other allele, also compute llh for this influence
	llh_ssm_influenced_by_other_cna = -float("inf")
	llh_ssm_not_influenced_by_other_cna = -float("inf")
	if next_CNA_same_seg_index:
		# LLH if SSM is influenced by other CNA
		llh_ssm_influenced_by_other_cna = get_llh_for_ssm(ssm, freqs, sample_size, k, kp, True, 
			change_minor, male, change_major, change_minor, sequencing_error)
		# LLH if SSM is not influenced by other CNA
		llh_ssm_not_influenced_by_other_cna = get_llh_for_ssm(ssm, freqs, sample_size, k, kp, False, 
			change_minor, male, change_major, change_minor, sequencing_error)

	# see whether SSM is influenced by CNA and if there are two, by which one
	# only one CNA in segment
	if next_CNA_same_seg_index == False:
		# SSM has higher likelihood with influence
		if llh_ssm_influenced_by_cna > llh_ssm_not_influenced_by_cna:
			impact_matrix[ssm["index"]][cna["physical_cnas"][physical_cna_index]["CNA_index"]] = 1
	# two CNAs in one segment
	else:
		# let SSM be influenced from the one which gives higher likelihood
		if llh_ssm_influenced_by_cna > llh_ssm_influenced_by_other_cna:
			impact_matrix[ssm["index"]][cna["physical_cnas"][physical_cna_index]["CNA_index"]] = 1
		else:
			impact_matrix[ssm["index"]][cna["physical_cnas"][physical_cna_index+1]["CNA_index"]] = 1

# computes the LLH for a SSM, SSM can be influenced by CN change or not
# influence: whether SSM is influenced or not
# change_cna: CN change by which the SSM can be influenced
# change_major: change on A
# change_minor: change on B
def get_llh_for_ssm(ssm, freqs, sample_size, k, kp, influence, change_cna, male, change_major, change_minor, sequencing_error):

	llh_ssm = 0
	
	# interates through all samples
	for s in range(sample_size):
		# computes CN of variant
		cn_variant = compute_cn_variant(freqs, s, k, kp, influence, change_cna)
		# computes CN of reference
		cn_reference = compute_cn_reference(ssm["chromosome"], male, freqs, s, kp,
			change_major, change_minor, cn_variant)
		# computes totoal CN
		cn_total = cn_variant + cn_reference
		# expected proportion of reads containing the reference allele
		p = float((cn_reference * (1 - sequencing_error)) + (cn_variant * sequencing_error)) / cn_total
		llh_ssm = llh_ssm + compute_log_llh_binomial(ssm["d"][s], ssm["a"][s], p)

	return llh_ssm

# computes the log likelohood of a binomial distribution
# n: tries
# k: success
# p: rate
def compute_log_llh_binomial(n, k, p):
	log_n_choose_k = gammaln(n+1) - gammaln(k+1) - gammaln(n -k + 1)
	return log_n_choose_k + (k * log(p)) + ((n - k) * log(1 - p))

# computes the copy number of the variant allele
# freqs: subclonal frequencies
# sample: sample index
# k, kp: subclones k and kp
# influence: whether CNC should influence SSM
# change: change of CN
def compute_cn_variant(freqs, sample, k, kp, influence, change):
	cn_variant = freqs[str(k)][sample] + (influence * change * freqs[str(kp)][sample])
	assert cn_variant >= 0
	return cn_variant

# computes the copy number of the reference allele
def compute_cn_reference(chromosome, male, freqs, sample, kp, change_major, change_minor, cn_variant):
	# compute normal CN
	if chromosome != "X" and chromosome != "Y":
		normal_cn = 2
	elif male == True:
		normal_cn = 1
	else:
		assert chromosome != "Y"
		normal_cn = 2

	# computes copy number of refence allele
	cn_reference =  normal_cn + (change_major * freqs[str(kp)][sample]) + (change_minor * freqs[str(kp)][sample]) - cn_variant
	assert cn_reference >= 0
	return cn_reference

# writes CNAs and their information to file
def write_cna_file(cnas, filename):

	prev_index = -1

	with open(filename, "w") as f:
		f.write("CNA_index\tseg_index\tsubclone_ID\tphase\tchange\n")
		for cna_key in cnas.keys():
			for physical_cna in cnas[cna_key]["physical_cnas"]:
				f.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(physical_cna["CNA_index"],
					physical_cna["seg_index"], cnas[cna_key]["subclone"],
					physical_cna["phase"], physical_cna["change"]))
				assert prev_index + 1 == physical_cna["CNA_index"]
				prev_index = physical_cna["CNA_index"]

# writes SSMs and their information to file
def write_ssm_file(ssms, filename):

	with open(filename, "w") as f:
		f.write("SSM_index\tseg_index\tsubclone_ID\n")
		for ssm_key in ssms.keys():
			ssm = ssms[ssm_key]
			f.write("{0}\t{1}\t{2}\n".format(ssm["index"], ssm["seg_index"],
				ssm["subclone"]))
			
# writes the impact matrix to file
def write_impact_matrix(impact_matrix, filename):
	
	with open(filename, "w") as f:
		f.write("SSM_index\tCNA_index\n")
		influences = np.where(impact_matrix == 1)
		for i in range(len(influences[0])):
			f.write("{0}\t{1}\n".format(influences[0][i], influences[1][i]))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--summary_json_input_file")
	parser.add_argument("--SSM_input_file")
	parser.add_argument("--CNA_input_file")
	parser.add_argument("--tree_input_path")
	parser.add_argument("--output_files_prefix")
	parser.add_argument("--sequencing_error", type = float, default = 0.001)
	parser.add_argument("--XY", action='store_true')
	parser.add_argument("--XX", action='store_true')
	args = parser.parse_args()

	# check sex
	if args.XY:
		if args.XX == False:
			male = True
		else:
			raise("Specify only XY or XX, not both.")
	elif args.XX:
		male = False
	else:
		raise("At the current stage, sex must be given.")

	# get ID of tree with highest likelihood, get more information about tree
	best_tree_id, best_tree = get_tree_w_best_llh(args.summary_json_input_file)
	# get subclones, frequencies and relationships
	subclones, freqs, z_matrix = get_subclones_freq_relationships(best_tree)
	# read SSMs
	ssms = get_all_SSMs(args.SSM_input_file)
	# read CNAs
	cnas, normal_seg_index, cna_num = get_all_CNAs(args.CNA_input_file, male)
	# get mutation assignment
	get_mutation_assignment("{0}/{1}.json".format(args.tree_input_path, best_tree_id), ssms, cnas)
	# build impact matrix and assign SSMs to segment indices
	impact_matrix = assign_seg_index_to_ssms_get_impact_matrix(ssms, cnas, z_matrix, 
		normal_seg_index, cna_num, freqs, male, args.sequencing_error)
	
	# write files for SubMARine
	write_freq_file(subclones, freqs, "{0}_freq.csv".format(args.output_files_prefix))
	write_cna_file(cnas, "{0}_cnas.csv".format(args.output_files_prefix))
	write_ssm_file(ssms, "{0}_ssms.csv".format(args.output_files_prefix))
	write_impact_matrix(impact_matrix, "{0}_impact.csv".format(args.output_files_prefix))

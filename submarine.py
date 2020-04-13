import sys
import argparse
import exceptions_onctopus as eo
import math
import lineage
import io_file as oio
import numpy as np
import copy
import constants as cons
import operator
import cnv
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import snp_ssm
import segment
import copy
from itertools import compress
import json

def new_dfs(z_matrix, my_lineages, seg_num=None, filename=None, count_threshold=-1, ppm=None, test_iteration=False,
	test_reconstructions=False, analyze_ambiguity_during_runtime=False):
	
	z_matrix = np.asarray(z_matrix)

	total_count = 0
	valid_count = 0

	# create list structure "undef_rels" to iterate through all possible settings of matrix
	undef_rels = []
	REL = 0
	K = 1
	KP = 2
	SBCLR = 3
	USED_P1 = 4
	USED_M1 = 5
	for kp in range(len(z_matrix)):
		for k in range(0, kp):
			if z_matrix[k][kp] == 0:
				undef_rels.append([0, k, kp, None, False, False])
	number_undef_rels = len(undef_rels)

	# different variables needed for this function
	lin_num = len(my_lineages)
	zero_count = lin_num * lin_num
	zero_count, triplet_xys, triplet_ysx, triplet_xsy = check_and_update_complete_Z_matrix_from_matrix(z_matrix, zero_count, lin_num)
	matrix_after_first_round = np.copy(z_matrix)
	# go once through segments and get gains, losses and SSMs
	gain_num = []
	loss_num = []
	CNVs = []
	present_ssms = []
	ssm_infl_cnv_same_lineage = []
	# iterate through all segments once to get all CN changes and SSMs appearances
	get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lin_num, my_lineages,
		ssm_infl_cnv_same_lineage)
	present_ssms = np.asarray(present_ssms)
	# combine information to Z-matrix and Co object
	zmco = Z_Matrix_Co(z_matrix, triplet_xys, triplet_ysx, triplet_xsy, present_ssms, matrix_after_first_round)
	# create variables needed for sum rule
	last = None
	defparent = None
	avFreqs = None
	linFreqs = None
	if ppm is not None:
		last = lin_num - 1
		# get definite parents and available frequencies
		linFreqs = np.asarray([my_lineages[i].freq for i in range(len(my_lineages))])
		defparent, avFreqs = get_definite_parents_available_frequencies(linFreqs, ppm)
	sbclr_0 = Subclonal_Reconstruction_for_DFS(zmco, present_ssms, ppm, defparent, avFreqs)

	# checks whether given partial subclonal reconstruction is valid
	if not is_reconstruction_valid(my_lineages, z_matrix, ppm, seg_num, gain_num, loss_num, CNVs, present_ssms, ssm_infl_cnv_same_lineage,
		check_only_validity_possible=True):
		raise eo.ReconstructionInvalid("Reconstruction is invalid")

	# testing whether function iterates correctly over all possible settings in Z-matrix
	if test_iteration:
		all_options = []
	# testing whether function allows and forbids correct subclonal reconstructions
	if test_reconstructions:
		reconstructions = []

	# pointer i iterates through list with undefined relationships
	i = 0
	while i < number_undef_rels + 1 and number_undef_rels != 0:
		# current relationship is undefined
		if i < number_undef_rels and undef_rels[i][REL] == 0:
			# set relationship to present
			undef_rels[i][REL] = 1
			# next steps not needed for testing whether iteration finds all settings
			if test_iteration:
				i += 1
				continue
			# create first subclonal reconstruction to update
			if i == 0:
				sbclr = copy.deepcopy(sbclr_0)
			# use previous subclonal reconstruction to update
			else:
				sbclr = copy.deepcopy(undef_rels[i-1][SBCLR])
			try:
				# update relationship in subclonal reconstruction to present
				update_ancestry(1, undef_rels[i][K], undef_rels[i][KP], last=last, ppm=sbclr.ppm, defparent=sbclr.defparent, 
					linFreqs=linFreqs, avFreqs=sbclr.avFreqs, zmco=sbclr.zmco, seg_num=seg_num, 
					zero_count=zero_count, gain_num=gain_num, loss_num=loss_num, CNVs=CNVs, present_ssms=sbclr.present_ssms)
				undef_rels[i][SBCLR] = sbclr
			except (eo.ZInconsistence, eo.ADRelationNotPossible, eo.ZUpdateNotPossible, eo.NoParentsLeft, eo.NoParentsLeftNoise,
				eo.RelationshipAlreadySet) as e:
				# update not possible
				# thus, count one tree that was enumerated
				total_count += 1
				if total_count % 100 == 0:
					logging.info("Total count: {0}".format(total_count))
				# update relationship in subclonal reconstruction to other value
				undef_rels[i][REL] = -1
				# create subclonal reconstruction to update
				if i == 0:
					sbclr = copy.deepcopy(sbclr_0)
				else:
					sbclr = copy.deepcopy(undef_rels[i-1][SBCLR])
				# update relationship in subclonal reconstruction to absent
				total_count, i = update_sbclr_dfs(-1, undef_rels[i][K], undef_rels[i][KP], last, sbclr, seg_num, zero_count, 
					gain_num, loss_num, CNVs, total_count, undef_rels, i, REL, SBCLR, linFreqs)
				undef_rels[i][SBCLR] = sbclr

		# current relationship is present
		elif i < number_undef_rels and undef_rels[i][REL] == 1:
			# set relationship to absent
			undef_rels[i][REL] = -1
			# next steps not needed for testing whether iteration finds all settings
			if test_iteration:
				i += 1
				continue
			# create subclonal reconstruction to update
			if i == 0:
				sbclr = copy.deepcopy(sbclr_0)
			else:
				sbclr = copy.deepcopy(undef_rels[i-1][SBCLR])
			total_count, i = update_sbclr_dfs(-1, undef_rels[i][K], undef_rels[i][KP], last, sbclr, seg_num, zero_count, 
				gain_num, loss_num, CNVs, total_count, undef_rels, i, REL, SBCLR, linFreqs)
			undef_rels[i][SBCLR] = sbclr

		# current relationship is absent
		elif i < number_undef_rels and undef_rels[i][REL] == -1:
			# look for last relationship that was set to present
			i -= 1
			while i >= 0 and undef_rels[i][REL] != 1:
				i -= 1
			# no relationship is present anymore, this means all possible settings where tried
			if i < 0:
				break
			# last present relationship is found
			if undef_rels[i][REL] == 1:
				# set following relationships to ambiguous again
				for j in range(i+1, number_undef_rels):
					undef_rels[j][REL] = 0
					undef_rels[j][SBCLR] = None
			# decrease i because it's going to be increased after condistions
			i -= 1

		# all relationships were set
		elif i == number_undef_rels:
			i -= 1

			# when function should be tested, append current setting to list
			if test_iteration:
				all_options.append([x[REL] for x in undef_rels])
				continue
			# current subclonal reconstruction is complete
			total_count += 1
			if total_count % 100 == 0:
				logging.info("Total count: {0}".format(total_count))

			# check sum rule
			if check_sum_rule(my_lineages, undef_rels[i][SBCLR].zmco.z_matrix):
				valid_count += 1
				if valid_count % 100 == 0:
					logging.info("Valid count: {0}".format(valid_count))
					if analyze_ambiguity_during_runtime:
						used_p1 = [x[USED_P1] for x in undef_rels]
						used_m1 = [x[USED_M1] for x in undef_rels]
						logging.info("Ambiguity: {0}".format(np.logical_and(used_p1, used_m1).all()))

				# if Z-matrix should be written to file and maximum number in file is not reached yet
				if filename is not None and valid_count <= count_threshold:
					with open(filename, "a") as f:
						my_string = json.dumps(convert_zmatrix_0_m1(undef_rels[i][SBCLR].zmco.z_matrix))
						f.write("{0}\n".format(my_string))
				# if used relationships should be anaylzed during runtime
				if analyze_ambiguity_during_runtime:
					for j in range(number_undef_rels):
						if undef_rels[j][REL] == 1:
							undef_rels[j][USED_P1] = True
						elif undef_rels[j][REL] == -1:
							undef_rels[j][USED_M1] = True
						else:
							raise eo.MyException("should never happen")
				# testing whether function allows and forbids correct subclonal reconstructions
				if test_reconstructions:
					reconstructions.append(undef_rels[i][SBCLR])
			# decrease i because it's going to be increased after condistions
			i -= 1

		i +=  1

	# if ambiguity should be analyzed during runtime
	output = ""
	if analyze_ambiguity_during_runtime:
		for j in range(number_undef_rels):
			if undef_rels[j][USED_P1] == False or undef_rels[j][USED_M1] == False:
				output += "False, {0}, {1}, {2}, {3}\n".format(undef_rels[j][K], undef_rels[j][KP], 
					undef_rels[j][USED_P1], undef_rels[j][USED_M1])	
		if output == "" and number_undef_rels > 0:
			output = "True\n"
		elif output == "":
			output = "All ancestral relationships are defined.\n"

	if number_undef_rels == 0:
		valid_count = 1

	# testing iteration
	if test_iteration:
		return all_options
	# testing whether function allows and forbids correct subclonal reconstructions
	if test_reconstructions:
		return reconstructions, total_count, valid_count
	# if ambiguity should be analyzed during runtime
	if analyze_ambiguity_during_runtime:
		return total_count, valid_count, output

	
	return total_count, valid_count

def update_sbclr_dfs(value, k, kp, last, sbclr, seg_num, zero_count, gain_num, loss_num, CNVs, total_count, undef_rels, i, REL, SBCLR, linFreqs):
	undef_rels[i][REL] = value
	# update relationship in subclonal reconstruction
	try:
		update_ancestry(value, k, kp, last=last, ppm=sbclr.ppm, defparent=sbclr.defparent, 
			linFreqs=linFreqs, avFreqs=sbclr.avFreqs, zmco=sbclr.zmco, seg_num=seg_num, 
			zero_count=zero_count, gain_num=gain_num, loss_num=loss_num, CNVs=CNVs, present_ssms=sbclr.present_ssms)
		undef_rels[i][SBCLR] = sbclr
	except (eo.ZInconsistence, eo.ADRelationNotPossible, eo.ZUpdateNotPossible, eo.NoParentsLeft, eo.NoParentsLeftNoise,
		eo.RelationshipAlreadySet) as e:
		# update not possible
		# thus, count one tree that was enumerated
		total_count += 1
		if total_count % 100 == 0:
			logging.info("Total count: {0}".format(total_count))
		# decrease i because it's going to be increased after condistions in main loop
		i -= 1

	return total_count, i

# given a Z-matrix, counts how often lineage relationships are ambiguous
def count_ambiguous_relationships(z_matrix):
	z_matrix = np.asarray(z_matrix)

	ambi_num = 0

	# forst row skipt because it does not contain ambiguous relationships by definition
	for k in range(1,len(z_matrix)):
		ambi_num += len(np.where(z_matrix[k] == 0)[0])

	return ambi_num

def update_ancestry_w_preprocessing(my_lineages, z_matrix, ppm, seg_num, value, k, kprime):
	# get present mutations from lineages
	lineage_num = len(my_lineages)
	# go once through segment and get gains, losses and SSMs
	gain_num = []
	loss_num = []
	CNVs = []
	present_ssms = []
	ssm_infl_cnv_same_lineage = []
	# iterate through all segments once to get all CN changes and SSMs appearances
	get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
		ssm_infl_cnv_same_lineage)

	# get definite parents and available frequencies
	frequencies = np.asarray([my_lineages[i].freq for i in range(len(my_lineages))])
	defp, avFreqs = get_definite_parents_available_frequencies(frequencies, ppm)

	# copy
	origin_z_matrix = copy.deepcopy(z_matrix)
	origin_ppm = copy.deepcopy(ppm)

	last = lineage_num - 1

	# create zmco
	dummy_zero_count = lineage_num * lineage_num
	zero_count, triplet_xys, triplet_ysx, triplet_xsy = check_and_update_complete_Z_matrix_from_matrix(z_matrix, dummy_zero_count, lineage_num,
		CNVs=CNVs, present_ssms=present_ssms, z_matrix_after_CN_influence_check=origin_z_matrix)
	assert (np.asarray(origin_z_matrix) == np.asarray(z_matrix)).all()
	triplets_list = [[triplet_xys, triplet_ysx, triplet_xsy]]
	zmcos = create_Z_Matrix_Co_objects([z_matrix], origin_z_matrix, [present_ssms], triplets_list)
	zmco = zmcos[0]

	update_ancestry(value, k, kprime, last=last, ppm=ppm, defparent=defp, linFreqs=frequencies, avFreqs=avFreqs, zmco=zmco, seg_num=seg_num, 
		zero_count=zero_count, gain_num=gain_num, loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms)

# given a subclonal reconstruction with lineages, mutations, Z-matrix and possible parent matrix, check whether it is valid
def is_reconstruction_valid(my_lineages, z_matrix, ppm, seg_num, gain_num=None, loss_num=None, CNVs=None, present_ssms=None,
	ssm_infl_cnv_same_lineage=None, check_only_validity_possible=False):
	# get present mutations from lineages
	lineage_num = len(my_lineages)
	# go once through segment and get gains, losses and SSMs
	if gain_num is None and loss_num is None and CNVs is None and present_ssms is None and ssm_infl_cnv_same_lineage is None:
		gain_num = []
		loss_num = []
		CNVs = []
		present_ssms = []
		ssm_infl_cnv_same_lineage = []
		# iterate through all segments once to get all CN changes and SSMs appearances
		get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
			ssm_infl_cnv_same_lineage)

	# copy elements
	origin_present_ssms = copy.deepcopy(present_ssms)
	origin_z_matrix = copy.deepcopy(z_matrix)
	origin_ppm = copy.deepcopy(ppm)

	# propagate tree rules
	dummy_zero_count = lineage_num * lineage_num
	try:
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = check_and_update_complete_Z_matrix_from_matrix(z_matrix, dummy_zero_count, lineage_num,
			CNVs=CNVs, present_ssms=present_ssms, z_matrix_after_CN_influence_check=origin_z_matrix)
	except eo.MyException as e:
		return False

	# propagate crossing rule
	zero_count = check_crossing_rule_function(my_lineages, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy)

	# propagate relationship absense rule
	try:
		z_matrix_list, z_matrix_fst_rnd, triplets_list = (
		        post_analysis_Z_matrix(my_lineages, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy,
		        matrix_splitting=False, first_absence_propagation=True, CNVs=CNVs, present_ssms=present_ssms, gain_num=gain_num,
			loss_num=loss_num))
	except eo.MyException as e:
		return False

	# propagate sum rule
	zmcos = create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, [present_ssms], triplets_list)
	zmco = zmcos[0]
	frequencies = np.asarray([my_lineages[i].freq for i in range(len(my_lineages))])
	try:
		sum_rule_worked, avFreqs, ppm = sum_rule_algo_outer_loop(frequencies, zmco, seg_num, zero_count,
			gain_num, loss_num, CNVs, present_ssms)
	except eo.MyException as e:
		return False

	if check_only_validity_possible:
		return True

	if not (np.asarray(z_matrix) == np.asarray(origin_z_matrix)).all():
		return False
	if origin_ppm is not None and not (ppm == origin_ppm).all():
		return False
	if not present_ssms == origin_present_ssms:
		return False

	return True

# given lineage frequencies and a possible parent matrix, get the definite parent of each lineage if available
# and compute the available frequencies of lineages
def get_definite_parents_available_frequencies(freqs, ppm):
	lin_num = len(freqs)

	# compute definite parents
	defp = np.asarray([-1] * lin_num)
	for k in range(1, lin_num):
		pp = np.where(ppm[k] == 1)[0]
		if len(pp) == 1:
			defp[k] = pp[0]

	# compute available freq
	avFreqs = np.copy(freqs)
	for k in range(lin_num):
		children = np.where(defp == k)[0]
		for child in children:
			avFreqs[k] = np.subtract(avFreqs[k], freqs[child])

	return defp, avFreqs

# given the list parental_list, where parental_list[k] = k* meaning that lineage k+1 has lineage k^* as parent
# build the sublin lists, where a list contains all descendants of a lineage
def build_sublin_lists_from_parental_info(mylins, parental_list):
	lin_num = len(mylins)

	# initialize lists of cancerous lineages with their children
	for i in range(1, len(parental_list)):
		mylins[parental_list[i]].sublins.append(i+1)

	# if there are only three lineages, lin 1 can only have lin 2 as child and lin 0 is done per definition
	# nothing to do
	if lin_num == 3:
		return
	
	# go backwards through sublins lists and add descendants of children
	for i in range(lin_num-3, 0, -1):
		children_num = len(mylins[i].sublins)
		for c in range(children_num):
			mylins[i].sublins.extend(mylins[mylins[i].sublins[c]].sublins)
		mylins[i].sublins.sort()

# given two lineages k and k', returns their lowest common ancestor
# if k is ancestor of k', k is returned
def get_lca(k, kprime, z_matrix):
	if k > kprime:
		raise eo.MyException("k needs to be smaller k'")

	if z_matrix[k][kprime] == 1:
		return k

	for kstar in range(k-1, -1, -1):
		if z_matrix[kstar][k] == 1 and z_matrix[kstar][kprime] == 1:
			return kstar

# finds the lca for a list of lineages
def get_lca_from_multiple_lineages(possible_parents, z_matrix):
	lca = get_lca(possible_parents[0], possible_parents[1], z_matrix)

	for i in range(2, len(possible_parents)):
		lca_new = get_lca(lca, possible_parents[i], z_matrix)
		lca = lca_new

	return lca
	
# checks whether all ambiguous entries in the original Z-matrix represent present and absent relationships in all 
# Z-matrices without ambiguities
def check_untightess_zmatrix(z_matrix, z_matrix_list, total_number):
	output = ""

	lin_num = len(z_matrix)

	# if Z-matrix list contains less matrices than existing, it makes no sense to process this dataset
	if len(z_matrix_list) < total_number:
		return "False, -1, -1\n"

	# data structure to keep track of tightness
	tightness_matrix = np.zeros(lin_num * lin_num * 3).reshape(lin_num, lin_num, 3)

	# note where original Z-matrix is ambiguous
	for k in range(lin_num-1):
		for k_prime in range(k, lin_num):
			if z_matrix[k][k_prime] == 0:
				tightness_matrix[k][k_prime][0] = 1

	# check all unambiguous Z-matrices
	for unam_m in z_matrix_list:
		for k in range(lin_num-1):
			for k_prime in range(k, lin_num):
				# if original matrix is ambiguous at position
				if tightness_matrix[k][k_prime][0] == 1:
					# current matrix has present relationship
					if unam_m[k][k_prime] == 1:
						tightness_matrix[k][k_prime][1] = 1
					# current matrix has absent relationship
					elif unam_m[k][k_prime] == -1:
						tightness_matrix[k][k_prime][2] = 1
					# unknown relationship
					else:
						raise Exception("Relationship has to be either present or absent.")

	# check whether both values were used for ambiguous entries
	for k in range(lin_num-1):
		for k_prime in range(k, lin_num):
			if tightness_matrix[k][k_prime][0] == 1:
				if tightness_matrix[k][k_prime][1] == 1 and tightness_matrix[k][k_prime][2] == 1:
					continue
				else:
					output += "False, {0}, {1}, {2}, {3}\n".format(k, k_prime, tightness_matrix[k][k_prime][1], 
						tightness_matrix[k][k_prime][2])
	
	# all ambiguous entries were ambiguous
	if output == "":
		return "True\n"
	else:
		return output

# returns the upper bound as logarithm  on possible reconstructions
# \sum_{k=1}^{K-1} log(# possible parents of lineage k)
def upper_bound_number_reconstructions(ppm):
	if ppm[0][0] == 1:
		raise Exception("Wrong format of possible parent matrix.")
	return np.sum([np.log(np.count_nonzero(ppm[k])) for k in range(1, len(ppm))])

def depth_first_search(ppm_file=None, z_matrix_file=None, lin_file=None, cna_file=None, ssm_file=None, output_prefix=None, overwrite=False):

	# get ppm data
	ppm = np.loadtxt(ppm_file, delimiter=",")
	# get Z-matrix
	z_matrix = oio.read_matrix_from_file(z_matrix_file)
	if z_matrix[0][0] == 0:
		z_matrix = convert_zmatrix_0_m1(z_matrix)	
	# get lineages
	my_lins = oio.read_JSON_result_file(lin_file)
	# get CNAs
	if cna_file is None:
		my_cnas = []
	else:
		my_cnas = oio.read_cnas(cna_file, use_cna_indices=True)
	# get SSMs
	if ssm_file is None:
		my_ssms = []
	else:
		my_ssms = oio.read_ssms(ssm_file, phasing=False, use_SSM_index=True)
	# get number of segments
	seg_num = get_seg_num(my_cnas, my_ssms)

	# create filenames
	valid_count_file = "{0}.valid_count.txt".format(output_prefix)
	ambi_file = "{0}.ambiguity.txt".format(output_prefix)
	if overwrite == False:
		oio.raise_if_file_exists(valid_count_file)
		oio.raise_if_file_exists(ambi_file)
	# start logging
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)
	logging.basicConfig(filename="{0}.dfs.log".format(output_prefix), filemode='w', level=10)	

	# compute correct number and all possible reconstructions
	total_count, valid_count, output = new_dfs(z_matrix, my_lins, seg_num, ppm=ppm, analyze_ambiguity_during_runtime=True)

	# last logging
	logging.info("Total number of enumerated trees: {0}".format(total_count))
	logging.info("Valid number of trees: {0}".format(valid_count))

	# write to file
	with open(valid_count_file, "w") as f:
		f.write("{0}\n".format(valid_count))
	with open(ambi_file, "w") as f:
		f.write(output)


# given a subclonal reconstruction with ambiguous lineage relationships, this function iterivly tries all possible
# values and returns the number of valid reconstructions
def compute_number_ambiguous_recs(my_lineages, seg_num, z_matrix, recursive=False, filename=None, count_threshold=-1, ppm=None, 
	check_validity=False):
	if check_validity and not is_reconstruction_valid(my_lineages, z_matrix, ppm, seg_num):
		raise eo.MyException("reconstruction is not valid")
	# different variables needed for this function
	lin_num = len(my_lineages)
	zero_count = lin_num * lin_num
	zero_count, triplet_xys, triplet_ysx, triplet_xsy = check_and_update_complete_Z_matrix_from_matrix(z_matrix, zero_count, lin_num)
	matrix_after_first_round = copy.deepcopy(z_matrix)
	# go once through segment and get gains, losses and SSMs
	gain_num = []
	loss_num = []
	CNVs = []
	present_ssms = []
	ssm_infl_cnv_same_lineage = []
	# iterate through all segments once to get all CN changes and SSMs appearances
	get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lin_num, my_lineages,
		ssm_infl_cnv_same_lineage)
	# combine information to Z-matrix and Co object
	zmco = Z_Matrix_Co(z_matrix, triplet_xys, triplet_ysx, triplet_xsy, present_ssms, matrix_after_first_round)

	if recursive == False:
		raise Exception("Not supported anymore")
		## create list with Z-matrix and Co objects
		#zmco_list = [zmco]
		## iterate through complete matrix
		#for k in range(lin_num-1):
		#	for k_prime in range(k+1, lin_num):
		#		# create new list for next round
		#		new_zmco_list = []
		#		# iterate through all Z-matrices in list
		#		for i in range(len(zmco_list)):
		#			# if current entry is ambiguous, fork matrix
		#			if zmco_list[i].z_matrix[k][k_prime] == 0:
		#				# copy current Z-matrix and co object for setting entry to 1 (relationship present)
		#				zmco_dup = copy.deepcopy(zmco_list[i])
		#				try:
		#					update_single_z_matrix_entry(1, k, k_prime, zmco_dup)
		#					new_zmco_list.append(zmco_dup)
		#				except (eo.ZInconsistence, eo.ADRelationNotPossible, eo.ZUpdateNotPossible) as e:
		#					pass
		#				# set entry of current Z-matrix and co object to -1 (relationship absent)
		#				try:
		#					update_single_z_matrix_entry(-1, k, k_prime, zmco_list[i])
		#					new_zmco_list.append(zmco_list[i])
		#				except (eo.ZInconsistence, eo.ADRelationNotPossible, eo.ZUpdateNotPossible) as e:
		#					pass
		#			else:
		#				# if current entry is not ambiguous, keep current Z-matrix and co object in list for next round
		#				new_zmco_list.append(zmco_list[i])
		#		# make new_zmco_list to standard list
		#		zmco_list = new_zmco_list

		## check sum rule for all reconstruction
		#is_sum_rule_fulfilled = [check_sum_rule(my_lineages, zmco_list[i].z_matrix) for i in range(len(zmco_list))]
		## only keep reconstructions that fulfill the sum rule
		#zmco_list_fulfills_sum_rule = list(compress(zmco_list, is_sum_rule_fulfilled))

		## number of valid reconstructions and valid reconstructions
		#return len(zmco_list_fulfills_sum_rule), zmco_list_fulfills_sum_rule
	
	# recursive function 
	else:
		last = None
		defparent = None
		avFreqs = None
		linFreqs = None
		if ppm is not None:
			last = lin_num - 1
			# get definite parents and available frequencies
			linFreqs = np.asarray([my_lineages[i].freq for i in range(len(my_lineages))])
			defparent, avFreqs = get_definite_parents_available_frequencies(linFreqs, ppm)

		return recursive_number_ambiguous_recs(0, 0, lin_num, zmco, my_lineages, 0, filename=filename, count_threshold=count_threshold,
			last=last, ppm=ppm, defparent=defparent, linFreqs=linFreqs, avFreqs=avFreqs, seg_num=seg_num,
			zero_count=zero_count, gain_num=gain_num, loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms)

# recursive function for depth-first search of ambiguous reconstructions
def recursive_number_ambiguous_recs(k_current, k_prime_checked, lin_num, zmco_current, my_lineages, count, filename=None, count_threshold=-1,
	last=None, ppm=None, defparent=None, linFreqs=None, avFreqs=None, seg_num=None, zero_count=None, gain_num=None, loss_num=None, 
	CNVs=None, present_ssms=None):
	# tracks whether this is the last recursive call
	last_call = True
	# iterate through matrix, starting after Z[k][k_prime_checked]
	for k in range(k_current, lin_num-1):
		for k_prime in range(k+1, lin_num):
			# only check entry if it wasn't checked already
			if k == k_current and k_prime <= k_prime_checked:
				continue
			if zmco_current.z_matrix[k][k_prime] == 0:
				# recursion will be called again
				last_call = False
				# copy current Z-matrix and co object for setting entry to 1 (relationship present)
				zmco_dup = copy.deepcopy(zmco_current)
				try:
					# ancestry is updated, lineage relationship absence constraints are propagated only if sum rule
					# should be applied as well, otherwise the SSM phasing takes care that invalid scenarios get caught
					# concerning the case that two losses of the same allele in the same segment should be set into a 
					# present relation, the upper function takes care of this as it can check for invalid reconstructions
					# and because this case is forbidden independent of SSM phasing, the relationship should always be
					# absent
					update_ancestry(1, k, k_prime, last=last, ppm=ppm, defparent=defparent, linFreqs=linFreqs,
						avFreqs=avFreqs, zmco=zmco_dup, seg_num=seg_num, zero_count=zero_count, gain_num=gain_num,
						loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms)
					# go into recursion
					count = recursive_number_ambiguous_recs(k, k_prime, lin_num, zmco_dup, my_lineages, count, filename, count_threshold,
						last=last, ppm=ppm, defparent=defparent, linFreqs=linFreqs, avFreqs=avFreqs, seg_num=seg_num,
						zero_count=zero_count, gain_num=gain_num, loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms)
				except (eo.ZInconsistence, eo.ADRelationNotPossible, eo.ZUpdateNotPossible, eo.NoParentsLeft,
					eo.NoParentsLeftNoise) as e:
					pass
				# set entry of current Z-matrix and co object to -1 (relationship absent)
				try:
					update_ancestry(-1, k, k_prime, last=last, ppm=ppm, defparent=defparent, linFreqs=linFreqs,
					avFreqs=avFreqs, zmco=zmco_current, seg_num=seg_num, zero_count=zero_count, gain_num=gain_num,
					loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms)
					# go into recursion
					count = recursive_number_ambiguous_recs(k, k_prime, lin_num, zmco_current, my_lineages, count, filename, count_threshold,
						last=last, ppm=ppm, defparent=defparent, linFreqs=linFreqs, avFreqs=avFreqs, seg_num=seg_num,
						zero_count=zero_count, gain_num=gain_num, loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms)
				except (eo.ZInconsistence, eo.ADRelationNotPossible, eo.ZUpdateNotPossible, eo.NoParentsLeft,
					eo.NoParentsLeftNoise) as e:
					pass
				# going back one level
				return count

	if last_call:
		# check sum rule
		if check_sum_rule(my_lineages, zmco_current.z_matrix):

			count += 1

			# if Z-matrix should be written to file and maximum number in file is not reached yet
			if filename is not None and count <= count_threshold:
				with open(filename, "a") as f:
					my_string = json.dumps(convert_zmatrix_0_m1(zmco_current.z_matrix))
					f.write("{0}\n".format(my_string))

	return count


		
# given the lineages and a Z-matrix, checks whether the sum rule is fulfilled for all samples
def check_sum_rule(my_lineages, z_matrix):
	sample_size = len(my_lineages[0].freq)
	# check all lineages that can have more than one child
	for k in range(len(my_lineages)-2):
		children = get_children(z_matrix, k)	
		# check whether sum rule is fulfilled for all samples
		for n in range(sample_size):
			if my_lineages[k].freq[n] + cons.EPSILON_FREQUENCY < sum([my_lineages[kp].freq[n] for kp in children]):
				return False
	return True

# checks for each present lineage relationship whether a CN influence exists
# if not, relationship is removed
def check_CN_influence_user_constraints(z_matrix, lineages, user_z=None):

	lin_num = len(lineages)
	
	# get hashs with CN changes and SSM appearances
	CN_changes_hash, SSMs_hash = create_CN_changes_and_SSM_hash_for_LDR(lineages)

	# iterate through Z-matrix
	for k in range(1, lin_num-1):
		for k_prime in range(k+1, lin_num):
			# check CN influecne, if a 1 is in Z-matrix
			if z_matrix[k][k_prime] == 1:
				# if no CN influence is present
				if not is_CN_influence_present(k, k_prime, CN_changes_hash, SSMs_hash):
					# user did not specify constraints, relationship is removed
					if user_z is None:
						z_matrix[k][k_prime] = 0
					# user did specify constraints, but not for this relationship
					elif user_z[k][k_prime] != 1:
						z_matrix[k][k_prime] = 0
			# if entry is not a 1 and user constrained it to be a -1
			elif user_z is not None and user_z[k][k_prime] == -1:
				z_matrix[k][k_prime] = -1

# checks whether a CN change in lineage k_prime influences the VAF of an SSM in lineage k
# this is the case, when one CN change in k_prime lies on the same segment and phase as one SSM in k
def is_CN_influence_present(k, k_prime, CN_changes_hash, SSMs_hash):
	# check all segments in which k_prime contains CN changes
	try:
		for seg in CN_changes_hash[k_prime].keys():
			# check all phases of the CN changes
			for phase in CN_changes_hash[k_prime][seg]:
				try:
					# if an SSM in k appears on the same segment with the same phase,
					# there is an influence
					if phase in SSMs_hash[k][seg]:
						return True
				# there is no SSM on the segment in lineage k
				except KeyError:
					pass
	# lineage k_prime doesn't contain any CN changes
	except:
		return False

	# when no CN influence was found, there is non
	return False

# creates a hash with the CN changes and one with SSMs
# CN_changes hash: [lineage][segment] = [list with phases]
# SSM hash: [lineage][segment] = [list with phases]
def create_CN_changes_and_SSM_hash_for_LDR(lineages):
	# create empty hashes
	CN_changes = {}
	SSMs = {}

	# iterate through all lineages
	for lin_index, my_lin in enumerate(lineages):
		# don't consider normal lineage
		if lin_index == 0:
			continue

		# add CNVs
		add_CN_changes_to_hash(CN_changes, my_lin.cnvs_a, cons.A, lin_index)
		add_CN_changes_to_hash(CN_changes, my_lin.cnvs_b, cons.B, lin_index)

		# add SSMs
		add_SSM_appearence_to_hash(SSMs, my_lin.ssms_a, cons.A, lin_index)
		add_SSM_appearence_to_hash(SSMs, my_lin.ssms_b, cons.B, lin_index)

	return CN_changes, SSMs

# for each lineage, an entry is added for the first SSM of one phase in a segment
def add_SSM_appearence_to_hash(SSMs_hash, ssms, phase, lin_index):
	for ssm in ssms:
		try:
			# it's sufficient that the entry is done for the first SSM with the phase on the segment
			if phase in SSMs_hash[lin_index][ssm.seg_index]:
				continue
			SSMs_hash[lin_index][ssm.seg_index].append(phase)
		except KeyError:
			try:
				SSMs_hash[lin_index][ssm.seg_index] = [phase]
			except KeyError:
				SSMs_hash[lin_index] = {}
				SSMs_hash[lin_index][ssm.seg_index] = [phase]

# adds the CNVs in a list to the hash
def add_CN_changes_to_hash(CN_changes_hash, cnvs, phase, lin_index):
	for cnv in cnvs:
		try:
			if phase in CN_changes_hash[lin_index][cnv.seg_index]:
				raise eo.MyException("Only one CNV in this lineage in this segment should have this phase.")
			CN_changes_hash[lin_index][cnv.seg_index].append(phase)
		except KeyError:
			try:
				CN_changes_hash[lin_index][cnv.seg_index] = [phase]
			except KeyError:
				CN_changes_hash[lin_index] = {}
				CN_changes_hash[lin_index][cnv.seg_index] = [phase]
				
# checks whether output files should be overwritten
# if they shouldn't be overwritten but files with the same name exists, return False
def check_overwrite(overwrite, output_prefix):
	if overwrite == False:
		try:
			oio.raise_if_file_exists("{0}.zmatrix".format(output_prefix))
			return True
		except eo.FileExistsException as e:
			logging.error("Files for output prefix {0} exist already.\nTerminating SubMARine.".format(output_prefix))
			return False

# sets entries in ancestry matrix Z according to user constraints
# checks for invalid values
def set_z_user_constraints(z_matrix, user_z):
	lin_num = len(z_matrix)
	for k in range(lin_num):
		for kp in range(lin_num):
			if k == 0 and kp > 0:
				if user_z[k][kp] == -1:
					error_message = ("Invalid user defined ancestry constraints. Germline has to be ancestor of subclone {0}."
						.format(kp))
					logging.error(error_message)
					raise eo.MyException(error_message)
			if k >= kp:
				if user_z[k][kp] == 1:
					error_message = ("Invalid user defined ancestry constraints."
						"Subclone {0} cannot be an ancestor of subclone {1}".format(k, kp))
					logging.error(error_message)
					raise eo.MyException(error_message)
			else:
				if user_z[k][kp] != 0:
					z_matrix[k][kp] = user_z[k][kp]

def check_overwrite_logging_lineages_Z_constraints(overwrite=None, output_prefix=None, use_logging=None, freq_file=None, userZ_file=None):
	# check whether output files exist already
	if check_overwrite(overwrite, output_prefix) == False:
		return

	# create logging file
	if use_logging:
		for handler in logging.root.handlers[:]:
			logging.root.removeHandler(handler)
		log_file_name = "{0}.log".format(output_prefix)
		logging.basicConfig(filename=log_file_name, filemode='w', level=logging.INFO)
	
	logging.info("Parsing input information.")

	# create lineage object list
	my_lins, sorting_id_mapping = get_lineages_from_freqs(freq_file)
	lin_num = len(my_lins)

	# get user constraints
	user_z = None
	if userZ_file is not None:
	    user_z = oio.read_userZ(userZ_file, lin_num, sorting_id_mapping=sorting_id_mapping)

	return my_lins, sorting_id_mapping, lin_num, user_z

def create_z_germline_trivial_user_vars_tree_rule(my_lins=None, user_z=None, lin_num=None):
	# create Z-matrix, apply germline constraints and trivial relationships
	z_matrix = get_Z_matrix(my_lins)[0]

	# set user constraints if exist
	if user_z is not None:
		set_z_user_constraints(z_matrix, user_z)
	
	# create variabls needed for tree rules
	logging.debug("check_and_update_complete_Z_matrix_from_matrix")
	zero_count = lin_num * lin_num
	zero_count, triplet_xys, triplet_ysx, triplet_xsy = check_and_update_complete_Z_matrix_from_matrix(
	        z_matrix, zero_count, lin_num)

	return z_matrix, triplet_xys, triplet_ysx, triplet_xsy

# change SSM phasing if SSMs are impacted by CNA
def change_ssm_phasing_according_to_impact(impact_matrix=None, my_ssms=None, my_cnas=None):
	# no CNAs or SSMs are gicen
	if impact_matrix is None:
		return True
	
	# check each SSM
	for current_ssm in my_ssms:
		ssm_index = current_ssm.index
		# SSM is impacted by CNAs
		if impact_matrix[ssm_index].sum() > 0:
			for cna_index in np.where(impact_matrix[ssm_index] == 1)[0]:
				
				if current_ssm.seg_index !=  my_cnas[cna_index].seg_index:
					raise eo.MyException("SSM {0} cannot be impacted by CNA {1} since they are not "
						"in the same segment.".format(ssm_index, cna_index))

				# adapt phasing of SSM
				if current_ssm.phase is None:
					current_ssm.phase = my_cnas[cna_index].phase
				# SSM is already phased to other allele
				elif current_ssm.phase != my_cnas[cna_index].phase:
					raise eo.MyException("Impact matrix is incorrect. SSM {0} cannot be "
						"impacted by multiple CNAs with different phasing".format(ssm_index))

	return True

# goes through all CNAs and SSMs and finds largest segment index to get number of segments
def get_seg_num(my_cnas, my_ssms):
	seg_num = 1

	for current_cna in my_cnas:
		seg_index = current_cna.seg_index
		if seg_index >= seg_num:
			seg_num = seg_index + 1
	for current_ssm in my_ssms:
		seg_index = current_ssm.seg_index
		if seg_index >= seg_num:
			seg_num = seg_index + 1

	return seg_num

# updates ancestral relationships according to impact matrix
def update_ADRs_according_impact_matrix(z_matrix=None, triplet_xys=None, triplet_ysx=None, triplet_xsy=None, 
		impact_matrix=None, my_ssms=None, my_cnas=None):

	# create temporary zmco object
	tmp_zmco = Z_Matrix_Co(z_matrix, triplet_xys, triplet_ysx, triplet_xsy, present_ssms=None, matrix_after_first_round=None)

	# check all influences
	impacts = np.where(impact_matrix == 1)
	for i in range(len(impacts[0])):
		ssm_lineage = my_ssms[impacts[0][i]].lineage
		cna_lineage = my_cnas[impacts[1][i]].lineage

		# SSM lineage needs to have lower index than CNA lineage
		if ssm_lineage > cna_lineage:
			raise eo.MyException("Invalid impact matrix. SSM's subclone needs to have lower index than CNA's subclone. "
				"Not the case for SSM {0}, assigned to subclone {1}, and CNA {2}, assigned to subclone {3}.".format(
				impacts[0][i], ssm_lineage, impacts[1][i], cna_lineage))
		# lineage of SSM and CNA are equal
		if ssm_lineage == cna_lineage:
			continue

		# update relationship
		update_ancestry(1, ssm_lineage, cna_lineage, zmco=tmp_zmco)

# propagate SSM phasing, different to CNA influence or lost allele
def propagate_negative_SSM_phasing_according_to_impact_matrix(my_ssms, my_cnas, z_matrix, impact_matrix, my_lins):

	if impact_matrix is None:
		return True

	# propagate SSM phasing with Equation 14 (different phase because of impact matrix)
	for current_ssm in my_ssms:
		for current_cna in my_cnas:
			ssm_index = current_ssm.index
			cna_index = current_cna.index
			if (impact_matrix[ssm_index][cna_index] == 0 and z_matrix[current_ssm.lineage][current_cna.lineage] == 1
				and current_ssm.seg_index == current_cna.seg_index):
				if current_ssm.phase is None:
					current_ssm.phase = other_phase(current_cna.phase)
				elif current_ssm.phase == current_cna.phase:
					raise eo.MyException("SSM phases cannot be propagated. SSM {0} needs to have both "
						"phases at the same time.".format(ssm_index))

	# propagate SSM phasing with Equation 7 (lost allele)
	for current_ssm in my_ssms:
		k = current_ssm.lineage
		ancestors = [k_star for k_star in range(1, k) if z_matrix[k_star][k] == 1]
		for phase in [cons.A, cons.B]:
			# get change of ancestors  of phase of segment where SSM lies on
			change = 0
			for k_star in ancestors:
				if phase == cons.A:
					change += sum([x.change for x in my_lins[k_star].cnvs_a if x.seg_index == current_ssm.seg_index])
				if phase == cons.B:
					change += sum([x.change for x in my_lins[k_star].cnvs_b if x.seg_index == current_ssm.seg_index])
			if phase == cons.A:
				change += sum([x.change for x in my_lins[k].cnvs_a if x.seg_index == current_ssm.seg_index])
			if phase == cons.B:
				change += sum([x.change for x in my_lins[k].cnvs_b if x.seg_index == current_ssm.seg_index])

			if change < 0:
				if current_ssm.phase == None:
					current_ssm.phase = other_phase(phase)
				elif current_ssm.phase == phase:
					raise eo.MyException("SSM phases cannot be propagated. SSM {0} needs to have both "
						"phases at the same time.".format(ssm_index))

# given the lineages, takes the SSMs and their phasing and sorts them according to their index
def get_new_ssm_phasing(my_lins):
	ssm_list = []
	for current_lineage in my_lins:
		all_ssms = current_lineage.ssms + current_lineage.ssms_a + current_lineage.ssms_b
		for current_ssm in all_ssms:
			ssm_list.append([current_ssm.index, current_ssm.phase])

	ssm_list = sorted(ssm_list, key = lambda x: x[0])
	return ssm_list

# checks whether the monotoncity restriction holds for each segment and allele
def check_monotonicity(my_cnas):
	my_cnas = sorted(my_cnas, key = lambda x: (x.seg_index, x.phase))

	for i in range(len(my_cnas)-1):
		# CNA of same segment and allele
		if my_cnas[i].seg_index == my_cnas[i+1].seg_index and my_cnas[i].phase == my_cnas[i+1].phase:
			# one CNA is gain and one is loss
			if (my_cnas[i].change > 0 and my_cnas[i+1].change < 0) or (my_cnas[i].change < 0 and my_cnas[i+1].change > 0):
				if my_cnas[i].phase == cons.A:
					phase = "A"
				else:
					phase = "B"
				raise eo.MyException("Segment {0} and phase {1} don't held monotonicity constraint.".format(
					my_cnas[i].seg_index, phase))

	return True

					
def go_extended_version(freq_file=None, cna_file=None, ssm_file=None, impact_file=None, userZ_file=None, userSSM_file=None,
		output_prefix=None, overwrite=False, use_logging=True, allow_noise=False, noise_threshold=0, maximal_noise=-1):

	# check whether output files exist already
	# create logging file
	# create lineage object list
	# get user constraints
	my_lins, sorting_id_mapping, lin_num, user_z = check_overwrite_logging_lineages_Z_constraints(overwrite=overwrite, 
		output_prefix=output_prefix, use_logging=use_logging, freq_file=freq_file, userZ_file=userZ_file)

	# read CNA file and assign CNAs
	my_cnas = oio.read_cnas(cna_file, sorting_id_mapping=sorting_id_mapping, use_cna_indices=True, lin_num=lin_num)
	check_monotonicity(my_cnas)
	cna_num = add_CNAs(my_lins=my_lins, lin_num=lin_num, my_cnas=my_cnas)

	# read SSM file
	my_ssms = oio.read_ssms(ssm_file, phasing=False, sorting_id_mapping=sorting_id_mapping, use_SSM_index=True, lin_num=lin_num)
	ssm_num = len(my_ssms)

	# read impact file
	impact_matrix = oio.create_impact_matrix(impact_file=impact_file, cna_num=cna_num, ssm_num=ssm_num)

	# start SupMARine
	logging.info("Starting SubMARine.")

	# create Z-matrix, apply germline constraints and trivial relationships
	# set user constraints if exist
	# create variabls needed for tree rules
	z_matrix, triplet_xys, triplet_ysx, triplet_xsy = create_z_germline_trivial_user_vars_tree_rule(my_lins=my_lins, user_z=user_z,
		lin_num=lin_num)

	# get segment number
	seg_num = get_seg_num(my_cnas, my_ssms)

	# change SSM phasing according to impact matrix
	change_ssm_phasing_according_to_impact(impact_matrix=impact_matrix, my_ssms=my_ssms, my_cnas=my_cnas)	
	# apply user SSM constraints
	if userSSM_file is not None:
		oio.get_SSM_constraints(userSSM_file=userSSM_file, my_ssms=my_ssms)
	# update ancestral relationships according to impact matrix
	update_ADRs_according_impact_matrix(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, 
		impact_matrix=impact_matrix, my_ssms=my_ssms, my_cnas=my_cnas)
	# propagate SSM phasing
	propagate_negative_SSM_phasing_according_to_impact_matrix(my_ssms, my_cnas, z_matrix, impact_matrix, my_lins)

	# assign SSMs to lineages
	ssm_num = add_SSMs(my_lins=my_lins, lin_num=lin_num, my_ssms=my_ssms)

	# crossing rule
	logging.debug("crossing rule")
	zero_count = lin_num * lin_num
	zero_count = check_crossing_rule_function(my_lins, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy)

	# go once through segment and get gains, losses and SSMs
	gain_num = []
	loss_num = []
	CNVs = []
	present_ssms = []
	ssm_infl_cnv_same_lineage = []
	# iterate through all segments once to get all CN changes and SSMs appearances
	get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lin_num, my_lins,
		ssm_infl_cnv_same_lineage)
	# copy present_ssm list for later
	origin_present_ssms = copy.deepcopy(present_ssms)
	# check for pairwise and easy triplet-wise constraints that lead to absent relationships
	logging.debug("propagate pairwise and easy-triplet wise constraints")
	z_matrix_list, z_matrix_fst_rnd, triplets_list = (
	        post_analysis_Z_matrix(my_lins, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy,
	        matrix_splitting=False, first_absence_propagation=True, CNVs=CNVs, present_ssms=present_ssms, gain_num=gain_num,
		loss_num=loss_num))
	# check whether sum rule leads to unambiguous relationships
	logging.debug("using sum rule")
	zmcos = create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, [present_ssms], triplets_list)
	if len(zmcos) > 1:
		raise eo.MyException("zmcos should only have one entry!")
	zmco = zmcos[0]
	frequencies = np.asarray([my_lins[i].freq for i in range(len(my_lins))])
	try:
		dummy, avFreqs, ppm, zmco, zero_count, present_ssms = outer_subpoplar_w_noise(frequencies, zmco, seg_num, zero_count,
			gain_num, loss_num, CNVs, present_ssms, allow_noise=allow_noise, noise_threshold=noise_threshold,
			maximal_noise=maximal_noise)
	except (eo.NoParentsLeft, eo.NoParentsLeftNoise) as e:
		logging.warning(str(e))
		logging.info("SubMARine coudn't finish because sum constraint cannot be satisfied.")
		return
	except eo.MyException as e:
		raise e

	# adapt lineages because SSMs can be phased differently after sum rule, also, relationships can differ
	logging.debug("adapt lineages after sum rule")
	my_lins = create_updates_lineages(my_lins, 0, [zmco.z_matrix], origin_present_ssms, [zmco.present_ssms])
	ssm_phasing = get_new_ssm_phasing(my_lins)

	# convert values in Z-matrix to values used in paper
	z_matrix_for_output = convert_zmatrix_0_m1(zmco.z_matrix)

	# if no output prefix is given, results are not written to files but returned
	if output_prefix is not None:
		# print to file
		logging.debug("Printing results to file.")
		oio.write_matrix_to_file(z_matrix_for_output, "{0}.zmatrix".format(output_prefix), overwrite)
		np.savetxt("{0}.pospars".format(output_prefix), ppm, delimiter=",", fmt='%1.0f')
		oio.write_result_file_as_JSON(my_lins, "{0}.lineage.json".format(output_prefix), test=overwrite)
		oio.write_new_ssm_phasing(ssm_phasing, "{0}.ssm_phasing".format(output_prefix), overwrite)

	logging.info("SubMARine is done.")

	return my_lins, z_matrix_for_output, avFreqs, ppm, ssm_phasing

# check whether all CNAs are assigned to subclone 1
def check_all_clonal(my_cnas):
	for current_cna in my_cnas:
		if current_cna.lineage != 1:
			raise eo.MyException("CNAs in basic version need to be all clonal but CNA {0} is assigned to subclone {1}.".format(
				current_cna.index, current_cna.lineage))
	
	return True

# check whether two alleles of the same segment got deleted
def get_deleted_segments(my_cnas):
	del_segments = []

	sorted_cnas = sorted(my_cnas, key = lambda x: x.seg_index)
	for i in range(len(sorted_cnas) - 1):
		if sorted_cnas[i].seg_index == sorted_cnas[i+1].seg_index and sorted_cnas[i].phase == other_phase(sorted_cnas[i+1].phase
			and sorted_cnas[i].change == -1 and sorted_cnas[i+1].change == -1):
			del_segments.append(sorted_cnas[i].seg_index)

	return del_segments

# checks whether SSMs are assigned to the segments that are deleted
def check_lost_alleles_for_basic(del_segments, my_ssms):

	sorted_ssms = sorted(my_ssms, key = lambda x: x.seg_index)

	del_seg_index = 0
	for i in range(len(sorted_ssms)):
		while sorted_ssms[i].seg_index > del_segments[del_seg_index] and del_seg_index < len(del_segments) - 1:
			del_seg_index += 1
		if sorted_ssms[i].seg_index == del_segments[del_seg_index]:
			raise eo.MyException("Lost alleles constraint not satisfies. SSM {0} appears in deleted segment {1}.".format(
				sorted_ssms[i].index, sorted_ssms[i].seg_index))

	return True

def go_basic_version(freq_file=None, userZ_file=None, output_prefix=None, overwrite=False, use_logging=True, cna_file=None,
	ssm_file=None, allow_noise=False, noise_threshold=0, maximal_noise=-1):

	# check whether output files exist already
	# create logging file
	# create lineage object list
	# get user constraints
	my_lins, sorting_id_mapping, lin_num, user_z = check_overwrite_logging_lineages_Z_constraints(overwrite=overwrite, 
		output_prefix=output_prefix, use_logging=use_logging, freq_file=freq_file, userZ_file=userZ_file)

	if cna_file is not None:
		logging.info("Preprocessing whether lost allele constraint is satisfied.")
		my_cnas = oio.read_cnas(cna_file, sorting_id_mapping=sorting_id_mapping, use_cna_indices=True, lin_num=lin_num)
		check_monotonicity(my_cnas)
		check_all_clonal(my_cnas)
		del_segments = get_deleted_segments(my_cnas)
		if len(del_segments) != 0:
			if ssm_file is not None:
				my_ssms = oio.read_ssms(ssm_file, phasing=False, sorting_id_mapping=sorting_id_mapping, 
					use_SSM_index=True, lin_num=lin_num)
				check_lost_alleles_for_basic(del_segments, my_ssms)

	# start SubMARine
	logging.info("Starting SubMARine.")

	# create Z-matrix, apply germline constraints and trivial relationships
	# set user constraints if exist
	# create variabls needed for tree rules
	z_matrix, triplet_xys, triplet_ysx, triplet_xsy = create_z_germline_trivial_user_vars_tree_rule(my_lins=my_lins, user_z=user_z,
		lin_num=lin_num)
	
	# crossing rule
	logging.debug("crossing rule")
	zero_count = lin_num * lin_num
	zero_count = check_crossing_rule_function(my_lins, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy)

	# check whether sum rule leads to unambiguous relationships
	logging.debug("using sum rule")
	# create needed variables
	gain_num = []
	loss_num = []
	CNVs = []
	present_ssms = []
	ssm_infl_cnv_same_lineage = []
	seg_num = 1
	get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lin_num, my_lins,
		ssm_infl_cnv_same_lineage)
	triplets_list = [[triplet_xys, triplet_ysx, triplet_xsy]]
	z_matrix = np.asarray(z_matrix)
	zmco = create_Z_Matrix_Co_objects([z_matrix], z_matrix, [present_ssms], triplets_list)[0]
	frequencies = np.asarray([my_lins[i].freq for i in range(len(my_lins))])
	try:
		dummy, avFreqs, ppm, zmco, zero_count, present_ssms = outer_subpoplar_w_noise(frequencies, zmco, seg_num, zero_count,
			gain_num, loss_num, CNVs, present_ssms, allow_noise=allow_noise, noise_threshold=noise_threshold,
			maximal_noise=maximal_noise)
	except (eo.NoParentsLeft, eo.NoParentsLeftNoise) as e:
		logging.warning(str(e))
		logging.info("SubMARine coudn't finish because sum constraint cannot be satisfied.")
		return str(e)
	except eo.MyException as e:
		raise e
	
	# adapt lineages because relationships can differ
	logging.debug("adapt subclones after sum rule")
	my_lins = create_updates_lineages(my_lins, 0, [zmco.z_matrix], zmco.present_ssms, [zmco.present_ssms])

	# convert values in Z-matrix to values used in paper
	z_matrix_for_output = convert_zmatrix_0_m1(zmco.z_matrix)

	# if no output prefix is given, results are not written to files but returned
	if output_prefix is not None:
		# print to file
		logging.debug("Printing results to file.")
		oio.write_matrix_to_file(z_matrix_for_output, "{0}.zmatrix".format(output_prefix), overwrite)
		np.savetxt("{0}.pospars".format(output_prefix), ppm, delimiter=",", fmt='%1.0f')
		oio.write_result_file_as_JSON(my_lins, "{0}.lineage.json".format(output_prefix), test=overwrite)

	logging.info("SubMARine is done.")

	return my_lins, z_matrix_for_output, avFreqs, ppm

# function to call Subpoplar/sum rule algorithm and that takes care of noise threshold
def outer_subpoplar_w_noise(frequencies, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms,
	allow_noise=False, noise_threshold=0, maximal_noise=-1):

	# copy objects so that the original stay unchanged during the Subpoplar try
	zmco_copy = copy.deepcopy(zmco)
	zero_count_copy = zero_count
	present_ssms_copy = copy.deepcopy(present_ssms)

	try:
		dummy, avFreqs, ppm = sum_rule_algo_outer_loop(frequencies, zmco_copy, seg_num, zero_count_copy,
			gain_num, loss_num, CNVs, present_ssms_copy, noise_threshold=noise_threshold)
	# noise threshold was reached
	# check whether it can be decreased, then do Subpoplar again
	except eo.NoParentsLeftNoise as e:
		del zmco_copy, zero_count_copy, present_ssms_copy

		# if no noise is allowed, raise error
		if allow_noise == False:
			raise e

		# compute new noise threshold
		noise_threshold = compute_minimal_noise_threshold(e.k, frequencies, e.avFreqs_from_initial_pps)
		# if noise threshold is too high, raise error
		if maximal_noise > 0 and noise_threshold > maximal_noise + cons.EPSILON_FREQUENCY:
			logging.warning("Allowed noise threshold of {0} is smaller than necessary noise thresholf of {1}.".format(
				maximal_noise, noise_threshold))
			raise e

		dummy, avFreqs, ppm, zmco_copy, zero_count_copy, present_ssms_copy = outer_subpoplar_w_noise(frequencies, zmco, seg_num, 
			zero_count, gain_num, loss_num, CNVs, present_ssms,
			allow_noise=allow_noise, noise_threshold=noise_threshold, maximal_noise=maximal_noise)

	if allow_noise:
		logging.info("Subpoplar finished with noise threshold of {0}.".format(noise_threshold))

	return dummy, avFreqs, ppm, zmco_copy, zero_count_copy, present_ssms_copy

def go_submarine(parents_file=None, freq_file=None, cna_file=None, ssm_file=None, seg_file=None, userZ_file=None, userSSM_file=None, output_prefix=None,
	overwrite=False):
	# checks whether output files exist already
	if check_overwrite(overwrite, output_prefix) == False:
		return

	# create lineage object list
	my_lins = get_lineages_from_input_files(parents_file, freq_file, cna_file, ssm_file)
	lin_num = len(my_lins)
	# create Z-matrix
	z_matrix = get_Z_matrix(my_lins)[0]
	# set all ambiguous entries to absent for now
	for k in range(1, lin_num):
		for kp in range(k+1, lin_num):
			if z_matrix[k][kp] == 0:
				z_matrix[k][kp] = -1
	# get segment number
	with open(seg_file, "r") as f:
		for line in f:
			seg_num = int(line.rstrip())
	# get user constraints
	user_z = None
	if userZ_file is not None:
	    user_z = oio.read_userZ(userZ_file, lin_num)
	user_ssm = None
	if userSSM_file is not None:
	    user_ssm = oio.read_userSSM(userSSM_file, lin_num, seg_num)

	# if CNAs are given, subclonal reconstruction has to be valid
	if cna_file is not None and not is_reconstruction_valid(my_lins, z_matrix, None, seg_num):
		logging.error("File with CNA information is given but provided subclonal reconstruction is not valid. Cannot run SubMARine.")
		return 

	# start SubMARine
	logging.info("Starting SubMARine.")
	my_lins, z_matrix, avFreqs, ppm = get_all_possible_z_matrices_with_lineages_new(my_lins, seg_num, user_z, user_ssm)
	logging.info("SubMARine finished successfully.")

	# print to file
	logging.info("Printing results to file.")
	#np.savetxt("{0}.zmatrix".format(output_prefix), z_matrix, delimiter=",", fmt='%1.0f')
	z_matrix_for_output = convert_zmatrix_0_m1(z_matrix)
	oio.write_matrix_to_file(z_matrix_for_output, "{0}.zmatrix".format(output_prefix), overwrite)
	np.savetxt("{0}.pospars".format(output_prefix), ppm, delimiter=",", fmt='%1.0f')
	oio.print_ssm_phasing(my_lins, "{0}_ssms.csv".format(output_prefix), overwrite)
	oio.write_result_file_as_JSON(my_lins, "{0}.lineage.json".format(output_prefix), test=overwrite)

# converts a 0 to a -1 and a -1 to a 0 as described in the paper
def convert_zmatrix_0_m1(z_matrix):
	lin_num = len(z_matrix)
	new_z_matrix = np.ones(lin_num * lin_num).reshape(lin_num, lin_num).tolist()
	for k in range(lin_num):
		for k2 in range(lin_num):
			if z_matrix[k][k2] == 0:
				new_z_matrix[k][k2] = -1
			elif z_matrix[k][k2] == -1:
				new_z_matrix[k][k2] = 0
			elif z_matrix[k][k2] == 1:
				new_z_matrix[k][k2] = 1
	return new_z_matrix

def convert_zmatrix_for_internal_use(z_matrix):
	assert z_matrix[0][0] == 0
	lin_num = len(z_matrix)
	for k in range(lin_num):
		for k2 in range(lin_num):
			if z_matrix[k][k2] == 0:
				z_matrix[k][k2] = -1
			# different versions of the code used different values to show ambiguity
			elif z_matrix[k][k2] == "?":
				z_matrix[k][k2] = 0
			elif z_matrix[k][k2] == -1:
				z_matrix[k][k2] = 0

# given the lineage frequencies, create a lineage object list
def get_lineages_from_freqs(freq_file=None, freqs=None, freq_num=None, lin_num=None, lin_ids=None):
	
	logging.info("Sorting subclones according to average frequencies.")

	# get frequencies
	if freq_file is not None:
		freqs, lin_ids = oio.read_frequencies(freq_file, ordering_given=False, return_ids=True)
		freq_num = len(freqs[0])
		lin_num = len(freqs) + 1
	
	# sort frequencies according to average frequency
	# get average of list
	av_freqs = [np.average(freqs[i]) for i in range(lin_num-1)]
	# indices of argsort
	sorted_indices = np.flip(np.argsort(av_freqs), 0)
	# check whether multiple lineages with same average frequencies exist
	## UPDATE: even if lineages have exact same frequencies in all samples, don't try to merge them anymore
	#i = 0
	#while i < lin_num - 2:
	#	same_freq = []
	#	next_lin = i+1
	#	check_next_lin = True
	#	# get all lineages that have the same average frequency
	#	while(check_next_lin):
	#		if freqs[sorted_indices[i]] == freqs[sorted_indices[next_lin]]:
	#			same_freq = same_freq + [sorted_indices[next_lin]]
	#			next_lin += 1
	#			if next_lin >= lin_num-1:
	#				check_next_lin = False
	#		else:
	#			check_next_lin = False

	#	# other lineages have the same frequency
	#	if len(same_freq) > 0:
	#		# append index of current lineage
	#		same_freq = [sorted_indices[i]] + same_freq
	#		# if lineages have different frequencies per sample, nothing needs to be done
	#		for j in range(len(same_freq)-1):
	#			#TODO also check for other attributes to differentiate them
	#			if freqs[same_freq[j]] == freqs[same_freq[j+1]]:
	#				error_message = ("Subclones {0} and {1} have exactly same frequencies, cannot be differentiated."
	#					.format(same_freq[j], same_freq[j+1]))
	#				logging.error(error_message)
	#				raise eo.MyException(error_message)
	#	# update i
	#	i = next_lin

	# create mapping between ordering and IDs
	mapping = create_ID_ordering_mapping(sorted_indices, lin_ids)

	# create lineages
	normal_lineage = lineage.Lineage([i for i in range(1, lin_num)], [1.0] * freq_num, [], [], [], [], [], [], [], [])
	other_lineages = [lineage.Lineage([], freqs[sorted_indices[i]], [], [], [], [], [], [], [], [])
		for i in range(lin_num-1)]

	return [normal_lineage] + other_lineages, mapping

# creates a mapping between IDs of lineages and ordering according to their frequencies
def create_ID_ordering_mapping(sorted_indices, lin_ids):
	mapping = {}
	output_message = ["Subclone index to ID mapping: 0->germline"]
	for i in range(len(sorted_indices)):
		my_key = lin_ids[sorted_indices[i]]
		try:
			# this statement either throws a KeyError or a TypeError
			mapping[my_key][0] = None
		except KeyError:
			mapping[my_key] = i+1
		except TypeError:
			raise eo.MyException("lineage ID has to be unique.")

		output_message.append("{0}->{1}".format(i+1, my_key))
	
	logging.info(", ".join(output_message))
	return mapping

# adds the CNAs to the lineages
def add_CNAs(cna_file=None, my_lins=None, lin_num=None, sorting_id_mapping=None, use_cna_indices=False, my_cnas=None):
	
	cna_num = -1

	# read CNAs
	if cna_file is not None:
		my_cnas = oio.read_cnas(cna_file, sorting_id_mapping=sorting_id_mapping, use_cna_indices=use_cna_indices)

	# assign CNAs
	if my_cnas is not None:
		cna_num = len(my_cnas)
		for cna in my_cnas:
			if cna.phase == cons.A:
				my_lins[cna.lineage].cnvs_a.append(cna)
			else:
				my_lins[cna.lineage].cnvs_b.append(cna)
		# sort CNAs
		for k in range(1, lin_num):
			my_lins[k].cnvs_a = sort_segments(my_lins[k].cnvs_a, segment=True)
			check_only_one_CNA_per_segment_lineage_phase(my_lins[k].cnvs_a)
			my_lins[k].cnvs_b = sort_segments(my_lins[k].cnvs_b, segment=True)
			check_only_one_CNA_per_segment_lineage_phase(my_lins[k].cnvs_b)

	return cna_num

# checks that each allel of one lineage can only have one CNA of the same segment
def check_only_one_CNA_per_segment_lineage_phase(cna_list_per_lin_phase):
	for i in range(len(cna_list_per_lin_phase)-1):
		if cna_list_per_lin_phase[i].seg_index == cna_list_per_lin_phase[i+1].seg_index:
			raise eo.MyException("CNAs {0} and {1} are in the same segment and thus cannot be assigned to the same "
				"allele and subclone. Please combine them to one CNA.".format(cna_list_per_lin_phase[i].index,
				cna_list_per_lin_phase[i+1].index))

def add_SSMs(ssm_file=None, my_lins=None, lin_num=None, phasing=None, sorting_id_mapping=None, use_SSM_index=False, my_ssms=None):

	ssm_num = -1

	# read SSMs
	if ssm_file is not None:
		my_ssms = oio.read_ssms(ssm_file, phasing=phasing, sorting_id_mapping=sorting_id_mapping, use_SSM_index=use_SSM_index)

	# assign SSMs
	if my_ssms is not None:
		ssm_num = len(my_ssms)
		for ssm in my_ssms:
			if ssm.phase == cons.A:
				my_lins[ssm.lineage].ssms_a.append(ssm)
			elif ssm.phase == cons.B:
				my_lins[ssm.lineage].ssms_b.append(ssm)
			else:
				my_lins[ssm.lineage].ssms.append(ssm)
		# sort SSMs per lineage
		for k in range(1, lin_num):
			my_lins[k].ssms_a = sorted(my_lins[k].ssms_a, key = lambda x: x.seg_index)
			my_lins[k].ssms_b = sorted(my_lins[k].ssms_b, key = lambda x: x.seg_index)

	return ssm_num

# given information about parents, frequencies, cnas and ssms in a file, create a lineage object
def get_lineages_from_input_files(parents_file=None, freq_file=None, cna_file=None, ssm_file=None):

	# read parent information
	if parents_file is not None:
		parent_vector = oio.read_parent_vector(parents_file)

	# get frequencies
	freqs = oio.read_frequencies(freq_file)
	freq_num = len(freqs[0])
	lin_num = len(freqs) + 1

	# create normal and other lineages, all with correct frequencies
	my_lins = ([lineage.Lineage([i for i in range(1, lin_num)], [1.0] * freq_num, [], [], [], [], [], [], [], [])] +
		[lineage.Lineage([], freqs[i-1], [], [], [], [], [], [], [], []) for i in range(1, lin_num)])

	# get sublinages of cancerous lineages
	if parents_file is not None:
		build_sublin_lists_from_parental_info(my_lins, parent_vector)

	# add and assign CNAs
	add_CNAs(cna_file=cna_file, my_lins=my_lins, lin_num=lin_num)

	# get and assign SSMs
	add_SSMs(ssm_file=ssm_file, my_lins=my_lins, lin_num=lin_num, phasing=True)

	return my_lins

# applies ambiguity algorithm on reconstruction
#   reconstruction does not have to be complete
def get_all_possible_z_matrices_with_lineages_new(my_lineages, seg_num, user_z=None, user_ssm=None):

	# get Z-matrix from lineage objects
	z_matrix = get_Z_matrix(my_lineages)[0]
	# only keep CN influence 1's
	logging.debug("check_CN_influence_user_constraints")
	check_CN_influence_user_constraints(z_matrix, my_lineages, user_z)
	z_matrix_after_CN_influence_check = copy.deepcopy(z_matrix)
	# get number of 0's in Z-matrix
	zero_count = get_0_number_in_z_matrix(z_matrix)

	# preprocessing to remove unnessary SSM phasing
	lineage_num = len(my_lineages)
	# go once through segment and get gains, losses and SSMs
	gain_num = []
	loss_num = []
	CNVs = []
	present_ssms = []
	ssm_infl_cnv_same_lineage = []
	# iterate through all segments once to get all CN changes and SSMs appearances
	get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
		ssm_infl_cnv_same_lineage)
	# copy present_ssm list for later
	origin_present_ssms = copy.deepcopy(present_ssms)
	# remove unnessary SSM phasing
	logging.debug("remove unnecessary phasing")
	change_unnecessary_phasing(len(my_lineages), CNVs, present_ssms, ssm_infl_cnv_same_lineage, z_matrix, seg_num, user_ssm)
	
	# check for tree rules and update if necessary
	logging.debug("check_and_update_complete_Z_matrix_from_matrix")
	zero_count, triplet_xys, triplet_ysx, triplet_xsy = check_and_update_complete_Z_matrix_from_matrix(
	        z_matrix, zero_count, lineage_num, CNVs, present_ssms, z_matrix_after_CN_influence_check)
	# check for pairwise and easy triplet-wise constraints that lead to absent relationships
	logging.debug("propagate pairwise and easy-triplet wise constraints")
	zero_count = check_crossing_rule_function(my_lineages, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy)
	z_matrix_list, z_matrix_fst_rnd, triplets_list = (
	        post_analysis_Z_matrix(my_lineages, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy,
	        matrix_splitting=False, first_absence_propagation=True, CNVs=CNVs, present_ssms=present_ssms, gain_num=gain_num,
		loss_num=loss_num))
	# check whether sum rule leads to unambiguous relationships
	logging.debug("using sum rule")
	zmcos = create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, [present_ssms], triplets_list)
	if len(zmcos) > 1:
		raise eo.MyException("zmcos should only have one entry!")
	zmco = zmcos[0]
	frequencies = np.asarray([my_lineages[i].freq for i in range(len(my_lineages))])
	try:
		dummy, avFreqs, ppm = sum_rule_algo_outer_loop(frequencies, zmco, seg_num, zero_count,
			gain_num, loss_num, CNVs, present_ssms)
	except eo.MyException as e:
		raise e

	# adapt lineages because SSMs can be phased differently after sum rule, also, relationships can differ
	logging.debug("adapt lineages after sum rule")
	my_lineages = create_updates_lineages(my_lineages, 0, [zmco.z_matrix], origin_present_ssms, [zmco.present_ssms])

	return my_lineages, zmco.z_matrix, avFreqs, ppm

# new sum rule algorithm (October 2019)
# processes each lineage and looks whether lineage has only one possible parent, this than becomes the
#   definite parent, updates follow
def sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms,
	noise_threshold=0):
	lin_num = len(zmco.z_matrix)
	# available frequencies are initialized
	avFreqs = copy.deepcopy(linFreqs)
	# get possible parents of all lineages
	ppm = get_possible_parents(zmco.z_matrix)
	# create array of definite parents
	defparent = [-1] * lin_num
	# create inital possible parent list, not based on frequencies and sum rule but on the other rules
	initial_pps_for_all = [-1] * lin_num
	
	# go over all lineages
	for k in range(1, lin_num):

		possible_parents = np.where(ppm[k] == 1)[0]
		initial_pps = possible_parents.copy()
		initial_pps_for_all[k] = initial_pps
		# iterate over all possible parents
		for k_star in possible_parents:
			# if k_star cannot be a possible parent
			if defparent[k] != k_star and np.greater(linFreqs[k], avFreqs[k_star]+cons.EPSILON_FREQUENCY+noise_threshold).any():
				ppm[k][k_star] = 0

				posdes = get_possible_descendants(zmco.z_matrix, k_star, k)
				# if no possible descendant of k_star is possible parent of k
				if sum([ppm[k][k_circ] for k_circ in posdes]) == 0:
					update_ancestry(-1, k_star, k, k, ppm, defparent, linFreqs, avFreqs, zmco, seg_num, zero_count, gain_num, 
						loss_num, CNVs, present_ssms, initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)

		possible_parents = np.where(ppm[k] == 1)[0]
		# if k has only one possible parent k_star
		if len(possible_parents) == 1 and defparent[k] == -1:
			make_def_child(possible_parents[0], k, k, ppm, defparent, linFreqs, avFreqs, zmco, seg_num, zero_count, gain_num, 
				loss_num, CNVs, present_ssms, initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
		elif len(possible_parents) < 1 and defparent[k] == -1:
			raise eo.NoParentsLeft("There are no possible parents for lineage {0}.".format(k))


	for k in range(2, lin_num):
		# if k doesnt't have definite parent yet
		if defparent[k] == -1:
			# ensure that k is descendant of definite ancetsors
			possible_parents = np.where(ppm[k] == 1)[0]
			lca = get_lca_from_multiple_lineages(possible_parents, zmco.z_matrix)
			update_ancestry(1, lca, k, lin_num-1, ppm, defparent, linFreqs, avFreqs, zmco, seg_num, zero_count, gain_num,
				loss_num, CNVs, present_ssms, noise_threshold=noise_threshold)
	
	return True, avFreqs, ppm

# given a list of initial possible parents, checks for lineage k whether these can still be possible parents except that available frequency is 
# too small
def still_possible_parents_except_freq(k, initial_pps_for_all, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms):

	still_pps = []

	# iterate through all initial possible parents
	for pp in initial_pps_for_all[k]:
		if zmco.z_matrix[pp][k] != 1:
			# copy zmco and present_ssms as these are not allowed to be changed
			copied_zmco = copy.deepcopy(zmco)
			copied_present_ssms= copy.deepcopy(present_ssms)
			# set relationship to ambiguous
			copied_zmco.z_matrix[pp][k] = 0
			# add new triplets because we have ambiguous entry again
			lin_num = len(copied_zmco.z_matrix)
			if lin_num > 3:
				for kcirc in range(lin_num):
					if kcirc < pp:
						update_triplet_hash(copied_zmco.triplet_xys, kcirc, pp, k)
						update_triplet_hash(copied_zmco.triplet_ysx, pp, k, kcirc)
						update_triplet_hash(copied_zmco.triplet_xsy, kcirc, k, pp)
					elif pp < kcirc < k:
						update_triplet_hash(copied_zmco.triplet_xys, pp, kcirc, k)
						update_triplet_hash(copied_zmco.triplet_ysx, kcirc, k, pp)
						update_triplet_hash(copied_zmco.triplet_xsy, pp, k, kcirc)
					elif k > kcirc:
						update_triplet_hash(copied_zmco.triplet_xys, pp, k, kcirc)
						update_triplet_hash(copied_zmco.triplet_ysx, k, kcirc, pp)
						update_triplet_hash(copied_zmco.triplet_xsy, pp, kcirc, k)
			# if updating relationship to present worked, lineage pp is still a potential parent except that it hasn't enough frequency
			try:
				update_ancestry(1, pp, k, zmco=copied_zmco, seg_num=seg_num, zero_count=zero_count, gain_num=gain_num, loss_num=loss_num,
					CNVs=CNVs, present_ssms=copied_present_ssms)
				still_pps.append(pp)
			except eo.MyException:
				pass
		else:
			still_pps.append(pp)

	initial_pps_for_all[k] = still_pps

def make_def_child(kstar, k, last, ppm, defparent, linFreqs, avFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms, initial_pps_for_all=None,
	noise_threshold=0):
	
	# available frequency gets adapted for all samples n
	avFreqs_tmp = np.subtract(avFreqs[kstar], linFreqs[k])
	
	# available frequency is not allowed to be smaller than 0
	if np.where(avFreqs_tmp + cons.EPSILON_FREQUENCY + noise_threshold < 0, True, False).any():
		still_possible_parents_except_freq(k, initial_pps_for_all, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)
		check_initial_possible_parents(k, linFreqs, avFreqs, initial_pps_for_all)

	avFreqs[kstar] = avFreqs_tmp
	
	defparent[k] = kstar
	
	# already processed k' which is possible child of kstar but not its definite child
	possible_children = get_possible_children_smaller_last(ppm, kstar, last, defparent)
	while len(possible_children) > 0:
		kprime = possible_children.pop(0)
	    
		# if k* cannot be possible parent of k'
		if np.greater(linFreqs[kprime], avFreqs[kstar] + cons.EPSILON_FREQUENCY + noise_threshold).any():
			ppm[kprime][kstar] = 0
			
			posdes = get_possible_descendants(zmco.z_matrix, kstar, kprime)
			# if no possible descendant of k* is possible parent of k'
			if sum([ppm[kprime][k_circ] for k_circ in posdes]) == 0:
				update_ancestry(-1, kstar, kprime, last, ppm, defparent, linFreqs, avFreqs, zmco, seg_num, zero_count, gain_num, loss_num, 
			    		CNVs, present_ssms, initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
			
			possible_parents = np.where(ppm[kprime] == 1)[0]
			# if k' has only one possible parent k_circ, which is not yet definite parent
			if len(possible_parents) == 1 and defparent[kprime] == -1:
		    		make_def_child(possible_parents[0], kprime, last, ppm, defparent, linFreqs, avFreqs, zmco, seg_num, zero_count, gain_num, 
					loss_num, CNVs, present_ssms, initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
		
		possible_children = get_possible_children_smaller_last_greater_kprime(ppm, kstar, kprime, last, defparent)
	
	update_ancestry(1, kstar, k, last, ppm, defparent, linFreqs, avFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms,
		initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
	
	return True


def compute_minimal_noise_threshold(k, linFreqs, avFreqs_from_initial_pps):
	assert isinstance(linFreqs, np.ndarray)

	# difference between available frequencies and frequency of lineage k
	diff_freq = avFreqs_from_initial_pps - linFreqs[k]
	# for each possible parent, the most negative value is taken
	max_diffs = np.amin(diff_freq, axis=1) * (-1)
	# minimal noise threshold is chosen
	minimal_noise_threshold = np.min(max_diffs)

	return minimal_noise_threshold

# given subclone k which cannot have any parents, return in exception the available frequency of its possible parents
# having not enough frequency
def check_initial_possible_parents(k, linFreqs, avFreqs, initial_pps_for_all):
	freq_string = "frequencies"
	if len(linFreqs[0]) == 1:
		freq_string = "frequency"

	output = "There are no possible parents for subclone {0} with {1} of {2}, because".format(k, freq_string,
		",".join(["{0:.3f}".format(cfreq) for cfreq in linFreqs[k]]))
	avFreqs_from_initial_pps = []
	if len(initial_pps_for_all[k]) > 0:
		for i in range(len(initial_pps_for_all[k])-1):
			current_pp = initial_pps_for_all[k][i]
			current_avFreqs = avFreqs[current_pp]
			avFreqs_from_initial_pps.append(current_avFreqs)
			output = output + " subclone {0} has only available {1} of {2},".format(current_pp, freq_string,
				",".join(["{0:.3f}".format(cfreq) for cfreq in current_avFreqs]))
		current_pp = initial_pps_for_all[k][-1]
		current_avFreqs = avFreqs[current_pp]
		avFreqs_from_initial_pps.append(current_avFreqs)
		output = output + " subclone {0} has only available {1} of {2}.".format(current_pp, freq_string,
			",".join(["{0:.3f}".format(cfreq) for cfreq in current_avFreqs]))
		raise eo.NoParentsLeftNoise(output, k, avFreqs_from_initial_pps)
	else:
		output = output + "no other subclone can be parent because of equivalence and lost allele constraints."
		raise eo.NoParentsLeft(output)

def update_ancestry(value, kstar, k, last=None, ppm=None, defparent=None, linFreqs=None, avFreqs=None, zmco=None, seg_num=None, 
	zero_count=None, gain_num=None, loss_num=None, CNVs=None, present_ssms=None, initial_pps_for_all=None, noise_threshold=0):

	# if relationship to normal lineage should be removed, there is no possible parent left
	if kstar == 0 and value == -1:
		still_possible_parents_except_freq(k, initial_pps_for_all, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)
		check_initial_possible_parents(k, linFreqs, avFreqs, initial_pps_for_all)
		
	if zmco.z_matrix[kstar][k] == value:
		return True

	if zmco.z_matrix[kstar][k] != 0:
		raise eo.RelationshipAlreadySet("Z[{0}][{1}] is already {2}, cannot be set to {3}.".format(kstar, k, zmco.z_matrix[kstar][k], value))

	zmco.z_matrix[kstar][k] = value

	# if ancestor-descendant relationship gets transformed to present one, the possible parent matrix needs to be updated because multiple parts could change
	if value == 1 and last is not None:
		update_possible_parents_per_child(zmco.z_matrix, ppm, kstar, k)

	if value == -1 and last is not None:
		ppm[k][kstar] = 0

	# if k was already processed and does not have a definite parent yet
	if last is not None and k < last and defparent[k] == -1:
		possible_parents = np.where(ppm[k] == 1)[0]
		# if k has only one possible parent k^\circ
		if len(possible_parents) == 1:
			make_def_child(possible_parents[0], k, last, ppm, defparent, linFreqs, avFreqs, zmco, seg_num, zero_count, gain_num,
				loss_num, CNVs, present_ssms, initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)

	# if ancestor-descendant relationship gets transformed to present one
	if value == 1 and zmco.present_ssms is not None:
		# move unphased SSMs if necessary
		try:
			phasing_allows_relation(kstar, k, zmco.matrix_after_first_round, zmco.present_ssms, CNVs, value)
		except eo.ADRelationNotPossible as e:
			raise e
		move_unphased_SSMs_if_necessary(kstar, k, zmco.present_ssms, CNVs, zmco.matrix_after_first_round, value)
		# propagte absence rules
		if last is not None:
			dummy_zero_count = len(zmco.z_matrix)*len(zmco.z_matrix)
			post_analysis_Z_matrix(None, seg_num, zmco.z_matrix, dummy_zero_count, zmco.triplet_xys, zmco.triplet_ysx, zmco.triplet_xsy,
				get_CNVs=False, matrix_splitting=False, check_crossing_rule=False, return_mutation_information=False,
				gain_num=gain_num, loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms, absence_propagation=True, path_lineages=None,
				last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, defparent=defparent, initial_pps_for_all=initial_pps_for_all,
				noise_threshold=noise_threshold)

	# if ancestor-descendant relationship gets transformed to absent one
	if value == -1 and last is not None:
		possible_ancestors = get_possible_ancestors(zmco.z_matrix, k, kstar)
		for kcirc in possible_ancestors:
			possible_descendants = get_possible_descendants(zmco.z_matrix, kcirc, k)
			# if possible ancestor k^\circ of k* has no possible descendant that is possible parent of k and is not possible parent itself
			if sum([ppm[k][kbullet] for kbullet in possible_descendants]) == 0 and ppm[k][kcirc] == 0:
				update_ancestry(-1, kcirc, k, last, ppm, defparent, linFreqs, avFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms,
					initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)

        # transitivity update with parent checking
	dummy_zero_count = len(zmco.z_matrix) * len(zmco.z_matrix)
	try:
		update_Z_matrix_iteratively(zmco.z_matrix, dummy_zero_count, zmco.triplet_xys, zmco.triplet_ysx, zmco.triplet_xsy,
			(kstar, k), zmco.present_ssms, CNVs, zmco.matrix_after_first_round,
			last, ppm, avFreqs, linFreqs, zmco, seg_num, gain_num, loss_num, defparent, initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
	except eo.MyException as e:
		raise e

# returns all possible children of lineage kstar that have lower index than last (already processed) that are not its definite children
def get_possible_children_smaller_last(ppm, kstar, last, defparent):
	#return [kprime for kprime in np.where(ppm[:last,kstar] == 1)[0].tolist() if defparent[kprime] == -1]
	return get_possible_children_smaller_last_greater_kprime(ppm, kstar, -1, last, defparent)

# returns all possible children of lineage kstar that have lower index than last (already processed) and higher index than kprime and that are not its definite children
def get_possible_children_smaller_last_greater_kprime(ppm, kstar, kprime, last, defparent):
	return [k_dprime for k_dprime in np.add(np.where(ppm[kprime+1:last,kstar] == 1)[0], kprime+1).tolist() if defparent[k_dprime] == -1]

# returns all possible descendants of lineage k that have lower index than lineage kprime
def get_possible_descendants(zmatrix, k, kprime):
	return np.add(np.where(np.asarray(zmatrix[k][k:kprime]) != -1), k)[0]

# returns all possible ancestors of kstar that are also possible ancestors of k
def get_possible_ancestors(zmatrix, k, kstar):
	return np.intersect1d(np.where(np.asarray(zmatrix)[:kstar,kstar] != -1), np.where(np.asarray(zmatrix)[:kstar, k] != -1))

# given a Z-matrix, function computes possible parents for each non-normal lineage
# matrix with possible parents has following format:
#   ppm[k'][k] = 1 if lineage k is possible parent of lineage k', 0 otherwise
def get_possible_parents(z_matrix):
	lin_num = len(z_matrix)
	# create empty matrix
	ppm = np.zeros((lin_num, lin_num))

	# go through Z-matrix
	for kprime in range(1, lin_num):
            get_possible_parents_per_child(z_matrix, ppm, kprime)

	return ppm
    
# helping function of get_possible_parents
def get_possible_parents_per_child(z_matrix, ppm, kprime):
	for k in range(kprime-1, -1, -1):
		# if both lineages are in an umbiguous relationship, k is a potential parent
		if z_matrix[k][kprime] == 0:
			ppm[kprime][k] = 1
		# if k is the ancestor with the lowest index of k', it's a potential parent
		elif z_matrix[k][kprime] == 1:
			ppm[kprime][k] = 1
			# no other potential parent for k' exists
			return True

# if k became a new ancestor of k', lineages with k^\circ < k cannot be possible parents of k' anymore
def update_possible_parents_per_child(z_matrix, ppm, k, kprime):
	for kcirc in range(k-1, -1, -1):
		ppm[kprime][kcirc] = 0

# returns True if lineage k* has (potential) descendants that are still in possible parent-child relationship with lineage k
# returns False otherwise
def des_are_potential_parents(k_star, k, zmatrix, ppm):
    lin_num = len(zmatrix)
    # get all (potential) descendants of lineage k*
    descendats = [des for des in range(k_star, k) if (zmatrix[k_star][des] == 1 or zmatrix[k_star][des] == 0)]

    for des in descendats:
        # check whether descendant is in a possible parent-child relationship
        if ppm[k][des] == 1:
            return True

    return False

# given the possible parent matrix, creates a mapping for each subclone
def get_possible_parents_from_ppmatrix(ppm):
	pp_mapping = {}
	for i in range(1,len(ppm)):
		possible_parents = np.where(ppm[i] == 1)[0]
		pp_mapping[i] = possible_parents

	return pp_mapping

def create_Z_Matrix_Co_objects(z_matrix_list, z_matrix_fst_rnd, present_ssms_list, triplets_list):
	zmcos = []
	for i in range(len(z_matrix_list)):
		zmco = Z_Matrix_Co(z_matrix=z_matrix_list[i], triplet_xys=triplets_list[i][0], 
			triplet_ysx=triplets_list[i][1], triplet_xsy=triplets_list[i][2], 
			present_ssms=present_ssms_list[i], matrix_after_first_round=z_matrix_fst_rnd)
		zmcos.append(zmco)
	return zmcos

# gets all children of a lineage k given the Z-matrix
def get_children(z_matrix, k):
	lin_num = len(z_matrix)
	children = []

	# check all potential descendant of k
	for k_prime in range(k+1, lin_num):
		# k isn't an ancestor of k'
		if z_matrix[k][k_prime] != 1:
			continue
		
		k_prime_potential_child = True
		# check all lineages that could be between k and k'
		for k_star in range(k+1, k_prime):
			# k' has another ancestor k*>k, thus k can't be the parent
			if z_matrix[k_star][k_prime] == 1:
				k_prime_potential_child = False
				break

		# no other ancestor of k' was found, thus k is the parent
		if k_prime_potential_child == True:
			children.append(k_prime)

	return children


# counts the number of 0's in the Z-matrix
def get_0_number_in_z_matrix(z_matrix):
	zero_count = 0
	for row in z_matrix:
		for entry in row:
			if entry == 0:
				zero_count += 1
	return zero_count

def check_and_update_complete_Z_matrix_from_matrix(z_matrix, zero_count, lineage_num, CNVs=None, present_ssms=None,
	z_matrix_after_CN_influence_check=None):
	# 3 hashed to store triplets that contain 0s
	triplet_xys = {}
	triplet_ysx = {}
	triplet_xsy = {}

	# iterate through Z-matrix
	for x in range(lineage_num-3, 0, -1):
		for y in range(lineage_num-2, x, -1):
			for s in range(lineage_num-1, y, -1):
				# update Z triplet
				changed, changed_field, triplet_zeros, v_x, v_y, v_s = (update_Z_triplet(
					z_matrix[x][y], z_matrix[y][s], z_matrix[x][s]))
				# triplet contains at least one 0
				if triplet_zeros > 0:
					# add triplets to hash
					update_triplet_hash(triplet_xys, x, y, s)
					update_triplet_hash(triplet_ysx, y, s, x)
					update_triplet_hash(triplet_xsy, x, s, y)
				# triplet was changed
				if changed == True:
					zero_count = update_after_tiplet_change(z_matrix, zero_count, changed_field, 
						v_x, v_y, v_s, x, y, s, triplet_xys, triplet_ysx, triplet_xsy,
						CNVs=CNVs, present_ssms=present_ssms, 
						matrix_after_first_round=z_matrix_after_CN_influence_check)

	return zero_count, triplet_xys, triplet_ysx, triplet_xsy


# when a triplet was changed, things get updated:
#	zero count is decreased, Z-matrix is updated, it is checked, whether triplets which contain the
#	changed field can be updated as well
# z_matrix: the Z-matrix
# zero_count: number of 0 entires in the Z-matrix
# changed_field: the field in the triplet that was changed
# v_x, v_y, v_s: values of triplet entries
# x, y, s: indices of triplet entries
# triplet_xys, triplet_ysx, triplet_xsy: the three hashes with the triplets that contain a 0
# present_ssms: 3D list with booleans: [segment][A,B,unphased][lineage]
# CNVs: list with hash with information in which segments appear which CN changes in which lineages
# matrix_after_first_round: Z-matrix after the first round
# can raise exception ZInconsistence when updating of the Z_triplet let to inconsistance
def update_after_tiplet_change(z_matrix, zero_count, changed_field, v_x, v_y, v_s, x, y, s,
	triplet_xys, triplet_ysx, triplet_xsy, present_ssms=None, CNVs=None, matrix_after_first_round=None,
        last=None, ppm=None, avFreqs=None, linFreqs=None, zmco=None, seg_num=None, gain_num=None, loss_num=None,
	defparent=None, initial_pps_for_all=None, noise_threshold=0):
	# update number of 0's
	# each change decreases the number of 0's by one
	zero_count -= 1

	# get indices of pair that was influenced by change
	if changed_field == cons.X:
		index_pair = (x, y)
		new_z_entry = v_x
	elif changed_field == cons.Y:
		index_pair = (y, s)
		new_z_entry = v_y
	else:
		index_pair = (x, s)
		new_z_entry = v_s

	# not called from sum rule algorithm
	if last is None:
		# update entry in Z-matrix
		z_matrix[index_pair[0]][index_pair[1]] = new_z_entry

		# if phasing should and has to be considered, consider it
		if present_ssms is not None and new_z_entry == 1:
			i = index_pair[0]
			i_prime = index_pair[1]
			phasing_allows_relation(i, i_prime, matrix_after_first_round, present_ssms, CNVs, new_z_entry)
			move_unphased_SSMs_if_necessary(i, i_prime, present_ssms, CNVs, matrix_after_first_round, new_z_entry)

        	## if function is called from new sum rule algorithm
        	#if last is not None:
        	#    k = index_pair[0]
        	#    kprime = index_pair[1]
        	#    # lineages were put into relationship
        	#    if new_z_entry == 1:
        	#        # if lineage k is possible parent
        	#        if ppm[kprime][k] == 1:
        	#            # update ppm for  lineage k'
		#	    for my_i in range(k):
		#	    	ppm[kprime][my_i] = 0
        	#    # lineages are not in ADR
        	#    elif new_z_entry == -1:
        	#        # if lineage k was possible parent
        	#        if ppm[kprime][k] == 1:
        	#            # remove possible parent
        	#            ppm[kprime][k] = 0
		#        # if relationship is transformed into an absent one, lineage k is not a possible descendant of lineage k anymore
		#        # thus, posdes_parents might need updates
		#       	remove_possible_descendat_parent_if_necessary_because_rel_absence(k, kprime, zmco, last, 
		#       		ppm, avFreqs, linFreqs, seg_num,
		#       		gain_num, loss_num, CNVs, present_ssms, definite_children, posdes_parents)

        	#    # if  lineage k' was processed already or is currently being processed
        	#    if kprime <= last:
        	#        possible_parents = np.where(ppm[kprime] == 1)[0]
        	#        # lineages has only one possible parent left and is not yet its definite child
        	#        if len(possible_parents) == 1 and kprime not in definite_children[possible_parents[0]]:
		#	    try:
        	#            	update_avFreq_parents(possible_parents[0], kprime, last, ppm, avFreqs, linFreqs, zmco, seg_num, zero_count,
		#	    		gain_num, loss_num, CNVs, present_ssms, definite_children, posdes_parents) 
		#	    except eo.MyException as e:
		#	    	raise e

		# see whether previous triplets need to be updated
		try:
			zero_count = update_Z_matrix_iteratively(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy, 
				index_pair, present_ssms, CNVs, matrix_after_first_round, last, ppm, avFreqs, linFreqs, zmco, seg_num,
				gain_num, loss_num, defparent, initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
		except eo.ZInconsistence as e:
			raise e

		## propagate absence constraints if current relationship was set to present and function was called from sum rule
		#if new_z_entry == 1 and last is not None:
		#	post_analysis_Z_matrix(None, seg_num, zmco.z_matrix, zero_count, zmco.triplet_xys, zmco.triplet_ysx, zmco.triplet_xsy,
		#		get_CNVs=False, matrix_splitting=False, check_crossing_rule=False, return_mutation_information=False,
		#		gain_num=gain_num, loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms, absence_propagation=True, path_lineages=None,
		#		last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, defparent=defparent)

	# is called from sum rule algorithm
	else:
		update_ancestry(new_z_entry, index_pair[0], index_pair[1], last, ppm, defparent, linFreqs, avFreqs, zmco, seg_num, zero_count, 
			gain_num, loss_num, CNVs, present_ssms, initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)

	return zero_count
	

# checks if triplets in which the changed entry is involved also need to be updated
# checks this iteratively by checking also all triplets that are influenced by the first change
# z_matrix: the Z-matrix
# zero_count: number of 0 entires in the Z-matrix
# triplet_xys, triplet_ysx, triplet_xsy: the three hashes with the triplets that contain a 0
# index_pair: indices of the element in the triplet that was changed
# present_ssms: 3D list with booleans: [segment][A,B,unphased][lineage]
# CNVs: list with hash with information in which segments appear which CN changes in which lineages
# matrix_after_first_round: Z-matrix after the first round
## for new sum rule algorithm
# last: last lineage index that was checked in outer loop
# ppm: possible parent matrix
# avFreqs: available frequencies
# linFreqs: lineage frequencies
# can raise exception ZInconsistence when updating of the Z_triplet let to inconsistance
# can raise exception ADRelationNotPossible when SSM phasing prevents update of relationships
def update_Z_matrix_iteratively(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy, index_pair,
	present_ssms=None, CNVs=None, matrix_after_first_round=None, last=None, ppm=None, avFreqs=None, 
        linFreqs=None, zmco=None, seg_num=None, gain_num=None, loss_num=None, defparent=None, initial_pps_for_all=None,
	noise_threshold=0):
	first_index = index_pair[0]
	second_index = index_pair[1]
	# check whether lowest triplets are influenced by change
	try:
		# for all triplets, in which the changed value is at the xy position
		for s in sorted(list(triplet_xys[first_index][second_index].keys()), reverse=True):
			# update Z triplet
			changed, changed_field, triplet_zeros, v_x, v_y, v_s = (update_Z_triplet(
				z_matrix[first_index][second_index], z_matrix[second_index][s], z_matrix[first_index][s]))
			# if triplet doesn't contain 0's anymore, remove triplet from all hashs
			if triplet_zeros == 0:
				remove_triplet_from_all_hashes(triplet_xys, triplet_ysx, triplet_xsy, x=first_index,
					y=second_index, s=s)
			# triplet was changed
			if changed == True:
				# phasing should be considered
				if present_ssms is not  None:
					if changed_field == cons.X:
						i = first_index
						i_prime = second_index
						value = v_x
					elif changed_field == cons.Y:
						i = second_index
						i_prime = s
						value = v_y
					elif changed_field == cons.S:
						i = first_index
						i_prime = s
						value = v_s
					# updated entry is 1, phasing has to be considered
					if value == 1:
						phasing_allows_relation(i, i_prime, matrix_after_first_round, 
							present_ssms, CNVs, value)
						move_unphased_SSMs_if_necessary(i, i_prime, present_ssms, 
							CNVs, matrix_after_first_round, value)
				# update matrix
				zero_count = update_after_tiplet_change(z_matrix, zero_count, changed_field,
					v_x, v_y, v_s, x=first_index, y=second_index, s=s, 
					triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy,
                                        last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, seg_num=seg_num,
					present_ssms=present_ssms, CNVs=CNVs, gain_num=gain_num, loss_num=loss_num,
					matrix_after_first_round=matrix_after_first_round, defparent=defparent,
					initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)

	# no triplet was found
	except KeyError:
		pass
	except eo.ZInconsistence as e:
		raise e

	# check whether middle triplets are influenced by change
	try:
		# for all triplets, in which the changed value is at the xs position
		for y in sorted(list(triplet_xsy[first_index][second_index].keys()), reverse=True):
			# update Z triplet
			changed, changed_field, triplet_zeros, v_x, v_y, v_s = (update_Z_triplet(
				z_matrix[first_index][y], z_matrix[y][second_index], z_matrix[first_index][second_index]))
			# if triplet doesn't contain 0's anymore, remove triplet from all hashs
			if triplet_zeros == 0:
				remove_triplet_from_all_hashes(triplet_xys, triplet_ysx, triplet_xsy, x=first_index,
					y=y, s=second_index)
			# triplet was changed
			if changed == True:
				# phasing should be considered
				if present_ssms is not None:
					if changed_field == cons.X:
						i = first_index
						i_prime = y
						value = v_x
					elif changed_field == cons.Y:
						i = y
						i_prime = second_index
						value = v_y
					elif changed_field == cons.S:
						i = first_index
						i_prime = second_index
						value = s
					# updated entry is 1, phasing has to be considered
					if value == 1:
						phasing_allows_relation(i, i_prime, matrix_after_first_round, 
							present_ssms, CNVs, value)
						move_unphased_SSMs_if_necessary(i, i_prime, present_ssms, 
							CNVs, matrix_after_first_round, value)
				# update matrix
				zero_count = update_after_tiplet_change(z_matrix, zero_count, changed_field,
					v_x, v_y, v_s, x=first_index, y=y, s=second_index, 
					triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy,
                                        last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, seg_num=seg_num,
					present_ssms=present_ssms, CNVs=CNVs, gain_num=gain_num, loss_num=loss_num,
					matrix_after_first_round=matrix_after_first_round, defparent=defparent,
					initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
	# no triplet was found
	except KeyError:
		pass
	except eo.ZInconsistence as e:
		raise e

	# check whether highest triplets are influenced by change
	try:
		# for all triplets, in which the changed value is at the ys position
		for x in sorted(list(triplet_ysx[first_index][second_index].keys()), reverse=True):
			# update Z triplet
			changed, changed_field, triplet_zeros, v_x, v_y, v_s = (update_Z_triplet(
				z_matrix[x][first_index], z_matrix[first_index][second_index], z_matrix[x][second_index]))
			# if triplet doesn't contain 0's anymore, remove triplet from all hashs
			if triplet_zeros == 0:
				remove_triplet_from_all_hashes(triplet_xys, triplet_ysx, triplet_xsy, x=x,
					y=first_index, s=second_index)
			# triplet was changed
			if changed == True:
				# phasing should be considered
				if present_ssms is not None:
					if changed_field == cons.X:
						i = x
						i_prime = first_index
						value = v_x
					if changed_field == cons.Y:
						i = first_index
						i_prime = second_index
						value = v_y
					if changed_field == cons.S:
						i = x
						i_prime = second_index
						value = v_s
					# updated entry is 1, phasing has to be considered
					if value == 1:
						phasing_allows_relation(i, i_prime, matrix_after_first_round, 
							present_ssms, CNVs, value)
						move_unphased_SSMs_if_necessary(i, i_prime, present_ssms, 
							CNVs, matrix_after_first_round, value)
				# update matrix
				zero_count = update_after_tiplet_change(z_matrix, zero_count, changed_field,
					v_x, v_y, v_s, x=x, y=first_index, s=second_index, 
					triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy,
                                        last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, seg_num=seg_num,
					present_ssms=present_ssms, CNVs=CNVs, gain_num=gain_num, loss_num=loss_num,
					matrix_after_first_round=matrix_after_first_round, defparent=defparent,
					initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
	# no triplet was found
	except KeyError:
		pass
	except eo.ZInconsistence as e:
		raise e

	return zero_count


# removes a triplet entry from all three hashes
def remove_triplet_from_all_hashes(triplet_xys, triplet_ysx, triplet_xsy, x, y, s):
	remove_triplet_from_hash(triplet_xys, x, y, s)
	remove_triplet_from_hash(triplet_ysx, y, s, x)
	remove_triplet_from_hash(triplet_xsy, x, s, y)

# removes an triplet entry from the hash
def remove_triplet_from_hash(t_hash, i_1, i_2, i_3):
	# removes triplet
	try:
		del t_hash[i_1][i_2][i_3]
		# if first two indices aren't part of another triplet, delete hash for second index
		if len(list(t_hash[i_1][i_2].keys())) == 0:
			del t_hash[i_1][i_2]
			# if first index isn't part of another triplet, delete it from hash
			if len(list(t_hash[i_1].keys())) == 0:
				del t_hash[i_1]
	# triplet was already removed
	except KeyError:
		pass

# updates the given hash
# i_1 - i_3 are the indices of the lineages
# lineage indices are given in the order of the hash name
# hash contains an entry, if triplet contains at least one 0
def update_triplet_hash(t_hash, i_1, i_2, i_3):
	try:
		t_hash[i_1][i_2][i_3] = True
	except KeyError:
		try:
			t_hash[i_1][i_2] = {}
			t_hash[i_1][i_2][i_3] = True
		except KeyError:
			t_hash[i_1] = {}
			t_hash[i_1][i_2] = {}
			t_hash[i_1][i_2][i_3] = True


# x, y and s define the values of the corresponding fields in a triplet of the Z-matrix
# Z[x][y] = value x
# Z[y][s] = value y
# Z[x][s] = value s
# returns if something was changed, which field was changed, how many 0 are still in triplet
#	and the values of the three fields
# can raise Exception: ZInconsistence
def update_Z_triplet(x, y, s):
	if x == 0:
		if y == 0:
			# case 7
			if s == 0:
				return False, "", 3, x, y, s
			# case 6b
			elif s == 1:
				return False, "", 2, x, y, s
			# case 8b
			elif s == -1:
				return False, "", 2, x, y, s
		elif y == 1:
			# case 6c
			if s == 0:
				return False, "", 2, x, y, s
			# case 4a
			elif s == 1:
				return True, cons.X, 0, 1, y, s
			# case 10b
			elif s == -1:
				return True, cons.X, 0, -1, y, s
		elif y == -1:
			# case 8c
			if s == 0:
				return False, "", 2, x, y, s
			# case 10a
			elif s == 1:
				return False, "", 1, x, y, s
			# case 9a
			elif s == -1:
				return False, "", 1, x, y, s
	elif x == 1:
		if y == 0:
			# case 6a
			if s == 0:
				return False, "", 2, x, y, s
			# case 4c
			elif s == 1:
				return False, "", 1, x, y, s
			# case 10e
			elif s == -1:
				return True, cons.Y, 0, x, -1, s
		elif y == 1:
			# case 4b
			if s == 0:
				return True, cons.S, 0, x, y, 1
			# case 1
			elif s == 1:
				return False, "", 0, x, y, s
			# case 2b
			elif s == -1:
				raise eo.ZInconsistence("Inconsistent case: x=1, y=1, s=-1.")
		elif y == -1:
			# case 10c
			if s == 0:
				return False, "", 1, x, y, s
			# case 2c
			elif s == 1:
				return False, "", 0, x, y, s
			# case 3a
			elif s == -1:
				return False, "", 0, x, y, s
	elif x == -1:
		if y == 0:
			# case 8a
			if s == 0:
				return False, "", 2, x, y, s
			# case 10f
			elif s == 1:
				return True, cons.Y, 0, x, -1, s
			# case 9c
			elif s == -1:
				return False, "", 1, x, y, s
		elif y == 1:
			# case 10d
			if s == 0:
				return True, cons.S, 0, x, y, -1
			# case 2a
			elif s == 1:
				raise eo.ZInconsistence("Inconsistent case: x=-1, y=1, s=1.")
			# case 3c
			elif s == -1:
				return False, "", 0, x, y, s
		elif y == -1:
			# case 9b
			if s == 0:
				return False, "", 1, x, y, s
			# case 3b
			elif s == 1:
				return False, "", 0, x, y, s
			# case 5
			elif s == -1:
				return False, "", 0, x, y, s

# checks the crossing rule and adapts the Z-matrix if necessary
def check_crossing_rule_function(my_lineages, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy):
    lin_num = len(my_lineages)

    for k in range(1, lin_num):
        for k_prime in range(k+1, lin_num):
            no_violation = (np.asarray(my_lineages[k].freq) >= np.asarray(my_lineages[k_prime].freq)).all()
            if no_violation == False:
                # make relationship absent
                if z_matrix[k][k_prime] == 1:
                    raise eo.MyException("Subclone {0} cannot be an ancestor of subclone {1} because of crossing rule.".format(k, k_prime))
                z_matrix[k][k_prime] = -1
                zero_count -= 1
                # check whether Z-matrix needs to be updated iteratively
                zero_count = update_Z_matrix_iteratively(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy, 
                        (k, k_prime))

    return zero_count


# given the lineages of the results, the encoded Z matrix is refined
# values that need to be 0 and values that are 0s but can be 1s are derived
# my_lineages: list with lineages
# seg_num: number of segments
# z_matrix: Z-matrix
# zero_count: number of 0's in matrix
# triplet_xys, triplet_ysx, triplet_xsy: the three hashes with the triplets that contain a 0
# matrix_splitting: whether hard cases should be processed that might lead to splitting/forking of Z-matrix
# return_mutation_information: returns the collected information about mutations per segments and lineages
# gain_num: shows in which segments which lineages have gains
# loss_num: shows in which segments which lineages have losses
# CNVs: shows in which segments which lineages have which CNVs
# present_ssms: shows in which segments which lineages have which SSMs
# absence_propagation: whether this function is called only to propagate further absence relationships
# path_lineages: not clear yet whether I need this
# last: current last lineage processed in new sum rule algorithm (June 2019)
def post_analysis_Z_matrix(my_lineages, seg_num, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy,
	get_CNVs=False, matrix_splitting=True, check_crossing_rule=True, return_mutation_information=False,
	gain_num=[], loss_num=[], CNVs=[], present_ssms=[], absence_propagation=False, path_lineages=None, last=None,
	ppm=None, avFreqs=None, linFreqs=None, zmco=None, defparent=None, first_absence_propagation=False, 
	initial_pps_for_all=None, noise_threshold=0):

	# if function is called for propagation of relationship absence, these things were already done before
	if absence_propagation == False and first_absence_propagation == False:

		lineage_num = len(my_lineages)

		# go once through segment and get gains, losses and SSMs
		gain_num = []
		loss_num = []
		CNVs = []
		present_ssms = []
		ssm_infl_cnv_same_lineage = []

		# iterate through all segments once to get all CN changes and SSMs appearances
		get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
			ssm_infl_cnv_same_lineage)

		# copy present_ssm list for later
		origin_present_ssms = copy.deepcopy(present_ssms)

		# check SSM phasing as it can be unnecessary because ADRs were removed before
		change_unnecessary_phasing(len(my_lineages), CNVs, present_ssms, ssm_infl_cnv_same_lineage, z_matrix, seg_num)

		# check crossing rule
		if check_crossing_rule and not (isinstance(my_lineages[0].freq, float) or isinstance(my_lineages[0].freq, int)):
			zero_count = check_crossing_rule_function(my_lineages, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy)

	# iterate through all segments the first time and
	# only check the simple cases, that are directly to decide in the first run
	for seg_index in range(seg_num):

		# CN-free segment, check next segment
		if gain_num[seg_index] == 0 and loss_num[seg_index] == 0:
			continue

		# check now different combinations of CNVs together with SSMs
		zero_count = check_1a_CN_LOSS(loss_num[seg_index], CNVs[seg_index], z_matrix, zero_count, 
			triplet_xys, triplet_ysx, triplet_xsy, path_lineages, seg_num=seg_num, last=last,
			ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, loss_num=loss_num, 
			CNVs=CNVs, present_ssms=present_ssms, defparent=defparent, initial_pps_for_all=initial_pps_for_all, 
			noise_threshold=noise_threshold)
		if zero_count == 0:
			break
		zero_count = check_1c_CN_loss(loss_num[seg_index], CNVs[seg_index], z_matrix, zero_count, 
			present_ssms[seg_index], triplet_xys, triplet_ysx, triplet_xsy, path_lineages, seg_num=seg_num, last=last,
			ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, loss_num=loss_num, 
			CNVs=CNVs, present_ssms=present_ssms, defparent=defparent, initial_pps_for_all=initial_pps_for_all,
			noise_threshold=noise_threshold)
		if zero_count == 0:
			break
		zero_count = check_1d_2c_CN_losses(loss_num[seg_index], CNVs[seg_index], z_matrix, zero_count, 
			present_ssms[seg_index], triplet_xys, triplet_ysx, triplet_xsy, path_lineages, seg_num=seg_num, last=last,
			ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, loss_num=loss_num, 
			CNVs=CNVs, present_ssms=present_ssms, defparent=defparent, initial_pps_for_all=initial_pps_for_all,
			noise_threshold=noise_threshold)
		if zero_count == 0:
			break
		zero_count = check_2f_CN_gains(gain_num[seg_index], CNVs[seg_index], z_matrix, zero_count, 
			present_ssms[seg_index], triplet_xys, triplet_ysx, triplet_xsy, path_lineages, seg_num=seg_num, last=last,
			ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, loss_num=loss_num, 
			CNVs=CNVs, present_ssms=present_ssms, defparent=defparent, initial_pps_for_all=initial_pps_for_all,
			noise_threshold=noise_threshold)
		if zero_count == 0:
			break
		zero_count = check_2h_LOH(loss_num[seg_index], gain_num[seg_index], CNVs[seg_index], z_matrix, 
			zero_count, present_ssms[seg_index], triplet_xys, triplet_ysx, triplet_xsy, path_lineages, seg_num=seg_num,
			last=last,
			ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, loss_num=loss_num, 
			CNVs=CNVs, present_ssms=present_ssms, defparent=defparent, initial_pps_for_all=initial_pps_for_all,
			noise_threshold=noise_threshold)
		if zero_count == 0:
			break
		zero_count = check_2i_phased_changes(CNVs[seg_index], z_matrix, zero_count, present_ssms[seg_index],
			triplet_xys, triplet_ysx, triplet_xsy, path_lineages, seg_num=seg_num, last=last,
			ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, loss_num=loss_num, 
			CNVs=CNVs, present_ssms=present_ssms, defparent=defparent, initial_pps_for_all=initial_pps_for_all,
			noise_threshold=noise_threshold)
		if zero_count == 0:
			break
		zero_count = check_1f_2d_2g_2j_losses_gains(loss_num[seg_index], CNVs[seg_index], z_matrix, zero_count, 
			present_ssms[seg_index], triplet_xys, triplet_ysx, triplet_xsy,
			first_run=True, mutations=cons.LOSS, path_lineages=path_lineages, seg_num=seg_num,
			last=last,
			ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, loss_num=loss_num, 
			CNVs=CNVs, present_ssms=present_ssms, defparent=defparent, initial_pps_for_all=initial_pps_for_all,
			noise_threshold=noise_threshold)
		if zero_count == 0:
			break
		zero_count = check_1f_2d_2g_2j_losses_gains(gain_num[seg_index], CNVs[seg_index], z_matrix, zero_count, 
			present_ssms[seg_index], triplet_xys, triplet_ysx, triplet_xsy,
			first_run=True, mutations=cons.GAIN, path_lineages=path_lineages, seg_num=seg_num, last=last,
			ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, loss_num=loss_num, 
			CNVs=CNVs, present_ssms=present_ssms, defparent=defparent, initial_pps_for_all=initial_pps_for_all,
			noise_threshold=noise_threshold)
		if zero_count == 0:
			break
		loss_A_num = 0
		try:
			loss_A_num = len(list(CNVs[seg_index][cons.LOSS][cons.A].keys()))
		except KeyError:
			pass
		gain_B_num = 0
		try:
			gain_B_num = len(list(CNVs[seg_index][cons.GAIN][cons.B].keys()))
		except KeyError:
			pass
		zero_count = check_1f_2d_2g_2j_losses_gains(loss_A_num, CNVs[seg_index], z_matrix, zero_count, 
			present_ssms[seg_index], triplet_xys, triplet_ysx, triplet_xsy,
			first_run=True, mutations=cons.LOSS, path_lineages=path_lineages, seg_num=seg_num,
			last=last,
			ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, loss_num=loss_num, 
			CNVs=CNVs, present_ssms=present_ssms, defparent=defparent, mut_num_B=gain_B_num,
			mutations_B=cons.GAIN, initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
		if zero_count == 0:
			break
		loss_B_num = 0
		try:
			loss_B_num = len(list(CNVs[seg_index][cons.LOSS][cons.B].keys()))
		except KeyError:
			pass
		gain_A_num = 0
		try:
			gain_A_num = len(list(CNVs[seg_index][cons.GAIN][cons.A].keys()))
		except KeyError:
			pass
		if zero_count == 0:
			break
		zero_count = check_1f_2d_2g_2j_losses_gains(gain_A_num, CNVs[seg_index], z_matrix, zero_count, 
			present_ssms[seg_index], triplet_xys, triplet_ysx, triplet_xsy,
			first_run=True, mutations=cons.GAIN, path_lineages=path_lineages, seg_num=seg_num,
			last=last,
			ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, loss_num=loss_num, 
			CNVs=CNVs, present_ssms=present_ssms, defparent=defparent, mut_num_B=loss_B_num,
			mutations_B=cons.LOSS, initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)

	# iterate through all segments a second time and check the hard cases now
	# that are not easy to resolve and might result in multiple Z-matrices
	# matrix after first round of analysis
	z_matrix_fst_rnd = copy.deepcopy(z_matrix) 
	# lists for Z matrix and other variables are created
	z_matrix_list = [np.asarray(z_matrix)]
	triplets_list = [[triplet_xys, triplet_ysx, triplet_xsy]]
	present_ssms_list = [present_ssms]
	if matrix_splitting == True:
		if path_lineages is not None:
			raise eo.MyException("post_analysis_Z_matrix: matrix_splitting == True and path_lineages != None are not covered.")
		for seg_index in range(seg_num):
			check_1f_2d_2g_2j_losses_gains(loss_num[seg_index], CNVs[seg_index], None, None, 
				None,  None, None, None,
				first_run=False, mutations=cons.LOSS, z_matrix_fst_rnd=z_matrix_fst_rnd,
				z_matrix_list=z_matrix_list, triplets_list=triplets_list, present_ssms_list=present_ssms_list,
				seg_index=seg_index, CNVs_all=CNVs)
			check_1f_2d_2g_2j_losses_gains(gain_num[seg_index], CNVs[seg_index], None, None, 
				None, None, None, None,
				first_run=False, mutations=cons.GAIN, z_matrix_fst_rnd=z_matrix_fst_rnd,
				z_matrix_list=z_matrix_list, triplets_list=triplets_list, present_ssms_list=present_ssms_list,
				seg_index=seg_index, CNVs_all=CNVs)
	
	if return_mutation_information == True:
		return (z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list, gain_num, loss_num)
	if first_absence_propagation == True:
		return z_matrix_list, z_matrix_fst_rnd, triplets_list
	elif absence_propagation == True and matrix_splitting == False:
		return z_matrix_list, present_ssms_list[0], triplets_list
	elif absence_propagation == True and matrix_splitting == True:
		raise eo.MyException("post_analysis_Z_matrix: absence_propagation == True and matrix_splitting == True not covered")
	elif get_CNVs == False:
		return z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list
	else:
		return z_matrix_list, z_matrix_fst_rnd, origin_present_ssms, present_ssms_list, CNVs, triplets_list

# if lineages are based on ground truth datasets and were not reconstructed, SSM phasing needs
# to be checked
# CNVs: list with CN changes
# present_ssms: lists with phasing of SSMs
# ssm_infl_cnv_same_lineage: list storing whether SSMs are influenced by CN gains in same lineage
def change_unnecessary_phasing(lin_num, CNVs, present_ssms, ssm_infl_cnv_same_lineage, z_matrix, seg_num, user_ssm=None):
	# check all cancerous lineages
	for k in range(1, lin_num):
		# get descendants
		descendants = [k_prime for k_prime in range(k+1, lin_num) if z_matrix[k][k_prime] == 1]
		# get ancestors, normal lineage does not have to be considered here
		ancestors = [k_star for k_star in range(1, k) if z_matrix[k_star][k] == 1]
		
		# check all segments
		for seg_index in range(seg_num):

			# get CN changes of segment
			CNV_seg = CNVs[seg_index]

			keep_phased_A = False
			keep_phased_B = False

			# only if lineage k has phased SSMs in segment i, the phasing needs to be checked
			if present_ssms[seg_index][cons.A][k] == False and present_ssms[seg_index][cons.B][k] == False:
				continue

			# if ancestors or k have CN loss, phasing is kept
			if do_ancestors_have_CN_loss(ancestors + [k], CNV_seg) == True:
				keep_phased_A = True
				keep_phased_B = True
			# if descendants have CN change, phasing is kept
			elif do_descendants_have_CN_change(descendants, CNV_seg) == True:
				keep_phased_A = True
				keep_phased_B = True
			# if current lineage has CN gains and influenced SSMs, phasing is kept
			else:
				keep_phased_A, keep_phased_B = is_CN_gain_in_k(k, CNV_seg, 
					ssm_infl_cnv_same_lineage[seg_index])

			# eventually change phasing
			if keep_phased_A == False:
				# SSMs were phased to A and they are not constrained by user
				if present_ssms[seg_index][cons.A][k] == True:
					if user_ssm is None:
						present_ssms[seg_index][cons.A][k] = False
						present_ssms[seg_index][cons.UNPHASED][k] = True
					elif user_ssm[seg_index][cons.A][k] == False:
						present_ssms[seg_index][cons.A][k] = False
						present_ssms[seg_index][cons.UNPHASED][k] = True
			if keep_phased_B == False:
				# SSMs were phased to B and they are not constrained by user
				if present_ssms[seg_index][cons.B][k] == True:
					if user_ssm is None:
						present_ssms[seg_index][cons.B][k] = False
						present_ssms[seg_index][cons.UNPHASED][k] = True
					elif user_ssm[seg_index][cons.B][k] == False:
						present_ssms[seg_index][cons.B][k] = False
						present_ssms[seg_index][cons.UNPHASED][k] = True

# checks whether ancestors of k and k have a CN loss in any allele
# ancok: ancestors and lineage k itself
# CNV_seg: CNVs of current segment
def do_ancestors_have_CN_loss(ancok, CNV_seg):
	CN_loss_present = False

	# check all relevant lineages
	for a in ancok:
		# CN loss on A?
		try:
			if CNV_seg[cons.LOSS][cons.A][a] is not None:
				CN_loss_present = True
				break
		except KeyError:
			pass
		# CN loss on B?
		try:
			if CNV_seg[cons.LOSS][cons.B][a] is not None:
				CN_loss_present = True
				break
		except KeyError:
			pass

	return CN_loss_present

# checks whether descendants have a CN change in any allele
# des: descendant
# CNV_seg: CNVs of current segment
def do_descendants_have_CN_change(des, CNV_seg):
	CN_present = False
	# iterates through all descendants, CN changes and phases
	for d in des:
		for c in [cons.GAIN, cons.LOSS]:
			for p in [cons.A, cons.B]:
				try:
					if CNV_seg[c][p][d] is not None:
						CN_present = True
						break
				except KeyError:
					pass
			if CN_present == True:
				break
		if CN_present == True:
			break
	return CN_present

# checks whether lineage k itself contains CN gains and whether it contains SSMs that are influenced by these gains
# lineage k
# CNV_seg: CNVs of current segment
# ssm_infl_cnv_same_lin_i_k: list whether SSMs are influences by CN changes in same lineage or not
def is_CN_gain_in_k(k, CNV_seg, ssm_infl_cnv_same_lin_i):
	keep_phased_A = False
	keep_phased_B = False
	CN_gains_A_B = [False, False]

	# check if CN gains are present in k
	for p in [cons.A, cons.B]:
		try:
			if CNV_seg[cons.GAIN][p][k] is not None:
				CN_gains_A_B[p] = True
		except KeyError:
			pass
	
	# CN gain in phase A
	if CN_gains_A_B[cons.A] == True:
		# if A has SSMs that are influenced by gain
		if ssm_infl_cnv_same_lin_i[cons.A][k] == True:
			keep_phased_A = True
	# CN gain in phase B
	if CN_gains_A_B[cons.B] == True:
		# if B has SSMs that are influenced by gain
		if ssm_infl_cnv_same_lin_i[cons.B][k] == True:
			keep_phased_B = True

	return keep_phased_A, keep_phased_B

# my_lineages: list with lineages objects how they are after the optimization
# z_matrix_fst_rnd: Z-matrix after first round of updates, where only -1's are introduced
# z_matrix_list: list with all Z-matrices after second round of updates
# origin_present_ssms: list with phasing of SSMs before any update
# present_ssms_list: lists with phasing of SSMs after update of Z-matrix
def adapt_lineages_after_Z_matrix_update(my_lineages, z_matrix_fst_rnd, z_matrix_list, origin_present_ssms, 
	present_ssms_list):

	# if after checking hard cases and LDR the Z-matrix didn't change, the lineages don't have to be updated
	if len(z_matrix_list) == 1:
		if (np.array_equal(np.asarray(z_matrix_fst_rnd), z_matrix_list[0])):
			return my_lineages, [my_lineages]

	#TODO go here when ground truth simulated data is used
	# 	because it contains only phased SSMs, the SSM phasing in the lineages might need to be updated
	#	altough the Z-matrices don't differ, but is not so important right now because I don't use this
	#	information currently
	# copy lineages for each Z-matrix, update the sublineages and the phasing
	new_lineages_list = [create_updates_lineages(my_lineages, i, z_matrix_list, origin_present_ssms, present_ssms_list)
		for i in range(len(z_matrix_list))]

	# update  my_lineages
	my_lineages = new_lineages_list[0]

	return my_lineages, new_lineages_list

# creates new lineages with updated sublineages and phased SSMs
# my_lineages: lineages from after optimization
# i: index of the current Z-matrix
# z_matrix_list: list with all Z-matrices after second round of updates
# origin_present_ssms: list with phasing of SSMs before any update
# present_ssms_list: lists with phasing of SSMs after update of Z-matrix
def create_updates_lineages(my_lineages, i, z_matrix_list, origin_present_ssms, present_ssms_list):
	new_lineages = copy.deepcopy(my_lineages)
	# update the sublineages
	update_sublineages_after_Z_matrix_update(new_lineages, z_matrix_list[i])
	# update the phasing
	update_SSM_phasing_after_Z_matrix_update(new_lineages, origin_present_ssms, present_ssms_list[i])

	return new_lineages

# current_lineages: lineages that might need to be modified
# origin_present_ssms: list with phasing of SSMs before any update
# current_ssms_list: lists with phasing of SSMs after update of Z-matrix
#	3D list: [segment][A, B, unphased][lineage]
def update_SSM_phasing_after_Z_matrix_update(current_lineages, origin_present_ssms, current_ssms_list):
	seg_num = len(origin_present_ssms)
	lin_num = len(current_lineages)

	# get the SSMs orderd in lists by their segment indices
	ssms_per_segments = get_ssms_per_segments(current_lineages, seg_num)

	# compare all lineages
	for lin_index in range(lin_num):
		# compare all segments
		for seg_index in range(seg_num):
			# check all cases
			# original: A: true, B: true, unphased: false
			if (origin_present_ssms[seg_index][cons.A][lin_index] == True
				and origin_present_ssms[seg_index][cons.B][lin_index] == True
				and origin_present_ssms[seg_index][cons.UNPHASED][lin_index] == False):
				# A: true, B: true, unphased: false --> nothing
				if (current_ssms_list[seg_index][cons.A][lin_index] == True
					and current_ssms_list[seg_index][cons.B][lin_index] == True
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
						pass
				# A: true, B: false, unphased: true --> B -> U
				elif (current_ssms_list[seg_index][cons.A][lin_index] == True
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == True):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.UNPHASED, cons.B)
				# A: true, B: false, unphased: false --> B -> A
				elif (current_ssms_list[seg_index][cons.A][lin_index] == True
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.A, cons.B)
				# A: false, B: true, unphased: true --> A -> U
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == True
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == True):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.UNPHASED, cons.A)
				# A: false, B: true, unphased: false --> A -> B
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == True
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.B, cons.A)
				# A: false, B: false, unphased: true --> A & B -> U
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == True):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.UNPHASED, cons.A)
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.UNPHASED, cons.B)
			# original: A: true, B: false, unphased: true
			elif (origin_present_ssms[seg_index][cons.A][lin_index] == True
				and origin_present_ssms[seg_index][cons.B][lin_index] == False
				and origin_present_ssms[seg_index][cons.UNPHASED][lin_index] == True):
				# A: true, B: false, unphased: true --> nothing
				if (current_ssms_list[seg_index][cons.A][lin_index] == True
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == True):
					pass
				# A: true, B: false, unphased: false --> unphased -> A
				elif (current_ssms_list[seg_index][cons.A][lin_index] == True
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.A, cons.UNPHASED)
				else:
					raise eo.MyException("This SSM moving case is not possible.")
			# original: A: true, B: false, unphased: false
			elif (origin_present_ssms[seg_index][cons.A][lin_index] == True
				and origin_present_ssms[seg_index][cons.B][lin_index] == False
				and origin_present_ssms[seg_index][cons.UNPHASED][lin_index] == False):
				# A: true, B: false, unphased: false --> nothing
				if (current_ssms_list[seg_index][cons.A][lin_index] == True
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					pass
				# A: false, B: true, unphased: false --> A -> B
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == True
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.B, cons.A)
				# A: false, B: false, unphased: true --> A -> U
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == True):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.UNPHASED, cons.A)
				else:
					raise eo.MyException("This SSM moving case is not possible.")
			# original: A: false, B: true, unphased: true
			elif (origin_present_ssms[seg_index][cons.A][lin_index] == False
				and origin_present_ssms[seg_index][cons.B][lin_index] == True
				and origin_present_ssms[seg_index][cons.UNPHASED][lin_index] == True):
				# A: false, B: true, unphased: true --> nothing
				if (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == True
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == True):
					pass
				# A: false, B: true, unphased: false --> unphased -> B
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == True
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.B, cons.UNPHASED)
				else:
					raise eo.MyException("This SSM moving case is not possible.")
			# original: A: false, B: true, unphased: false
			elif (origin_present_ssms[seg_index][cons.A][lin_index] == False
				and origin_present_ssms[seg_index][cons.B][lin_index] == True
				and origin_present_ssms[seg_index][cons.UNPHASED][lin_index] == False):
				# A: true, B: false, unphased: false --> B -> A
				if (current_ssms_list[seg_index][cons.A][lin_index] == True
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.A, cons.B)
				# A: false, B: true, unphased: false --> nothing
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == True
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					pass
				# A: false, B: false, unphased: true --> B -> U
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == True):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.UNPHASED, cons.B)
				else:
					raise eo.MyException("This SSM moving case is not possible.")
			# original: A: false, B: false, unphased: true
			elif (origin_present_ssms[seg_index][cons.A][lin_index] == False
				and origin_present_ssms[seg_index][cons.B][lin_index] == False
				and origin_present_ssms[seg_index][cons.UNPHASED][lin_index] == True):
				# A: true, B: false, unphased: false --> unphased -> A
				if (current_ssms_list[seg_index][cons.A][lin_index] == True
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.A, cons.UNPHASED)
				# A: false, B: true, unphased: false --> unphased -> B
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == True
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == False):
					move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, cons.B, cons.UNPHASED)
				# A: false, B: false, unphased: true --> nothing
				elif (current_ssms_list[seg_index][cons.A][lin_index] == False
					and current_ssms_list[seg_index][cons.B][lin_index] == False
					and current_ssms_list[seg_index][cons.UNPHASED][lin_index] == True):
					pass
				else:
					raise eo.MyException("This SSM moving case is not possible.")

		# update phasing of current lineage
		current_lineages[lin_index].ssms = get_updated_SSM_list(lin_index, cons.UNPHASED,
			ssms_per_segments)
		current_lineages[lin_index].ssms_a = get_updated_SSM_list(lin_index, cons.A,
			ssms_per_segments)
		current_lineages[lin_index].ssms_b = get_updated_SSM_list(lin_index, cons.B,
			ssms_per_segments)

# flattens the list
# lin_index: index of the lineage
# phase: current phase
# ssms_per_segments: list in which SSMs are put in lists with their segment index
#	3D index: [lin_index][A, B, unphased][seg_index]
def get_updated_SSM_list(lin_index, phase, ssms_per_segments):
	return [j for i in ssms_per_segments[lin_index][phase] for j in i]

# changes the phasing of the SSMs in their segment lists
# ssms_per_segments: list in which SSMs are put in lists with their segment index
#	3D index: [lin_index][A, B, unphased][seg_index]
# seg_index: index of the current segment
# lin_index: index of the current lineage
# new_phase: phase to which SSMs should be assigned
# old phase: phase in which SSMs where before
def move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, new_phase, old_phase):
	sort_ssms = False
	# if lineage already has SSMs in the phase in the current segment, SSMs need to be sorted afterwards
	if len(ssms_per_segments[lin_index][new_phase][seg_index]) > 0:
		sort_ssms = True

	# swap SSMs from old to new phase
	ssms_per_segments[lin_index][new_phase][seg_index] += ssms_per_segments[lin_index][old_phase][seg_index]
	ssms_per_segments[lin_index][old_phase][seg_index] = []

	# SSMs get sorted
	if sort_ssms:
		#TODO
		ssms_per_segments[lin_index][new_phase][seg_index] = sorted(ssms_per_segments[lin_index][new_phase][seg_index],
			key=lambda x: (x.chr, x.pos))
	
# for all lineages, assigns all SSMs per phase to a list with its segment index
#	3D index: [lin_index][A, B, unphased][seg_index]
# my_lineages: lineages
# seg_num: number of segments
def get_ssms_per_segments(my_lineages, seg_num):

	lin_num = len(my_lineages)
	
	# create list
	ssms_per_segments = [[] for _ in range(lin_num)]

	# fill list for all lineages
	for i in range(lin_num):
		current_lin = my_lineages[i]
		# create lists for each phase
		ssms_per_segments[i] = [[], [], []]
		ssms_per_segments[i][cons.A] = get_ssms_per_segments_lineage_phase(current_lin.ssms_a, seg_num)
		ssms_per_segments[i][cons.B] = get_ssms_per_segments_lineage_phase(current_lin.ssms_b, seg_num)
		ssms_per_segments[i][cons.UNPHASED] = get_ssms_per_segments_lineage_phase(current_lin.ssms, seg_num)

	return ssms_per_segments
	
# assigns all SSMs to a list with its segment index
# my_ssms: list with all SSMs of the lineage
# seg_num: number of segments
def get_ssms_per_segments_lineage_phase(my_ssms, seg_num):
	# create list with empty list for each segment
	ssms_per_segment_tmp = [[] for _ in range(seg_num)]

	# append each SSM to the segment list with its segment index
	for ssm in my_ssms:
		ssms_per_segment_tmp[ssm.seg_index].append(ssm)

	return ssms_per_segment_tmp



# updates the sublineages of each lineage according to the entries in the Z-matrix
# current_lineages: lineages that might need to be modified
# current_z_matrix: Z-matrix
def update_sublineages_after_Z_matrix_update(current_lineages, current_z_matrix):
	for i, lineage_relation in enumerate(current_z_matrix):
		# if lineage i is ancestor of lineage j, the entry lineage_relation[j] is 1
		# and the index of j will be given to the sublineages list of lineage i
		current_lineages[i].sublins = np.where(lineage_relation == 1)[0].tolist()
	

# iterates through all segments, to get the number of CN gains and losses, the CNVs itself and in which
#	phases SSMs appear, all info per segment
# present_ssms: 3D list with booleans: [segment][A,B,unphased][lineage]
# ssm_infl_cnv_same_lineage 3D list: [segment][lineage][A, B]
# evaluation_param: if function is called from evaluation function
def get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
	ssm_infl_cnv_same_lineage, evaluation_param=False):

	cnvs_a_index = [0] * lineage_num
	cnvs_b_index = [0] * lineage_num
	ssms_a_index = [0] * lineage_num
	ssms_b_index = [0] * lineage_num
	ssms_index = [0] * lineage_num

	for seg_index in range(seg_num):

		# set temporary variables to 0
		tmp_gain_num = 0
		tmp_loss_num = 0
		tmp_CNVs = {}
		# store where SSMs appear in the lineages, whether they are phased to A, B or unphased
		tmp_present_ssms = [[False] * lineage_num for _ in range(3)]
		tmp_ssm_infl_cnv_same_lineage = [[False] * lineage_num for _ in range(2)]

		# go through all lineages to get current CN changes and to see where SSMs occur
		for lin_index in range(1, lineage_num):
			# look at CN changes
			tmp_gain_num, tmp_loss_num = add_CN_change_to_hash(my_lineages, lin_index, seg_index,
				tmp_CNVs, tmp_gain_num, tmp_loss_num, cons.A, cnvs_a_index)
			tmp_gain_num, tmp_loss_num = add_CN_change_to_hash(my_lineages, lin_index, seg_index,
				tmp_CNVs, tmp_gain_num, tmp_loss_num, cons.B, cnvs_b_index)
			# don't check for LOH anymore
			## only check for LOH if the current function is not called for evaluation
			#if evaluation_param == False:
			#	is_it_LOH(tmp_gain_num, tmp_loss_num, tmp_CNVs)

			# look what kind of SSMs appear
			get_present_ssms(tmp_present_ssms, lin_index, my_lineages, seg_index,
				cons.A, ssms_a_index, tmp_ssm_infl_cnv_same_lineage)
			get_present_ssms(tmp_present_ssms, lin_index, my_lineages, seg_index,
				cons.B, ssms_b_index, tmp_ssm_infl_cnv_same_lineage)
			get_present_ssms(tmp_present_ssms, lin_index, my_lineages, seg_index,
				cons.UNPHASED, ssms_index)

		gain_num.append(tmp_gain_num)
		loss_num.append(tmp_loss_num)
		CNVs.append(tmp_CNVs)
		present_ssms.append(tmp_present_ssms)
		ssm_infl_cnv_same_lineage.append(tmp_ssm_infl_cnv_same_lineage)

# case 1f) two losses, different alleles and lineages
# 	check for SSMs in downstream lineages
# case 2d) check for SSMs in upstream lineages when there are losses before
# case 2d_new) check for SSMs in upstream lineages when there are gains and losses
# case 2g) check for SSMs in upstream lineages when there are gains
# case 2j) check for SSMs in lineages between when two losses happen
# case 2j_new) check for SSMs between lineages were ancestor has loss and descendants have change
# first run only produces result if relation between lineages is known
# second run then checks for cases where relation is not known
# z_matrix_fst_rnd: matrix after first round of analysis, is not defined in first round, only in second round
# triplets_list: list with triplet lists, each entry is [triplet_xys, triplet_ysx, triplet_xsy]
# present_ssms_list: list where entries are copies of "present_ssms", is a 3D list where for each segment 
#	and lineage the phasing of SSMs is stored
# seg_index: segment index
# CNVs_all: contains CNV list/hash for all segments
def check_1f_2d_2g_2j_losses_gains(spec_mut_num, current_CNVs, z_matrix, zero_count, current_present_ssms, 
	triplet_xys, triplet_ysx, triplet_xsy, first_run=True, mutations=cons.LOSS,
	z_matrix_fst_rnd=None, z_matrix_list=None, triplets_list=None, present_ssms_list=None, seg_index=None,
	CNVs_all=None, path_lineages=None, seg_num=None, last=None, ppm=None, avFreqs=None, linFreqs=None, zmco=None, gain_num=None, 
	loss_num=None, CNVs=None, present_ssms=None, defparent=None, mut_num_B=0, mutations_B=None, initial_pps_for_all=None,
	noise_threshold=0):
	# at least 2 CN changes of the specific mutation type or 1 CN change of each
	if spec_mut_num < 2 and (spec_mut_num < 1 or mut_num_B < 1):
		return zero_count

	# both phases need to be affected if only one type of mutations is considered
	if len(list(current_CNVs[mutations].keys())) != 2 and mutations_B is None:
		return zero_count

	if current_present_ssms is not None:
		lin_num = len(current_present_ssms[cons.UNPHASED])
	else:
		lin_num = len(present_ssms_list[0][0][cons.UNPHASED])

	# get affected lineages
	affected_A = list(current_CNVs[mutations][cons.A].keys())
	if mutations_B is None:
		affected_B = list(current_CNVs[mutations][cons.B].keys())
	else:
		try:
			affected_B = list(current_CNVs[mutations_B][cons.B].keys())
		except KeyError:
			return zero_count
	# get overlap of affected and path lineages
	if path_lineages is not None:
		affected_A = np.intersect1d(np.asarray(affected_A), path_lineages).tolist()
		affected_B = np.intersect1d(np.asarray(affected_B), path_lineages).tolist()

	# check all pairs between the alleles
	for lin_A in affected_A:
		for lin_B in affected_B:
			# don't check for equal lineages
			if lin_A == lin_B:
				continue
			k_prime, k_prime_prime = sorted([lin_A, lin_B])

			# first run (first round)
			if first_run:

				# the lineages are in an ancestor-descendant relation
				if z_matrix[k_prime][k_prime_prime] == 1:
					if mutations == cons.LOSS or mutations_B == cons.LOSS:
						# check all lower lineages (case 1f)
						if mutations_B is None:
							for low_lin in range(k_prime_prime+1, lin_num):
								if current_present_ssms[cons.UNPHASED][low_lin]:
									zero_count, old_z_status = update_z_matrix_first_round_m1(
										z_matrix,
										zero_count, k_prime_prime, low_lin,
										triplet_xys, triplet_ysx, triplet_xsy,
										seg_num=seg_num, last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, 
										zmco=zmco, gain_num=gain_num, loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms,
										defparent=defparent, initial_pps_for_all=initial_pps_for_all,
										noise_threshold=noise_threshold)
									if old_z_status == cons.Z_ONE:
										raise eo.MyException("This shouldn't happen. "
											"Both alleles were deleted, there "
											"can't be more SSMs in this lineage.")
									if zero_count == 0:
										return zero_count
						# check all lineage between (case 2j, 2j_new)
						if mutations_B is None or (k_prime == lin_A and mutations == cons.LOSS) or (k_prime == lin_B and mutations_B == cons.LOSS):
							for mid_lin in range(k_prime+1, k_prime_prime):
								if current_present_ssms[cons.UNPHASED][mid_lin]:
									#zero_count, old_z_status_p = update_z_matrix_first_round_m1(
									#	z_matrix,
									#	zero_count, k_prime, mid_lin,
									#	triplet_xys, triplet_ysx, triplet_xsy,
									#	seg_num=seg_num, last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, 
									#	zmco=zmco, gain_num=gain_num, loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms,
									#	defparent=defparent)
									zero_count, old_z_status_pp = update_z_matrix_first_round_m1(
										z_matrix,
										zero_count, mid_lin, k_prime_prime,
										triplet_xys, triplet_ysx, triplet_xsy,
										seg_num=seg_num, last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, 
										zmco=zmco, gain_num=gain_num, loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms,
										defparent=defparent, initial_pps_for_all=initial_pps_for_all,
										noise_threshold=noise_threshold)
									# if middle lineage has unphased SSMs, it's not possible
									# that is was in any relation with the other lineage
									#if old_z_status_p + old_z_status_pp >= 1:
									#	raise eo.MyException("Not possible that middle "
									#		"lineage is in any ancestor-descendant "
									#		"relation with k' or k''.")
									if old_z_status_pp >= 1:
										raise eo.MyException("Not possible that middle "
											"lineage is in any ancestor-descendant "
											"relation with k''.")
									if zero_count == 0:
										return zero_count

					# check all higher lineages (case 2d and case 2g and case 2d_new)
					for high_lin in range(1, k_prime):
						if current_present_ssms[cons.UNPHASED][high_lin]:
							# update relations of to k_prime AND k_prime_prime
							zero_count, old_z_status_p = update_z_matrix_first_round_m1(z_matrix, 
								zero_count, high_lin, k_prime, 
								triplet_xys, triplet_ysx, triplet_xsy, seg_num=seg_num,
								last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, 
								loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms,
								defparent=defparent, initial_pps_for_all=initial_pps_for_all,
								noise_threshold=noise_threshold)
							zero_count, old_z_status_pp = update_z_matrix_first_round_m1(z_matrix,
								zero_count, high_lin, k_prime_prime, 
								triplet_xys, triplet_ysx, triplet_xsy, seg_num=seg_num,
								last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, 
								loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms,
								defparent=defparent, initial_pps_for_all=initial_pps_for_all,
								noise_threshold=noise_threshold)
							# if higher lineage has unphased SSMs it can't be in an
							# ancestor-descendant relation to k' and/or k''
							if old_z_status_p + old_z_status_pp >= 1:
								raise eo.MyException("Not possible that higher"
									" lineage is in any relation to k' or k''")
							if zero_count == 0:
								return zero_count
			
	return zero_count

# given a lineage index and a segment index, it's determined whether the lineage has SSMs in this segment
# present_ssms_list: list where entries are copies of "present_ssms", is a 3D list where for each segment
#	and lineage the phasing of SSMs is stored
# lin_index: index of lineage which is checked
# seg_index: index of segment which is checked
def has_SSMs_in_segment(present_ssms_list, lin_index, seg_index):
	# as only the phasing but not the presence of SSMs can change for different Z-matrices,
	# it is sufficient to check the first entry in the present_ssms_list which belongs to the
	# first Z-matrix
	# if lineage has some SSMs in segment in some phase or unphased, it has SSMs
	has_SSMs = (present_ssms_list[0][seg_index][cons.A][lin_index]
		or present_ssms_list[0][seg_index][cons.B][lin_index]
		or present_ssms_list[0][seg_index][cons.UNPHASED][lin_index])
	return has_SSMs

# returns the three values, to which a triplet defined through the three lineage indices should be updated
# value: value to which should be updated
# changed_field: position of triplet that should be updated, triplet defined through k, k_prime, k_prime_prime
# current_matrix
# k, k_prime, k_prime_prime: indices of lineages
def get_values_to_update(value, changed_field, current_matrix, k, k_prime, k_prime_prime):
	if changed_field == cons.X:
		return value, current_matrix[k_prime][k_prime_prime], current_matrix[k][k_prime_prime]
	if changed_field == cons.Y:
		return current_matrix[k][k_prime], value, current_matrix[k][k_prime_prime]
	if changed_field == cons.S:
		return current_matrix[k][k_prime], current_matrix[k_prime][k_prime_prime], value

# checks whether an ancestor-descendant relation between the lineage is allowed
# can throw exception: eo.ADRelationNotPossible
# k, k_prime: lineage indices, k < k_prime
# matrix_after_first_round: Z-matrix after the first round
# present_ssms: 3D list with booleans: [segment][A,B,unphased][lineage]
# CNVs: list with hash with information in which segments appear which CN changes in which lineages
# value: value to which the field [k,k_prime] in the current matrix should be updated
def phasing_allows_relation(k, k_prime, matrix_after_first_round, present_ssms, CNVs, value):
	# only if value to which the field [k,k_prime] in the current matrix should be updated is 1
	# phasing should be considered
	if value != 1:
		raise eo.MyException("Phasing shoudn't be considered here!")

	# if ancestor-descendant relation between lineages was already given after first round, 
	# nothing needs to be done
	if matrix_after_first_round[k][k_prime] == 1:
		return True

	# check all segments whether the SSMs and CN changes in the two lineages allow an
	# ancestor-descendant relation
	for seg_index in range(len(present_ssms)):
		phasing_allows_relation_per_allele_lineage(k, k_prime, present_ssms, CNVs, cons.A, seg_index)
		phasing_allows_relation_per_allele_lineage(k, k_prime, present_ssms, CNVs, cons.B, seg_index)
		phasing_allows_relation_per_allele_lineage(k_prime, k, present_ssms, CNVs, cons.A, seg_index)
		phasing_allows_relation_per_allele_lineage(k_prime, k, present_ssms, CNVs, cons.B, seg_index)
	
	# if exception was not thrown before, an ancestor-descendant relation between the lineages is possible
	return True

# checks whether an ancestor-descendant relation between the lineage is allowed
# when the SSMs are phased to the same phase then the CN changes, no relation is allowed
# otherwise it is
# can throw exception: eo.ADRelationNotPossible
# lineage_ssm: index of lineage with SSMs
# lineage_cnv: index of lineage with CN changes
# present_ssms: 3D list with booleans: [segment][A,B,unphased][lineage]
# CNVs: list with hash with information in which segments appear which CN changes in which lineages
#	list[segments]:hash[loss, gain][A, B][lineage index]={cnv}
# phase: phase that's considered here
# seg_index: index of the current segment that is checked
def phasing_allows_relation_per_allele_lineage(lineage_ssm, lineage_cnv, present_ssms, CNVs, phase, seg_index):
	# if lineage_ssm has no SSMs phased to allele 'phase', ancestor-descendant relation between lineages is possible
	if not present_ssms[seg_index][phase][lineage_ssm]:
		return True

	# check if phasing allows an ancestor-descendant relation between lineages
	# dependant on which lineage would be an ancestor
	#
	# lineage_ssm would be the ancestor, would be influenced by gains and losses
	if lineage_ssm < lineage_cnv:
		try:
			if lineage_cnv in list(CNVs[seg_index][cons.GAIN][phase].keys()):
				raise eo.ADRelationNotPossible("Gain in lineage {0} in segment {1} forbids ancestor-descendant relation to "
                                        "lineage {2}".format(lineage_cnv, seg_index, lineage_ssm))
		except KeyError:
			pass
		try:
			if lineage_cnv in list(CNVs[seg_index][cons.LOSS][phase].keys()):
				raise eo.ADRelationNotPossible("Loss in lineage {0} in segment {1} forbids ancestor-descendant relation to "
                                "lineage {2}.".format(lineage_cnv, seg_index, lineage_ssm))
		except KeyError:
			pass
	# lineage_ssm would be the descendant, would only be influenced by losses
	else:
		try:
			if lineage_cnv in list(CNVs[seg_index][cons.LOSS][phase].keys()):
				raise eo.ADRelationNotPossible("Loss in lineage {0} in segment {1} forbids ancestor-descendant relation to "
                                "lineage {2}.".format(lineage_cnv, seg_index, lineage_ssm))
		except KeyError:
			pass
	
	return True

# k, k_prime: indices of lineages
#	k < k_prime
# present_ssms: 3D list with booleans: [segment][A,B,unphased][lineage]
# CNVs: hash with all CNVs
# matrix_after_first_round: Z-matrix after the first round
# value: value to which the field [k, k_prime] should be updated
def move_unphased_SSMs_if_necessary(k, k_prime, present_ssms, CNVs, matrix_after_first_round, value):
	# only if value is 1, SSMs can be moved
	if value != 1:
		raise eo.MyException("Value is not 1, so SSMs should not be moved!")

	# if lineage k and k_prime are already in an ancestor-descendant relation after the first round,
	# they were set in this relation in the optimization, 
	# thus the phasing of all SSMs was already considered
	if matrix_after_first_round[k][k_prime] == 1:
		return

	# it is checked whether the CNVs in lineage k_prime have an influence on the SSMs in lineage k
	# and vice versa
	unphased_checking(k, k_prime, present_ssms, CNVs)
	unphased_checking(k_prime, k, present_ssms, CNVs)
		
# if lineage with SSMs has unphased SSMs, all segments with unphased SSMs are checked
#	whether they are influenced by CN change in other lineage
# lineage_ssm: index of lineage that has unphased SSMs in a segment
# lineage_cnv: index of lineage that has CNVs in same segment
# present_ssms: 3D list with booleans: [segment][A,B,unphased][lineage]
# CNVs: list with hash with information in which segments appear which CN changes in which lineages
#	list[segments]:hash[loss, gain][A, B][lineage index]={cnv}
def unphased_checking(lineage_ssm, lineage_cnv, present_ssms, CNVs):
	# if lineage with SSMs doesn't have unphased SSMs, the function is quit
	unphased_ssms = [present_ssms[i][cons.UNPHASED][lineage_ssm] for i in range(len(present_ssms))]
	if sum(unphased_ssms) == 0:
		return

	# all segments are checked
	for seg_index in range(len(present_ssms)):
		# if lineage doesn't have unphased SSMs for this segment, nothing needs to be done
		if present_ssms[seg_index][cons.UNPHASED][lineage_ssm] == False:
			continue

		try:
			# CNVsof lineage are derived
			loss_a, loss_b, gain_a, gain_b = get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)
		except eo.no_CNVs:
			# when lineage has no CN changes, nothing needs to be done for the segment
			continue

		# otherwise lineage has one CN change on one allele
		# mutation and phase of CNVs is determined
		mutation = cons.LOSS
		phase = cons.A
		if gain_a or gain_b:
			mutation = cons.GAIN
		if loss_b or gain_b:
			phase = cons.B
		# LOH of ancestral lineage is possible, than mutations in SSM lineage have to be phased to the 
		# allele that's not deleted
		if gain_a and loss_b:
			mutation = cons.LOSS
			phase = cons.B
		# if CN change influences the SSMs in the other lineage, the SSMs will be moved to another phase
		cn_change_influences_ssms(lineage_ssm, lineage_cnv, present_ssms, seg_index, mutation, phase)

# gets CNVs of lineage
#	and checks whether multiple CN changes appear
# lineage_cnv: index of lineage with potential CNVs
# CNVs: hash with all CNVs
# seg_index: index of current segment
def get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm):
	# Do different kinds of CNVs happen in the lineage that's checked for CN changes?
	loss_a = has_CNV_in_phase(lineage_cnv, CNVs, seg_index, cons.LOSS, cons.A)
	loss_b = has_CNV_in_phase(lineage_cnv, CNVs, seg_index, cons.LOSS, cons.B)
	gain_a = has_CNV_in_phase(lineage_cnv, CNVs, seg_index, cons.GAIN, cons.A)
	gain_b = has_CNV_in_phase(lineage_cnv, CNVs, seg_index, cons.GAIN, cons.B)
	# when no CNVs appear, nothing needs to be done
	if sum([loss_a, loss_b, gain_a, gain_b]) == 0:
		raise eo.no_CNVs("No CN changes in segment {0} of lineage {1}".format(
			seg_index, lineage_cnv))
	# not possible that descendant lineage has losses or gains on both alleles
	if (lineage_ssm < lineage_cnv) and ((loss_a and loss_b == True) or (gain_a and gain_b == True)):
		raise eo.ADRelationNotPossible("When lineage {0} has CNVs on both alleles, there cannot be "
			"an ancestor-descendant relation with lineage {1}.".format(
			lineage_cnv, lineage_ssm))
	# not possible that ancestral lineage has losses on both alleles
	if (lineage_ssm > lineage_cnv) and (loss_a and loss_b == True):
		raise eo.ADRelationNotPossible("When lineage {0} has losses on both alleles, there cannot be "
			"an ancestor-descendant relation with lineage {1}.".format(
			lineage_cnv, lineage_ssm))
	# not possible that descendant lineage has LOH
	if (lineage_ssm < lineage_cnv) and ((loss_a and gain_b == True) or (loss_b and gain_a == True)):
		raise eo.ADRelationNotPossible("In case of LOH in lineage {0}, there shouldn't be any checking "
			"for hard cases!".format(lineage_cnv))
	
	return loss_a, loss_b, gain_a, gain_b

# checks whether the current lineage has a CNV of the given type in the given phase
# lineage_cnv: index of lineage with potential CNVs
# CNVs: hash with all CNVs
# seg_index: index of current segment
# mutation_type: Does the current lineage have a CNV of this kind?
# phase: Does the current lineage have a CNV in this phase?
def has_CNV_in_phase(lineage_cnv, CNVs, seg_index, mutation_type, phase):
	try:
		if lineage_cnv in list(CNVs[seg_index][mutation_type][phase].keys()):
			return True
		else:
			return False
	except KeyError:
		return False
		

# if the CN change in one lineage could influence SSMs in another linages, the SSMs are moved to the
#	other allele
# lineage_ssm: index of lineage that has unphased SSMs in a segment
# lineage_cnv: index of lineage that has CNVs in same segment
# present_ssms: 3D list with booleans: [segment][lineage][A,B,unphased]
# seg_index: index of segment in which the SSMs are considered
# mutation: type of CN change: either loss or gain
# phase: phase to which CN change belongs
def cn_change_influences_ssms(lineage_ssm, lineage_cnv, present_ssms, seg_index, mutation, phase):
	# lineage with SSMs is ancestor of lineage with CN changes
	if lineage_ssm < lineage_cnv:
		# mutation is a loss or gain
		if mutation == cons.LOSS or mutation == cons.GAIN:
			# unphased SSMs of lineage_ssm are moved to the other phase of the CN change of lineage_cnv
			move_unphased_SSMs(present_ssms, seg_index, lineage_ssm, other_phase(phase))
	# lineage with SSMs is desendant of lineage with CN changes
	else:
		# SSMs of decendant lineage can ony be influenced by CN loss
		if mutation == cons.LOSS:
			# unphased SSMs of lineage_ssm are moved to the other phase of the CN change of lineage_cnv
			move_unphased_SSMs(present_ssms, seg_index, lineage_ssm, other_phase(phase))

# changes the phase of the unphased SSMs in a segment of a lineage
# present_ssms: 3D list with booleans: [segment][lineage][A,B,unphased]
# seg_index: index of segment in which the SSMs are considered
# current_lin: index of lineage who's SSM phases should be changed
# phase: phase to which unphased SSMs should be changed
def move_unphased_SSMs(present_ssms, seg_index, current_lin, phase):
	# check that no SSMs are phased to the phase already that will be influenced by the CN change
	if present_ssms[seg_index][other_phase(phase)][current_lin] == True:
		raise eo.MyException("Some SSMs are already phased to phase that will be influenced by the "
			"CN change. This should not happen!")
	# phases of unphased SSMs are updated
	present_ssms[seg_index][phase][current_lin] = True
	present_ssms[seg_index][cons.UNPHASED][current_lin] = False

# given a phase, the other phase is returned
def other_phase(phase):
	if phase == cons.A:
		return cons.B
	return cons.A

# case 2i) considers all changes that can happen somewhere, SSMs in upstream lineages are influenced
def check_2i_phased_changes(current_CNVs, z_matrix, zero_count, current_present_ssms, triplet_xys, triplet_ysx, triplet_xsy, path_lineages=None,
	seg_num=None, last=None, ppm=None, avFreqs=None, linFreqs=None, zmco=None, gain_num=None, loss_num=None, CNVs=None, present_ssms=None,
	defparent=None, initial_pps_for_all=None, noise_threshold=0):
	for changes in current_CNVs.keys():
		for phases in current_CNVs[changes].keys():
			for my_lin in current_CNVs[changes][phases]:
				# if path lineages are given, current lineage has to lie on path
				if path_lineages is not None and my_lin not in path_lineages:
					continue
				# check all higher lineages whether they have phased SSMs
				for higher_lin in range(1, my_lin):
					if (current_present_ssms[phases][higher_lin]):
						zero_count, old_z_status = update_z_matrix_first_round_m1(z_matrix, 
							zero_count, higher_lin, my_lin,
							triplet_xys, triplet_ysx, triplet_xsy, seg_num=seg_num, last=last,
							ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, 
							loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms, defparent=defparent,
							initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
						if zero_count == 0:
							return zero_count 
	return zero_count 

# case 2h) LOH in a lineage, influence on SSMs in upstream lineage
def check_2h_LOH(current_loss_num, current_gain_num, current_CNVs, z_matrix, zero_count, current_present_ssms, triplet_xys, triplet_ysx, triplet_xsy, 
	path_lineages=None, seg_num=None, last=None, ppm=None, avFreqs=None, linFreqs=None, zmco=None, gain_num=None, loss_num=None, CNVs=None, 
	present_ssms=None, defparent=None, initial_pps_for_all=None, noise_threshold=0):
	# both current_loss_num and current_gain_num have to be at least one
	if current_loss_num == 0 or current_gain_num == 0:
		return zero_count

	# check whether the lineages with a gain also contain a loss
	gain_phase = list(current_CNVs[cons.GAIN].keys())
	if len(gain_phase) == 2:
		raise eo.MyException("Gains and losses within one allele in one segment are not allowed.")
	gain_phase = gain_phase[0]
	loss_phase = cons.B
	if gain_phase == loss_phase:
		loss_phase = cons.A
	gain_lins = list(current_CNVs[cons.GAIN][gain_phase].keys())
	# if path lineages are given, gain_lin has to be part of it
	if path_lineages is not None:
		raise eo.MyException("Check how this should be done with multiple gain_lins.")
		if gain_lin not in path_lineages:
			return zero_count
	# find intersection of gain_lins and lins with loss
	loss_lins = list(current_CNVs[cons.LOSS][loss_phase].keys())
	loh_lins = np.intersect1d(np.asarray(gain_lins), np.asarray(loss_lins)).tolist()
	# check for upstream lineages
	for my_lin in loh_lins:
		for higher_lin in range(1, my_lin):
			if (current_present_ssms[cons.A][higher_lin] or current_present_ssms[cons.B][higher_lin]
				or current_present_ssms[cons.UNPHASED][higher_lin]):
					zero_count, old_z_status = update_z_matrix_first_round_m1(z_matrix, zero_count,
						higher_lin, my_lin, triplet_xys, triplet_ysx, triplet_xsy, seg_num=seg_num,
						last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, loss_num=loss_num, 
						CNVs=CNVs, present_ssms=present_ssms, defparent=defparent, initial_pps_for_all=initial_pps_for_all,
						noise_threshold=noise_threshold)
					if zero_count == 0:
						return zero_count
	return zero_count

# case 2f) two gains on different alleles in the same lineage, influence SSMs in upstream lineages
def check_2f_CN_gains(current_gain_num, current_CNVs, z_matrix, zero_count, current_present_ssms, triplet_xys, triplet_ysx, triplet_xsy, path_lineages=None,
	seg_num=None, last=None, ppm=None, avFreqs=None, linFreqs=None, zmco=None, gain_num=None, loss_num=None, CNVs=None, present_ssms=None,
	defparent=None, initial_pps_for_all=None, noise_threshold=0):
	# needs at least two CN gains
	if current_gain_num < 2:
		return zero_count

	# check whether gains are contained on both alleles
	if len(current_CNVs[cons.GAIN]) < 2:
		return zero_count

	lin_num = len(current_present_ssms[0])
	affacted_lineages = list(current_CNVs[cons.GAIN][cons.B].keys())
	# get overlap of affected and path lineages
	if path_lineages is not None:
		affected_lineages = np.intersect1d(np.asarray(affected_lineages), path_lineages).tolist()

	# check whether a lineage has two gains
	# it's enough to check whether the lineage that gains an additional B-allele also gains
	# the A-allele
	for my_lin in affacted_lineages:
		if my_lin in current_CNVs[cons.GAIN][cons.A]:
			# check whether higher lineages have unphased SSMs 
			for higher_lin in range(1, my_lin):
				if (current_present_ssms[cons.UNPHASED][higher_lin]):
					zero_count, old_z_status = update_z_matrix_first_round_m1(z_matrix, zero_count,
						higher_lin, my_lin, triplet_xys, triplet_ysx, triplet_xsy, seg_num=seg_num,
						last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, 
						loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms, defparent=defparent,
						initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
					if zero_count == 0:
						return zero_count
	return zero_count


# case 1d) two losses in same lineage, no SSMs in downstream lineages
# case 2d) two losses in same lineage, influence on SSMs in upstream lineages
def check_1d_2c_CN_losses(current_loss_num, currentCNVs, z_matrix, zero_count, current_present_ssms, triplet_xys, triplet_ysx, triplet_xsy, path_lineages=None,
	seg_num=None, last=None, ppm=None, avFreqs=None, linFreqs=None, zmco=None, gain_num=None, loss_num=None, CNVs=None, present_ssms=None,
	defparent=None, initial_pps_for_all=None, noise_threshold=0):
	# needs at least two CN losses
	if current_loss_num < 2:
		return zero_count

	# check whether losses are contained on both alleles
	try:
		if len(list(currentCNVs[cons.LOSS][cons.A].keys())) == 0 or len(list(currentCNVs[cons.LOSS][cons.B].keys())) == 0:
			return zero_count
	except KeyError:
		return zero_count

	lin_num = len(current_present_ssms[0])
	affacted_lineages = list(currentCNVs[cons.LOSS][cons.A].keys())
	# get overlap of affected and path lineages
	if path_lineages is not None:
		affected_lineages = np.intersect1d(np.asarray(affected_lineages), path_lineages).tolist()

	# check whether a lineage has two losses
	# it's enough to check whether the lineage that loose the A-allele also lost the
	# B-allele
	for my_lin in affacted_lineages:
		if my_lin in currentCNVs[cons.LOSS][cons.B]:
			# check all lower lineages, it's not possible that they have an SSMs at all
			for lower_lin in range(my_lin+1, lin_num):
				# if lower lineage has unphased SSMs it can't be the child
				if (current_present_ssms[cons.UNPHASED][lower_lin]):
					zero_count, old_z_status = update_z_matrix_first_round_m1(z_matrix, 
						zero_count, my_lin, lower_lin,
						triplet_xys, triplet_ysx, triplet_xsy, seg_num=seg_num, last=last,
						ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, 
						loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms, defparent=defparent,
						initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
					# check for errors in results
					if old_z_status == cons.Z_ONE:
						raise eo.MyException("This shouldn't happen, "
							"when both alleles are deleted there can't "
							"be SSMs.")
					if zero_count == 0:
						return zero_count
			# check upstream lineages, if they have unphased SSMs and are not in a relation with
			# the current lineage, a relation would change the likelihood, better not to
			# allow it
			for higher_lin in range(1, my_lin):
				if (current_present_ssms[cons.UNPHASED][higher_lin]):
					zero_count, old_z_status = update_z_matrix_first_round_m1(z_matrix, zero_count,
						higher_lin, my_lin, triplet_xys, triplet_ysx, triplet_xsy, seg_num=seg_num,
						last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, 
						loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms, defparent=defparent,
						initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
					if zero_count == 0:
						return zero_count
	return zero_count

# case 1c): downstream SSM can be on a deleted allele
def check_1c_CN_loss(current_loss_num, current_CNVs, z_matrix, zero_count, current_present_ssms, triplet_xys, triplet_ysx, triplet_xsy, path_lineages=None, 
	seg_num=None, last=None, ppm=None, avFreqs=None, linFreqs=None, zmco=None, gain_num=None, loss_num=None, CNVs=None, present_ssms=None,
	defparent=None, initial_pps_for_all=None, noise_threshold=0):
	# needs at least one CN loss
	if current_loss_num == 0:
		return zero_count

	# check for A allele
	zero_count = check_1c_CN_loss_phase(current_loss_num, current_CNVs, z_matrix, zero_count, cons.A, current_present_ssms,
		triplet_xys, triplet_ysx, triplet_xsy, path_lineages, seg_num=seg_num, last=last,
		ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms,
		defparent=defparent, initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
	# all 0 entries in the Z matrix were checked and changed
	if zero_count == 0:
		return zero_count
	# check for B allele
	zero_count = check_1c_CN_loss_phase(current_loss_num, current_CNVs, z_matrix, zero_count, cons.B, current_present_ssms,
		triplet_xys, triplet_ysx, triplet_xsy, path_lineages, seg_num=seg_num, last=last,
		ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms,
		defparent=defparent, initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)

	return zero_count

def check_1c_CN_loss_phase(current_loss_num, current_CNVs, z_matrix, zero_count, phase, current_present_ssms, triplet_xys, triplet_ysx, triplet_xsy, 
	path_lineages=None, 
	seg_num=None, last=None, ppm=None, avFreqs=None, linFreqs=None, zmco=None, gain_num=None, loss_num=None, CNVs=None, present_ssms=None,
	defparent=None, initial_pps_for_all=None, noise_threshold=0):
	# CN loss needs to affect a lineage
	try:
		affected_lineages = sorted(list(current_CNVs[cons.LOSS][phase].keys()))
		# get overlap of affected and path lineages
		if path_lineages is not None:
			affected_lineages = np.intersect1d(np.asarray(affected_lineages), path_lineages).tolist()
	except KeyError:
		return zero_count

	lin_num = len(current_present_ssms[0])
	# check for each affected lineage and all lower lineages
	for lin_index in affected_lineages:
		for lower_lin in range(lin_index+1, lin_num):
			if current_present_ssms[phase][lower_lin]:
				zero_count, old_z_status = update_z_matrix_first_round_m1(z_matrix, zero_count, 
					lin_index, lower_lin, triplet_xys, triplet_ysx, triplet_xsy, seg_num=seg_num, last=last,
					ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, loss_num=loss_num, CNVs=CNVs, 
					present_ssms=present_ssms, defparent=defparent, initial_pps_for_all=initial_pps_for_all, 
					noise_threshold=noise_threshold)
				# check for errors in results
				if old_z_status == cons.Z_ONE:
					raise eo.MyException("This shouldn't happen, deleted allele can't "
						"have SSMs.")
				# all 0 entries in the Z matrix were checked and changed
				if zero_count == 0:
					return zero_count
	return zero_count

# case 1a): one allele can only be deleted once
def check_1a_CN_LOSS(current_loss_num, currentCNVs, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy, path_lineages=None, seg_num=None,
	last=None, ppm=None, avFreqs=None, linFreqs=None, zmco=None, gain_num=None, loss_num=None, CNVs=None, present_ssms=None, defparent=None,
	initial_pps_for_all=None, noise_threshold=0):
	# needs more than one CN loss
	if current_loss_num <= 1:
		return zero_count

	# check for A allele
	zero_count = check_1a_CN_LOSS_phase(cons.A, currentCNVs, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy, path_lineages,
		seg_num=seg_num, last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, loss_num=loss_num, 
		CNVs=CNVs, present_ssms=present_ssms, defparent=defparent, initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
	# all 0 entries in the Z matrix were checked and changed
	if zero_count == 0:
		return zero_count
	# check for B allele
	zero_count = check_1a_CN_LOSS_phase(cons.B, currentCNVs, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy, path_lineages, 
		seg_num=seg_num, last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, gain_num=gain_num, loss_num=loss_num, 
		CNVs=CNVs, present_ssms=present_ssms, defparent=defparent, initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)

	return zero_count
	
def check_1a_CN_LOSS_phase(phase, currentCNVs, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy, path_lineages=None, seg_num=None,
	last=None, ppm=None, avFreqs=None, linFreqs=None, zmco=None, gain_num=None, loss_num=None, CNVs=None, present_ssms=None, defparent=None,
	initial_pps_for_all=None, noise_threshold=0):
	# if CN loss affects lineages in this phase
	try:
		affected_lineages = sorted(list(currentCNVs[cons.LOSS][phase].keys()))
		# get overlap of affected and path lineages
		if path_lineages is not None:
			affected_lineages = np.intersect1d(np.asarray(affected_lineages), path_lineages).tolist()
	except KeyError:
		return zero_count
	# more than one lineage needs to be affected
	lin_num = len(affected_lineages)
	if lin_num <= 1:
		return zero_count

	# check all pairs of lineages
	for i in range(lin_num):
		for j in range(i+1, lin_num):
			lin_high = affected_lineages[i]
			lin_low = affected_lineages[j]
			zero_count, old_z_status = update_z_matrix_first_round_m1(z_matrix, zero_count, lin_high, lin_low,
				triplet_xys, triplet_ysx, triplet_xsy, seg_num=seg_num, last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, 
				zmco=zmco, gain_num=gain_num, loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms, defparent=defparent,
				initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
			# check for errors in results
			if old_z_status == cons.Z_ONE:
				raise eo.MyException("This shouldn't happen, deleted allele can't be"
					" deleted twice.")
			# all 0 entries in the Z matrix were checked and changed
			if zero_count == 0:
				return zero_count
	return zero_count

# updates the Z matrix, only values of 0s are changed to -1s
# lin_high: lineage with higher frequency, actually smaller index
# lin_low: lineage with lower frequency, actually higher index
def update_z_matrix_first_round_m1(z_matrix, zero_count, lin_high, lin_low, triplet_xys, triplet_ysx, triplet_xsy, seg_num=None, last=None,
	ppm=None, avFreqs=None, linFreqs=None, zmco=None, gain_num=None, loss_num=None, CNVs=None, present_ssms=None, defparent=None,
	initial_pps_for_all=None, noise_threshold=0):
	# if current entry is 1, nothing needs to be done
	if z_matrix[lin_high][lin_low] == cons.Z_ONE:
		return zero_count, cons.Z_ONE
	# if current entry is -1, also nothing needs to be done
	if z_matrix[lin_high][lin_low] == cons.Z_MINUSONE:
		return zero_count, cons.Z_MINUSONE
	# if current entry is 0, it needs to be changes to -1
	if z_matrix[lin_high][lin_low] == cons.Z_ZERO:
		# not called from sum rule algorithm
		if last is None:
			z_matrix[lin_high][lin_low] = cons.Z_MINUSONE
			zero_count -= 1

			# check whether other entries in Z matrix could be updated iteratively because of triplet changes
			zero_count = update_Z_matrix_iteratively(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy, 
				(lin_high, lin_low), seg_num=seg_num, last=last, ppm=ppm, avFreqs=avFreqs, linFreqs=linFreqs, zmco=zmco, 
				gain_num=gain_num, loss_num=loss_num, CNVs=CNVs, present_ssms=present_ssms, defparent=defparent,
				initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)

		# called from sum rule algorithm
		else:
			update_ancestry(-1, lin_high, lin_low, last, ppm, defparent, linFreqs, avFreqs, zmco, seg_num, zero_count, gain_num, 
				loss_num, CNVs, present_ssms, initial_pps_for_all=initial_pps_for_all, noise_threshold=noise_threshold)
			
		return zero_count, cons.Z_ZERO

# iterates through the SSMs of a lineage for a given segment
# finds out in which phases SSMs are present
def get_present_ssms(present_ssms, lin_index, my_lineages, seg_index, phase, ssms_index_list, 
	ssm_infl_cnv_same_lineage=None):

	current_ssms_index = ssms_index_list[lin_index]

	# get all ssms
	my_lin = my_lineages[lin_index]
	if phase == cons.A:
		my_ssms = my_lin.ssms_a
	elif phase == cons.B:
		my_ssms = my_lin.ssms_b
	else:
		my_ssms = my_lin.ssms

	# if lineage has no SSMs, function is stopped
	if my_ssms is None:
		return

	# see if SSMs are present in current segment
	if current_ssms_index < len(my_ssms) and my_ssms[current_ssms_index].seg_index == seg_index:
		present_ssms[phase][lin_index] = True
		# increase current_ssms_index until it points to the next segment
		while ((current_ssms_index < len(my_ssms) and 
			my_ssms[current_ssms_index].seg_index == seg_index)):
				# if SSM is influened by gain in same lineage, store this information
				if my_ssms[current_ssms_index].infl_cnv_same_lin == True:
					ssm_infl_cnv_same_lineage[phase][lin_index] = True
				current_ssms_index += 1
		ssms_index_list[lin_index] = current_ssms_index
		

# given the number of gains and losses it is checked whether LOH is present
def is_it_LOH(gain_num, loss_num, CNVs):
	# positive and negative CN change
	if gain_num > 0 and loss_num > 0:
		# exactly one gain and one loss
		if gain_num == 1 and loss_num == 1:
			key_gain = list(CNVs[cons.GAIN].keys())[0]
			key_loss = list(CNVs[cons.LOSS].keys())[0]
			# lineages of loss and gain are equal
			if list(CNVs[cons.GAIN][key_gain].keys()) == list(CNVs[cons.LOSS][key_loss].keys()):
				return True
			else:
				raise eo.NotProperLOH("Different lineages of loss and gain.")
		else:
			raise eo.NotProperLOH("More CN changes than allowed!")
	else:
		return False

# given a lineage and the index where to start to look for CNVs, 
# CNVs are inserted to the hash and the number of variations is updated
def add_CN_change_to_hash(my_lineages, lin_index, seg_index, CNVs, gain_num, loss_num, phase, cnv_index_list):
	# get current CNV
	cnv_index = cnv_index_list[lin_index]
	try:
		if phase == cons.A:
			my_cnv = my_lineages[lin_index].cnvs_a[cnv_index]
		else:
			my_cnv = my_lineages[lin_index].cnvs_b[cnv_index]
	# if last CNV in list was already processed
	except IndexError:
		return gain_num, loss_num
	# if no CNV exists for this lineage
	except TypeError:
		return gain_num, loss_num

	# check if current CNVs belongs to the current segment
	while my_cnv.seg_index == seg_index:
		# CN gain
		if my_cnv.change > 0:
			gain_num += 1
		# CN loss
		elif my_cnv.change < 0:
			loss_num += 1
		else:
			raise eo.MyException("Unparsed CN change!")
		# insert CN to hash at right position
		CNVs_insert(CNVs, phase, lin_index, my_cnv)
		# update CNV index as CNV was used
		cnv_index_list[lin_index] += 1
		cnv_index += 1
		try:
			if phase == cons.A:
				my_cnv = my_lineages[lin_index].cnvs_a[cnv_index]
			else:
				my_cnv = my_lineages[lin_index].cnvs_b[cnv_index]
		# if last CNV in list was already processed
		except IndexError:
			return gain_num, loss_num

	return gain_num, loss_num

# inserts the copy number variation at the right position in the has
def CNVs_insert(CNVs, phase, lin_index, my_cnv):
	# compute change direction of CNV
	change = 1
	if my_cnv.change < 0:
		change = -1
	try:
		if lin_index in CNVs[change][phase]:
			CNVs[change][phase][lin_index] = [CNVs[change][phase][lin_index]] + [my_cnv]
		else:
			CNVs[change][phase][lin_index] = my_cnv
	except KeyError:
		try:
			CNVs[change][phase] = {}
			CNVs[change][phase][lin_index] = my_cnv
		except KeyError:
			CNVs[change] = {}
			CNVs[change][phase] = {}
			CNVs[change][phase][lin_index] = my_cnv

def get_Z_matrix(my_lineages):
	
	# create empty Z matrix
	z_matrix = [[0] * len(my_lineages) for _ in range(len(my_lineages))]

	# fill with 1s
	one_count = 0
	for x in range(len(my_lineages)):
		for y in my_lineages[x].sublins:
			z_matrix[x][y] = 1
			one_count += 1
	# first row shouldn't count for the ones
	one_count = one_count - len(my_lineages) + 1
	# compute number of relevant 0s
	zero_count = get_number_of_untrivial_z_entries(len(my_lineages)) - one_count

	# fill diagonal and lower half with -1s as there can be no relations
	for x in range(len(my_lineages)):
		for y in range(0,x+1):
			z_matrix[x][y] = -1

	return z_matrix, zero_count

# in the Z-matrix, some entries are trivial as their solution is already given
# these are the entries of the first row and of the lower, left triangle including
#	the diagonal
# this leads to a specific number of entries that don't have a trivial solution
def get_number_of_untrivial_z_entries(sublin_num):
	if sublin_num > 2:
		return (((sublin_num - 1) * (sublin_num - 2)) / 2)
	return 0

def sort_segments(segment_list, segment=False):
	if segment == False:
		return sorted(segment_list, key = lambda x: (x.chr, x.start))
	else:
		return sorted(segment_list, key = lambda x: x.seg_index)


def sort_cnvs(cnv_list):
	return sorted(cnv_list, key = lambda x:(x.chr, x.start, x.end, x.change))

def sort_snps_ssms(mut_list):
	return sorted(mut_list, key = lambda x: (x.chr, x.pos))

class Z_Matrix_Co(object):

	def __init__(self, z_matrix, triplet_xys, triplet_ysx, triplet_xsy, present_ssms, matrix_after_first_round):
		self.z_matrix = z_matrix
		self.triplet_xys = triplet_xys
		self.triplet_ysx = triplet_ysx
		self.triplet_xsy = triplet_xsy
		self.present_ssms = present_ssms
		self.matrix_after_first_round = matrix_after_first_round

class Subclonal_Reconstruction_for_DFS(object):

	def __init__(self, zmco, present_ssms, ppm, defparent, avFreqs):
		self.zmco = zmco
		self.present_ssms = present_ssms
		self.ppm = ppm
		self.defparent = defparent
		self.avFreqs = avFreqs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents_file", default=None, type=str, help ="File with parent information for lineages")
    parser.add_argument("--freq_file", default=None, type=str, help ="File with frequency information for lineages")
    parser.add_argument("--cna_file", default=None, type=str, help ="File with CNA information")
    parser.add_argument("--ssm_file", default=None, type=str, help ="File with SSM information")
    parser.add_argument("--seg_file", default=None, type=str, help ="File with number of segments")
    parser.add_argument("--userZ_file", default=None, type=str, help ="File with user constraints on lineage relationships")
    parser.add_argument("--userSSM_file", default=None, type=str, help ="File with user constraints on SSM phases")
    parser.add_argument("--lineage_file", default=None, type=str, help ="File with lineage information from SubMARine")
    parser.add_argument("--impact_file", default=None, type=str, help ="File containing impact information of CNAs on SSMs")
    parser.add_argument("--z_matrix_file", default=None, type=str, help ="File with Z matrix from SubMARine")
    parser.add_argument("--possible_parent_file", default=None, type=str, help ="File with possible parent matrix from SubMARine")
    parser.add_argument("--output_prefix", default=None, type=str, help ="prefix of output files")
    parser.add_argument("--overwrite", action='store_true', help="old output files will be overwritten")
    parser.add_argument("--dfs", action='store_true', help="performs depth-first search")
    parser.add_argument("--basic_version", action='store_true', help="starts basic version")
    parser.add_argument("--extended_version", action='store_true', help="starts extended version")
    parser.add_argument("--allow_noise", action='store_true', help="allows noise in Subpoplar algorithm")
    parser.add_argument("--maximal_noise", default=-1, type=float, help ="maximal noise threshold to which Subpoplar can be extended")
    parser.add_argument("--noise_threshold", default=0, type=float, help ="noise threshold with which Subpoplar starts")
    args = parser.parse_args()

    if args.allow_noise == True or args.noise_threshold > 0:
    	allow_noise = True

    if args.dfs:
        depth_first_search(ppm_file=args.possible_parent_file, z_matrix_file=args.z_matrix_file, lin_file=args.lineage_file,
                cna_file=args.cna_file, ssm_file=args.ssm_file, output_prefix=args.output_prefix, overwrite=args.overwrite)
    elif args.basic_version:
        go_basic_version(freq_file=args.freq_file, userZ_file=args.userZ_file, output_prefix=args.output_prefix, overwrite=args.overwrite,
		cna_file=args.cna_file, ssm_file=args.ssm_file, allow_noise=allow_noise, noise_threshold=args.noise_threshold, maximal_noise=args.maximal_noise)
    elif args.extended_version:
        go_extended_version(freq_file=args.freq_file, cna_file=args.cna_file, ssm_file=args.ssm_file, impact_file=args.impact_file, 
		userZ_file=args.userZ_file, userSSM_file=args.userSSM_file,
		output_prefix=args.output_prefix, overwrite=args.overwrite, allow_noise=allow_noise, noise_threshold=args.noise_threshold, maximal_noise=args.maximal_noise)
    else:
        go_submarine(args.parents_file, args.freq_file, args.cna_file, args.ssm_file, args.seg_file, args.userZ_file, args.userSSM_file, args.output_prefix, args.overwrite)


import unittest
import io_file as oio
import constants as cons
import segment
import snp_ssm
import cnv
import lineage
import exceptions_onctopus as eo
import os
import pdb
import sys

class OnctopusIOTest(unittest.TestCase):

	def test_write_new_ssm_phasing(self):

		ssm_phasing = [[0, cons.A], [1, cons.B], [2, cons.UNPHASED]]
		output_file = "testdata/unittests/ssm_phasing_out.txt"
		true_output = "SSM_index\tphase\n0\tA\n1\tB\n2\t0\n"

		oio.write_new_ssm_phasing(ssm_phasing, output_file, overwrite=True)

		output = ""
		with open(output_file, "r") as f:
			for line in f:
				output += line

		self.assertEqual(true_output, output)

	def test_read_userSSM(self):

		input_name = "submarine_example/ex1_userSSMs.csv"
		user_ssm = oio.read_userSSM(input_name, 4, 3)

		self.assertEqual(len(user_ssm), 3)
		self.assertEqual(len(user_ssm[0]), 3)
		self.assertEqual(len(user_ssm[0][0]), 4)
		self.assertTrue(user_ssm[2][cons.B][3])
		self.assertTrue(user_ssm[1][cons.A][2])
		self.assertFalse(user_ssm[1][cons.A][3])

	def test_read_userZ(self):

		input_name = "submarine_example/ex1_userZ.csv"
		userZ = oio.read_userZ(input_name, 4)

		self.assertEqual(userZ[1][3], 1)
		self.assertEqual(userZ[2][3], -1)
		self.assertEqual(userZ[1][2], 0)

	def test_read_ssms(self):
		input_name = "submarine_example/ex1_ssms.csv"
		ssms = oio.read_ssms(input_name)

		self.assertEqual(3, len(ssms))
		self.assertEqual(ssms[0].seg_index, 0)
		self.assertEqual(ssms[0].chr, 1)
		self.assertEqual(ssms[1].pos, 10)
		self.assertEqual(ssms[1].lineage, 2)
		self.assertEqual(ssms[0].phase, cons.UNPHASED)
		self.assertEqual(ssms[1].phase, cons.A)
		self.assertEqual(ssms[2].phase, cons.B)
		self.assertFalse(ssms[2].infl_cnv_same_lin)
		self.assertTrue(ssms[1].infl_cnv_same_lin)

		# with use_SSM_index, sorting_id_mapping, and lin_num, works
		input_name = "testdata/unittests/ssms2.csv"
		sorting_id_mapping = {}
		sorting_id_mapping["1"] = 1
		sorting_id_mapping["2"] = 3
		sorting_id_mapping["3"] = 2
		lin_num = 4

		ssms = oio.read_ssms(input_name, phasing=False, sorting_id_mapping=sorting_id_mapping, use_SSM_index=True,
			lin_num=lin_num)

		self.assertEqual(ssms[0].seg_index, 0)
		self.assertEqual(ssms[0].index, 0)
		self.assertEqual(ssms[0].lineage, 1)
		self.assertEqual(ssms[1].seg_index, 1)
		self.assertEqual(ssms[1].index, 1)
		self.assertEqual(ssms[1].lineage, 3)
		self.assertEqual(ssms[2].seg_index, 2)
		self.assertEqual(ssms[2].index, 2)
		self.assertEqual(ssms[2].lineage, 2)

		# with lin_num, error: lin_num too high
		input_name = "testdata/unittests/ssms2.csv"
		sorting_id_mapping = {}
		sorting_id_mapping["1"] = 1
		sorting_id_mapping["2"] = 3
		sorting_id_mapping["3"] = 2
		lin_num = 1

		with self.assertRaises(eo.MyException):
			ssms = oio.read_ssms(input_name, phasing=False, sorting_id_mapping=sorting_id_mapping, use_SSM_index=True,
				lin_num=lin_num)

		# with use_SSM_index, error: wrong sorting
		input_name = "testdata/unittests/ssms3_wrong.csv"
		sorting_id_mapping = {}
		sorting_id_mapping["1"] = 1
		sorting_id_mapping["2"] = 3
		sorting_id_mapping["3"] = 2
		lin_num = 4

		with self.assertRaises(eo.MyException):
			ssms = oio.read_ssms(input_name, phasing=False, sorting_id_mapping=sorting_id_mapping, use_SSM_index=True,
				lin_num=lin_num)

		# sorting_id_mapping but key is wrong
		input_name = "testdata/unittests/ssms4_wrong.csv"
		sorting_id_mapping = {}
		sorting_id_mapping["1"] = 1
		sorting_id_mapping["2"] = 3
		sorting_id_mapping["3"] = 2
		lin_num = 4

		with self.assertRaises(eo.MyException):
			ssms = oio.read_ssms(input_name, phasing=False, sorting_id_mapping=sorting_id_mapping, use_SSM_index=True,
				lin_num=lin_num)

	def test_create_impact_matrix(self):

		impact_file = "testdata/unittests/impact.csv"
		cna_num = 4
		ssm_num = 3

		impact_matrix = oio.create_impact_matrix(impact_file=impact_file, cna_num=cna_num, ssm_num=ssm_num)

		self.assertEqual(impact_matrix[1][1], 1)
		self.assertEqual(impact_matrix[2][3], 1)

		# CNA index too high
		impact_file = "testdata/unittests/impact.csv"
		cna_num = 1
		ssm_num = 3

		with self.assertRaises(eo.MyException):
			impact_matrix = oio.create_impact_matrix(impact_file=impact_file, cna_num=cna_num, ssm_num=ssm_num)

		# SSM index too high
		impact_file = "testdata/unittests/impact.csv"
		cna_num = 4
		ssm_num = 1

		with self.assertRaises(eo.MyException):
			impact_matrix = oio.create_impact_matrix(impact_file=impact_file, cna_num=cna_num, ssm_num=ssm_num)

	def test_get_SSM_constraints(self):

		# SSM index too high
		userSSM_file = "testdata/unittests/userSSM2_wrong.csv"
		my_ssms = ["something"]

		with self.assertRaises(eo.MyException):
			oio.get_SSM_constraints(userSSM_file=userSSM_file, my_ssms=my_ssms)

		# undefined phase
		userSSM_file = "testdata/unittests/userSSM3_wrong.csv"
		my_ssms = ["something"]

		with self.assertRaises(eo.MyException):
			oio.get_SSM_constraints(userSSM_file=userSSM_file, my_ssms=my_ssms)

		# SSM has already a different phase
		userSSM_file = "testdata/unittests/userSSM.csv"
		ssm1 = snp_ssm.SSM()
		ssm1.phase = cons.B
		my_ssms = [ssm1]

		with self.assertRaises(eo.MyException):
			oio.get_SSM_constraints(userSSM_file=userSSM_file, my_ssms=my_ssms)

		# works
		userSSM_file = "testdata/unittests/userSSM.csv"
		ssm1 = snp_ssm.SSM()
		my_ssms = [ssm1]

		oio.get_SSM_constraints(userSSM_file=userSSM_file, my_ssms=my_ssms)

		self.assertEqual(ssm1.phase, cons.A)


	def test_read_cnas(self):
		input_name = "submarine_example/ex1_cnas.csv"
		cnas = oio.read_cnas(input_name)

		self.assertEqual(2, len(cnas))
		self.assertEqual(cnas[0].change, 2)
		self.assertEqual(cnas[0].chr, 1)
		self.assertEqual(cnas[0].start, 10)
		self.assertEqual(cnas[0].end, 100)
		self.assertEqual(cnas[0].lineage, 2)
		self.assertEqual(cnas[0].phase, cons.A)
		self.assertEqual(cnas[1].phase, cons.B)
		self.assertEqual(cnas[1].seg_index, 2)

		# with use_cna_indices, sorting_id_mapping, and lin_num, works
		input_name = "testdata/unittests/cnas2.csv"
		sorting_id_mapping = {}
		sorting_id_mapping["2"] = 1
		sorting_id_mapping["3"] = 2
		lin_num = 4

		cnas = oio.read_cnas(input_name, sorting_id_mapping=sorting_id_mapping, use_cna_indices=True, lin_num=lin_num)

		self.assertEqual(cnas[0].change, 2)
		self.assertEqual(cnas[0].lineage, 1)
		self.assertEqual(cnas[0].index, 0)
		self.assertEqual(cnas[0].phase, cons.A)
		self.assertEqual(cnas[1].lineage, 2)
		self.assertEqual(cnas[1].index, 1)

		# with lin_num, error: lin_num too high
		input_name = "testdata/unittests/cnas2.csv"
		sorting_id_mapping = {}
		sorting_id_mapping["2"] = 1
		sorting_id_mapping["3"] = 2
		lin_num = 1

		with self.assertRaises(eo.MyException):
			cnas = oio.read_cnas(input_name, sorting_id_mapping=sorting_id_mapping, use_cna_indices=True, lin_num=lin_num)

		# with use_cna_indices, error: wrong sorting
		input_name = "testdata/unittests/cnas3_wrong.csv"
		sorting_id_mapping = {}
		sorting_id_mapping["2"] = 1
		sorting_id_mapping["3"] = 2
		lin_num = 4

		with self.assertRaises(eo.MyException):
			cnas = oio.read_cnas(input_name, sorting_id_mapping=sorting_id_mapping, use_cna_indices=True, lin_num=lin_num)

		# sorting_id_mapping but key is wrong
		input_name = "testdata/unittests/cnas4_wrong.csv"
		sorting_id_mapping = {}
		sorting_id_mapping["2"] = 1
		sorting_id_mapping["3"] = 2
		lin_num = 4

		with self.assertRaises(eo.MyException):
			cnas = oio.read_cnas(input_name, sorting_id_mapping=sorting_id_mapping, use_cna_indices=True, lin_num=lin_num)

	def test_read_frequencies(self):
		# ex1_frequencies.csv
		input_name = "submarine_example/ex1_frequencies.csv"
		freqs = oio.read_frequencies(input_name)

		self.assertEqual([[0.8, 0.75], [0.7, 0.15], [0.1, 0.6]], freqs)

		# frequencies.csv, ordering given, no IDs
		input_name = "testdata/unittests/frequencies.csv"
		freqs = oio.read_frequencies(input_name)

		self.assertEqual([[0.8, 0.75], [0.7, 0.15], [0.1, 0.6]], freqs)

		# frequencies.csv, ordering not given, no IDs
		input_name = "testdata/unittests/frequencies.csv"
		freqs = oio.read_frequencies(input_name, ordering_given=False)

		self.assertEqual([[0.8, 0.75], [0.1, 0.6], [0.7, 0.15]], freqs)

		# frequencies.csv, ordering not given, with IDs
		input_name = "testdata/unittests/frequencies.csv"
		freqs, ids = oio.read_frequencies(input_name, ordering_given=False, return_ids=True)

		self.assertEqual([[0.8, 0.75], [0.1, 0.6], [0.7, 0.15]], freqs)
		self.assertEqual(["1", "3", "2"], ids)
	
	def test_read_parent_vector(self):
		input_name = "submarine_example/ex1_parents.csv"
		parents = oio.read_parent_vector(input_name)

		self.assertEqual([0, 1, 1], parents)


		input_name = "testdata/unittests/parents.csv"
		parents = oio.read_parent_vector(input_name)

		self.assertEqual([0, 1, 2, 3], parents)

	def test_read_result_file(self):
		file_name = "testdata/unittests/out_result1"

		ll = oio.read_result_file(file_name, phasing_not_known=False)

		# test if list is correct

		# should have 4 lineages
		self.assertEqual(len(ll), 4)

		# frequencies
		self.assertEqual(ll[0].freq, 1.0)
		self.assertEqual(ll[1].freq, 0.9)

		# sublineages
		self.assertEqual(ll[0].sublins, [1, 2, 3])
		self.assertEqual(ll[1].sublins, [2, 3])
		self.assertEqual(ll[3].sublins, [])

		# snps
		self.assertEqual(len(ll[0].snps), 0)
		self.assertEqual(len(ll[0].snps_a), 5)
		self.assertEqual(len(ll[0].snps_b), 5)
		self.assertEqual(len(ll[1].snps_b), 0)
		self.assertEqual(ll[0].snps_a[0].seg_index, 0)
		self.assertEqual(ll[0].snps_a[0].chr, 1)
		self.assertEqual(ll[0].snps_a[0].pos, 0)

		# ssms
		self.assertEqual(len(ll[1].ssms), 0)
		self.assertEqual(len(ll[1].ssms_a), 4)
		self.assertEqual(ll[1].ssms_a[0].phase, cons.A)
		self.assertEqual(ll[1].ssms_a[0].lineage, 1)
		self.assertEqual(len(ll[3].ssms_b), 1)
		self.assertEqual(ll[3].ssms_b[0].phase, cons.B)
		self.assertEqual(ll[3].ssms_b[0].lineage, 3)

		# cnvs
		self.assertEqual(len(ll[1].cnvs_a), 1)
		self.assertEqual(len(ll[1].cnvs_b), 1)
		self.assertEqual(ll[1].cnvs_a[0].chr, 1)
		self.assertEqual(ll[1].cnvs_a[0].start, 0)
		self.assertEqual(ll[1].cnvs_a[0].change, 1)
		self.assertEqual(ll[1].cnvs_a[0].phase, cons.A)

		# read without phasing information
		ll = oio.read_result_file(file_name, phasing_not_known=True)
		self.assertEqual(ll[1].ssms_a[0].phase, None)

	def test_write_result_file(self):
		output_file = "testdata/unittests/out_result3"
		reference_file = "testdata/unittests/out_result3_reference"

		#create test_lineages, one empty(None), one only contains empty lists and the other with entries
		cnvs_a = [cnv.CNV(+1, 0, 1, 0, 7), cnv.CNV(-1, 2, 1, 12, 16)]
		cnvs_b = [cnv.CNV(-1, 1, 1, 8, 11)]
		snps = [snp_ssm.SNP_SSM(), snp_ssm.SNP_SSM()]
		snps[0].chr = 1
		snps[0].pos = 2
		snps[0].seg_index = 0
		snps[1].chr = 1
		snps[1].pos = 4
		snps[1].seg_index = 0
		snps_a = [snp_ssm.SNP_SSM(), snp_ssm.SNP_SSM(), snp_ssm.SNP_SSM()]
		snps_a[0].chr = 1
		snps_a[0].pos = 0
		snps_a[0].seg_index = 0
		snps_a[1].chr = 1
		snps_a[1].pos = 13
		snps_a[1].seg_index = 2
		snps_a[2].chr = 1
		snps_a[2].pos = 14
		snps_a[2].seg_index = 2
		snps_b = [snp_ssm.SNP_SSM()]
		snps_b[0].chr = 1
		snps_b[0].pos = 8
		snps_b[0].seg_index = 1
		ssms = snps_b[:]
		ssms_a = snps[:]
		ssms_b = snps_a[:]
		lineages = [lineage.Lineage(None, None, None, None, None, None, None, None, None, None), 
			lineage.Lineage([], 0.0, [], [], [], [], [], [], [], []), 
			lineage.Lineage([0,1], 0.2, cnvs_a, cnvs_b, snps, snps_a, snps_b, ssms, ssms_a, ssms_b)]

		oio.write_result_file(lineages, output_file, test=True)

		with open(output_file, 'r') as f:
			file_data = f.read()
		with open(reference_file, 'r') as f:
			reference_data = f.read()

		self.assertEqual(file_data, reference_data)


	def test_str_to_bool(self):
		s = 'True'
		self.assertTrue(oio.str_to_bool(s))

		s = 'False'
		self.assertFalse(oio.str_to_bool(s))

		s = 'something'
		with self.assertRaises(ValueError):
			oio.str_to_bool(s)


def suite():
	return unittest.TestLoader().loadTestsFromTestCase(OnctopusIOTest)

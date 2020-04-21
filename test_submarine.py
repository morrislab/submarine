import unittest
import submarine
import snp_ssm
import segment
import cnv
import exceptions_onctopus as eo
import io_file as oio
import numpy as np
import lineage
import constants as cons
import numpy as np
import lineage
import copy

class ModelTest(unittest.TestCase):

	def test_compute_minimal_noise_threshold(self):

		k = 3
		linFreqs = np.asarray([[1, 1], [0.9, 0.8], [0.8, 0.7], [0.4, 0.75]])
		avFreqs_from_initial_pps = np.asarray([[0.1, 0.2], [0.1, 0.1]])

		self.assertEqual(submarine.compute_minimal_noise_threshold(k, linFreqs, avFreqs_from_initial_pps), 0.55)

		# other example
		k = 3
		linFreqs = np.asarray([[1, 1], [0.9, 0.8], [0.8, 0.7], [0.7, 0.75]])
		avFreqs_from_initial_pps = np.asarray([[0.1, 0.2], [0.1, 0.1]])

		self.assertEqual(submarine.compute_minimal_noise_threshold(k, linFreqs, avFreqs_from_initial_pps), 0.6)

		# other example
		k = 3
		linFreqs = np.asarray([[1, 1], [0.9, 0.8], [0.6, 0.7], [0.7, 0.3]])
		avFreqs_from_initial_pps = np.asarray([[0.1, 0.2], [0.3, 0.1]])

		self.assertAlmostEqual(submarine.compute_minimal_noise_threshold(k, linFreqs, avFreqs_from_initial_pps), 0.4)


	def test_get_possible_parents_from_ppmatrix(self):

		ppm = np.zeros(25).reshape(5,5)
		ppm[1][0] = 1
		ppm[2][0] = 1
		ppm[2][1] = 1
		ppm[3][2] = 1
		ppm[4][1] = 1
		ppm[4][3] = 1

		pp_mapping = submarine.get_possible_parents_from_ppmatrix(ppm)

		self.assertEqual(pp_mapping[1], [0])
		self.assertTrue((pp_mapping[2] == [0, 1]).all())
		self.assertEqual(pp_mapping[3], [2])
		self.assertTrue((pp_mapping[4] == [1, 3]).all())

	def test_still_possible_parents_except_freq(self):

		# 1) easy example, 3 lineages, no mutations, both pp's are still possible
		k = 2
		initial_pps_for_all = [[], [], [0, 1]]
		z_matrix = [[-1, 1, 1], [-1, -1, -1], [-1, -1, -1]]
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, 3*3, 3)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=None,
			matrix_after_first_round=copy.deepcopy(z_matrix))
		seg_num = 1
		gain_num = 0
		loss_num = 0
		CNVs = []
		present_ssms = []

		submarine.still_possible_parents_except_freq(k, initial_pps_for_all, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertEqual(initial_pps_for_all[2], [0, 1])

		# 2) easy example, 4 lineages, no mutations, both pp's are still possible
		k = 3
		initial_pps_for_all = [[], [], [], [1, 2]]
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1]]
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, 4*4, 4)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=None,
			matrix_after_first_round=copy.deepcopy(z_matrix))
		seg_num = 1
		gain_num = 0
		loss_num = 0
		CNVs = []
		present_ssms = []

		submarine.still_possible_parents_except_freq(k, initial_pps_for_all, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertEqual(initial_pps_for_all[3], [1, 2])

		# 3) 4 lineages, mutations, only 0 can stay possible parent of 3
		k = 3
		initial_pps_for_all = [[], [], [], [0, 1, 2]]
		seg_num = 1
		gain_num = [0]
		loss_num = [2]
		CNVs_0 = {}
		CNVs_0[cons.LOSS] = {}
		CNVs_0[cons.LOSS][cons.A] = {}
		CNVs_0[cons.LOSS][cons.A][2] = "something"
		CNVs_0[cons.LOSS][cons.B] = {}
		CNVs_0[cons.LOSS][cons.B][3] = "something"
		CNVs = [CNVs_0]
		present_ssms = [[[False, False, False, False], [False, True, False, False], [False, False, False, False]]]
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, 4*4, 4)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=present_ssms,
			matrix_after_first_round=copy.deepcopy(z_matrix))

		submarine.still_possible_parents_except_freq(k, initial_pps_for_all, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertEqual(initial_pps_for_all[3], [0])

		# 4) 4 lineages, mutations, only 0 and 1 can stay possible parent of 3
		k = 3
		initial_pps_for_all = [[], [], [], [0, 1, 2]]
		seg_num = 1
		gain_num = [0]
		loss_num = [2]
		CNVs_0 = {}
		CNVs_0[cons.LOSS] = {}
		CNVs_0[cons.LOSS][cons.A] = {}
		CNVs_0[cons.LOSS][cons.A][2] = "something"
		CNVs_0[cons.LOSS][cons.A] = {}
		CNVs_0[cons.LOSS][cons.A][3] = "something"
		CNVs = [CNVs_0]
		present_ssms = [[[False, False, False, False], [False, True, False, False], [False, False, False, False]]]
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, 4*4, 4)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=present_ssms,
			matrix_after_first_round=copy.deepcopy(z_matrix))

		submarine.still_possible_parents_except_freq(k, initial_pps_for_all, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertEqual(initial_pps_for_all[3], [0, 1])

	def test_go_extended_version(self):

		freq_file = "submarine_example/frequencies3.csv"
		cna_file = "submarine_example/cnas3.csv"
		ssm_file = "submarine_example/ssms3.csv"
		impact_file = "submarine_example/impact3.csv"
		userZ_file = "submarine_example/userZ_3.csv"
		userSSM_file = "submarine_example/userSSM3.csv"
		output_prefix = "submarine_example/test_extended_version"
		overwrite = True
		use_logging = True

		z_matrix = [[0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0]]
		true_ssm_phasing = [[0, cons.A], [1, cons.B], [2, cons.B], [3, cons.A]]
		ssm0 = snp_ssm.SSM()
		ssm0.seg_index = 0
		ssm0.phase = cons.A
		ssm0.lineage = 1
		ssm0.index = 0
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		ssm1.phase = cons.B
		ssm1.lineage = 1
		ssm1.index = 1
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 1
		ssm2.phase = cons.B
		ssm2.lineage = 1
		ssm2.index = 2
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 2
		ssm3.phase = cons.A
		ssm3.lineage = 4
		ssm3.index = 3
		cnv0 = cnv.CNV(1, 0, -1, -1, -1)
		cnv0.phase = cons.A
		cnv0.lineage = 3
		cnv0.index = 0
		cnv1 = cnv.CNV(-1, 1, -1, -1, -1)
		cnv1.phase = cons.A
		cnv1.lineage = 3
		cnv1.index = 1
		cnv2 = cnv.CNV(-1, 2, -1, -1, -1)
		cnv2.phase = cons.B
		cnv2.lineage = 2
		cnv2.index = 2
		cnv3 = cnv.CNV(-1, 2, -1, -1, -1)
		cnv3.phase = cons.B
		cnv3.lineage = 4
		cnv3.index = 3
		lin0 = lineage.Lineage([1, 2, 3, 4, 5], [1.0, 1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([3, 4], [0.5, 0.5], [], [], [], [], [], [], [ssm0], [ssm1, ssm2])
		lin2 = lineage.Lineage([5], [0.49, 0.49], [], [cnv2], [], [], [], [], [], [])
		lin3 = lineage.Lineage([4], [0.48, 0.48], [cnv0, cnv1], [], [], [], [], [], [], [])
		lin4 = lineage.Lineage([], [0.47, 0.46], [], [cnv3], [], [], [], [], [ssm3], [])
		lin5 = lineage.Lineage([], [0.4, 0.47], [], [], [], [], [], [], [], [])
		true_my_lins = [lin0, lin1, lin2, lin3, lin4, lin5]
		true_avFreqs = np.asarray([[0.01, 0.01], [0.02, 0.02], [0.09, 0.02], [0.01, 0.02], [0.47, 0.46], [0.4, 0.47]])
		true_ppm = np.asarray([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0],
			[0, 0, 1, 0, 0, 0]])

		my_lins, z_matrix_for_output, avFreqs, ppm, ssm_phasing = submarine.go_extended_version(freq_file=freq_file, 
			cna_file=cna_file, ssm_file=ssm_file, impact_file=impact_file, userZ_file=userZ_file, 
			userSSM_file=userSSM_file, output_prefix=output_prefix, overwrite=overwrite, use_logging=use_logging)

		self.assertEqual(z_matrix, z_matrix_for_output)
		self.assertEqual(true_ssm_phasing, ssm_phasing)
		self.assertTrue((true_ppm == ppm).all())
		self.assertTrue(np.isclose(true_avFreqs, avFreqs).all())
		self.assertEqual(true_my_lins, my_lins)

		# example #5) that tests working with noise, without binary search
		freq_file = "testdata/unittests/frequencies5.csv"
		cna_file = "testdata/unittests/cnas6.csv"
		ssm_file = "testdata/unittests/ssms5.csv"
		impact_file = "testdata/unittests/impact2.csv"

		ppm_true = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]])

		my_lins, z_matrix_for_output, avFreqs, ppm, ssm_phasing = submarine.go_extended_version(freq_file=freq_file, 
			cna_file=cna_file, ssm_file=ssm_file, impact_file=impact_file, 
			allow_noise=True, do_binary_search=False)

		self.assertTrue((ppm == ppm_true).all())
		self.assertEqual(ssm_phasing[0], [0, None])

		# example #6) that tests working with noise, using binary search
		freq_file = "testdata/unittests/frequencies5.csv"
		cna_file = "testdata/unittests/cnas6.csv"
		ssm_file = "testdata/unittests/ssms5.csv"
		impact_file = "testdata/unittests/impact2.csv"

		ppm_true = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]])

		my_lins, z_matrix_for_output, avFreqs, ppm, ssm_phasing = submarine.go_extended_version(freq_file=freq_file, 
			cna_file=cna_file, ssm_file=ssm_file, impact_file=impact_file, 
			allow_noise=True, do_binary_search=True)

		self.assertTrue((ppm == ppm_true).all())
		self.assertEqual(ssm_phasing[0], [0, None])

	def test_check_lost_alleles_for_basic(self):
		# works
		del_segments = [1, 3, 4]
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 2
		ssm1.index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 5
		ssm2.index = 1
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 10
		ssm3.index = 2
		my_ssms = [ssm1, ssm2, ssm3]

		self.assertTrue(submarine.check_lost_alleles_for_basic(del_segments, my_ssms))

		# doesn't work
		del_segments = [1, 3, 4]
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 2
		ssm1.index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 6
		ssm2.index = 1
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 3
		ssm3.index = 2
		my_ssms = [ssm1, ssm2, ssm3]

		with self.assertRaises(eo.MyException):
			submarine.check_lost_alleles_for_basic(del_segments, my_ssms)

	def test_get_deleted_segments(self):

		cna1 = cnv.CNV(1, 0, 0, 1, 10)
		cna1.phase = cons.A
		cna1.lineage = 1
		cna2 = cnv.CNV(-1, 3, 3, 1, 10)
		cna2.phase = cons.A
		cna2.lineage = 1
		cna3 = cnv.CNV(-1, 3, 3, 1, 10)
		cna3.phase = cons.B
		cna3.lineage = 1
		cna4 = cnv.CNV(-1, 5, 3, 1, 10)
		cna4.phase = cons.A
		cna4.lineage = 1
		cna5 = cnv.CNV(-1, 5, 3, 1, 10)
		cna5.phase = cons.B
		cna5.lineage = 1
		my_cnas = [cna2, cna1, cna4, cna3, cna5]

		self.assertEqual([3, 5], submarine.get_deleted_segments(my_cnas))
		

	def test_check_all_clonal(self):

		# all CNAs are clonal
		cna1 = cnv.CNV(1, 0, 0, 1, 10)
		cna1.phase = cons.A
		cna1.lineage = 1
		cna2 = cnv.CNV(1, 3, 3, 1, 10)
		cna2.phase = cons.A
		cna2.lineage = 1
		my_cnas = [cna1, cna2]

		self.assertTrue(submarine.check_all_clonal(my_cnas))

		# not all CNAs are clonal
		cna1 = cnv.CNV(1, 0, 0, 1, 10)
		cna1.phase = cons.A
		cna1.lineage = 1
		cna2 = cnv.CNV(1, 3, 3, 1, 10)
		cna2.phase = cons.A
		cna2.lineage = 2
		my_cnas = [cna1, cna2]

		with self.assertRaises(eo.MyException):
			submarine.check_all_clonal(my_cnas)


	def test_check_monotonicity(self):

		# A has gain and loss
		cna1 = cnv.CNV(1, 0, 0, 1, 10)
		cna1.phase = cons.A
		cna2 = cnv.CNV(1, 3, 3, 1, 10)
		cna2.phase = cons.A
		cna3 = cnv.CNV(1, 0, 0, 1, 10)
		cna3.phase = cons.B
		cna4 = cnv.CNV(-1, 0, 0, 1, 10)
		cna4.phase = cons.A
		my_cnas = [cna1, cna2, cna3, cna4]

		with self.assertRaises(eo.MyException):
			submarine.check_monotonicity(my_cnas)

		# B has loss and gain
		cna1 = cnv.CNV(-1, 0, 0, 1, 10)
		cna1.phase = cons.B
		cna2 = cnv.CNV(1, 3, 3, 1, 10)
		cna2.phase = cons.A
		cna3 = cnv.CNV(1, 0, 0, 1, 10)
		cna3.phase = cons.A
		cna4 = cnv.CNV(1, 0, 0, 1, 10)
		cna4.phase = cons.B
		my_cnas = [cna1, cna2, cna3, cna4]

		with self.assertRaises(eo.MyException):
			submarine.check_monotonicity(my_cnas)

		# everything works fine
		cna1 = cnv.CNV(-1, 0, 0, 1, 10)
		cna1.phase = cons.B
		cna2 = cnv.CNV(1, 3, 3, 1, 10)
		cna2.phase = cons.A
		cna3 = cnv.CNV(1, 0, 0, 1, 10)
		cna3.phase = cons.A
		cna4 = cnv.CNV(-1, 0, 0, 1, 10)
		cna4.phase = cons.B
		my_cnas = [cna1, cna2, cna3, cna4]

		self.assertTrue(submarine.check_monotonicity(my_cnas))

	def test_get_new_ssm_phasing(self):

		ssm1 = snp_ssm.SSM()
		ssm1.index = 0
		ssm1.phase = cons.A
		ssm2 = snp_ssm.SSM()
		ssm2.index = 1
		ssm2.phase = cons.A
		ssm3 = snp_ssm.SSM()
		ssm3.index = 2
		ssm3.phase = cons.A
		ssm4 = snp_ssm.SSM()
		ssm4.index = 3
		ssm4.phase = cons.B
		lin0 = lineage.Lineage([1, 2, 3], [1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([2, 3], [1.0], [], [], [], [], [], [ssm4], [], [])
		lin2 = lineage.Lineage([3], [1.0], [], [], [], [], [], [ssm2], [], [ssm1])
		lin3 = lineage.Lineage([], [1.0], [], [], [], [], [], [], [ssm3], [])
		my_lins = [lin0, lin1, lin2, lin3]

		ssm_list = [[0, cons.A], [1, cons.A], [2, cons.A], [3, cons.B]]

		self.assertEqual(submarine.get_new_ssm_phasing(my_lins), ssm_list)

	def test_get_seg_num(self):

		# no mutations given
		my_cnas = []
		my_ssms = []
		self.assertEqual(submarine.get_seg_num(my_cnas, my_ssms), 1)

		# CNA has highest index
		cna1 = cnv.CNV(1, 0, 0, 1, 10)
		cna1.phase = cons.A
		cna2 = cnv.CNV(1, 3, 3, 1, 10)
		cna2.phase = cons.A
		my_cnas = [cna1, cna2]
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 1
		my_ssms = [ssm1, ssm2]

		self.assertEqual(submarine.get_seg_num(my_cnas, my_ssms), 4)
		
		# SSM has highest index
		cna1 = cnv.CNV(1, 0, 0, 1, 10)
		cna1.phase = cons.A
		cna2 = cnv.CNV(1, 3, 3, 1, 10)
		cna2.phase = cons.A
		my_cnas = [cna1, cna2]
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 10
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 1
		my_ssms = [ssm1, ssm2]

		self.assertEqual(submarine.get_seg_num(my_cnas, my_ssms), 11)

	def test_add_CNAs(self):

		# working example
		cna1 = cnv.CNV(1, 4, 0, 1, 10)
		cna1.phase = cons.A
		cna1.lineage = 1
		cna2 = cnv.CNV(1, 3, 3, 1, 10)
		cna2.phase = cons.A
		cna2.lineage = 3
		cna3 = cnv.CNV(1, 3, 0, 1, 10)
		cna3.phase = cons.A
		cna3.lineage = 1
		cna4 = cnv.CNV(1, 5, 3, 1, 10)
		cna4.phase = cons.B
		cna4.lineage = 3
		cna5 = cnv.CNV(1, 0, 0, 1, 10)
		cna5.phase = cons.B
		cna5.lineage = 3
		my_cnas = [cna1, cna2, cna3, cna4, cna5]
		lin0 = lineage.Lineage([1, 2, 3], [1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([2, 3], [1.0], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([3], [1.0], [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [1.0], [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]

		submarine.add_CNAs(my_lins=my_lins, lin_num=4, my_cnas=my_cnas)

		self.assertEqual(my_lins[1].cnvs_a[0].seg_index, 3)
		self.assertEqual(my_lins[1].cnvs_a[1].seg_index, 4)
		self.assertEqual(my_lins[3].cnvs_a[0].seg_index, 3)
		self.assertEqual(my_lins[3].cnvs_b[0].seg_index, 0)
		self.assertEqual(my_lins[3].cnvs_b[1].seg_index, 5)

		# working example
		cna1 = cnv.CNV(1, 4, 0, 1, 10)
		cna1.phase = cons.A
		cna1.lineage = 1
		cna2 = cnv.CNV(1, 4, 3, 1, 10)
		cna2.phase = cons.A
		cna2.lineage = 1
		my_cnas = [cna1, cna2]
		lin0 = lineage.Lineage([1, 2, 3], [1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([2, 3], [1.0], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([3], [1.0], [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [1.0], [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]

		with self.assertRaises(eo.MyException):
			submarine.add_CNAs(my_lins=my_lins, lin_num=4, my_cnas=my_cnas)

	def test_add_SSMs(self):

		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 10
		ssm1.lineage = 3
		ssm1.phase = cons.A
		ssm2 = snp_ssm.SSM()
		ssm2.seg_index = 1
		ssm2.lineage = 3
		ssm2.phase = cons.A
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 1
		ssm3.lineage = 1
		ssm3.phase = cons.B
		ssm4 = snp_ssm.SSM()
		ssm4.seg_index = 0
		ssm4.lineage = 1
		ssm4.phase = cons.B
		ssm5 = snp_ssm.SSM()
		ssm5.seg_index = 2
		ssm5.lineage = 2
		my_ssms = [ssm1, ssm2, ssm3, ssm4, ssm5]
		lin0 = lineage.Lineage([1, 2, 3], [1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([2, 3], [1.0], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([3], [1.0], [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [1.0], [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]

		submarine.add_SSMs(my_lins=my_lins, lin_num=4, my_ssms=my_ssms)

		self.assertEqual(my_lins[1].ssms_b[0].seg_index, 0)
		self.assertEqual(my_lins[1].ssms_b[1].seg_index, 1)
		self.assertEqual(my_lins[3].ssms_a[0].seg_index, 1)
		self.assertEqual(my_lins[3].ssms_a[1].seg_index, 10)
		self.assertEqual(my_lins[2].ssms[0].seg_index, 2)

	def test_change_ssm_phasing_according_to_impact(self):

		# 1) impact matrix is empty		
		impact_matrix = None

		self.assertTrue(submarine.change_ssm_phasing_according_to_impact(impact_matrix=impact_matrix))

		# 2) SSM not impacted by any CNA
		impact_matrix = np.zeros(10).reshape(1, 10)
		ssm1 = snp_ssm.SSM()
		ssm1.index = 0
		my_ssms = [ssm1]

		self.assertTrue(submarine.change_ssm_phasing_according_to_impact(impact_matrix=impact_matrix, my_ssms=my_ssms))

		# 3) SSM impacted but on other segment, ERROR
		impact_matrix = np.zeros(10).reshape(1, 10)
		impact_matrix[0][0] = 1
		ssm1 = snp_ssm.SSM()
		ssm1.index = 0
		ssm1.seg_index = 0
		my_ssms = [ssm1]
		cna1 = cnv.CNV(1, 4, 0, 1, 10)
		my_cnas = [cna1]
		with self.assertRaises(eo.MyException):
			submarine.change_ssm_phasing_according_to_impact(impact_matrix=impact_matrix, my_ssms=my_ssms,
				my_cnas=my_cnas)

		# 4/5) SSM impacted, prev. unphased, now new phase
		# SSM impacted, prev. phased to correct phase
		impact_matrix = np.zeros(10).reshape(2, 5)
		impact_matrix[0][0] = 1
		impact_matrix[1][1] = 1
		impact_matrix[1][3] = 1
		ssm1 = snp_ssm.SSM()
		ssm1.index = 0
		ssm1.seg_index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.index = 1
		ssm2.seg_index = 1
		my_ssms = [ssm1, ssm2]
		cna1 = cnv.CNV(1, 0, 0, 1, 10)
		cna1.index = 0
		cna1.phase = cons.A
		cna6 = cnv.CNV(1, 1, 0, 1, 10)
		cna6.index = 6
		cna6.phase = cons.B
		cna7 = cnv.CNV(1, 1, 0, 1, 10)
		cna8 = cnv.CNV(1, 1, 0, 1, 10)
		cna8.index = 8
		cna8.phase = cons.B
		cna9 = cnv.CNV(1, 1, 0, 1, 10)
		my_cnas = [cna1, cna6, cna7, cna8, cna9]

		submarine.change_ssm_phasing_according_to_impact(impact_matrix=impact_matrix, my_ssms=my_ssms, my_cnas=my_cnas)

		self.assertEqual(ssm1.phase, cons.A)
		self.assertEqual(ssm2.phase, cons.B)

		# 6) SSM impacted, prev. phased to other phase, ERROR
		impact_matrix = np.zeros(10).reshape(2, 5)
		impact_matrix[0][0] = 1
		impact_matrix[1][1] = 1
		impact_matrix[1][3] = 1
		ssm1 = snp_ssm.SSM()
		ssm1.index = 0
		ssm1.seg_index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.index = 1
		ssm2.seg_index = 1
		my_ssms = [ssm1, ssm2]
		cna1 = cnv.CNV(1, 0, 0, 1, 10)
		cna1.index = 0
		cna1.phase = cons.A
		cna6 = cnv.CNV(1, 1, 0, 1, 10)
		cna6.index = 6
		cna6.phase = cons.B
		cna7 = cnv.CNV(1, 1, 0, 1, 10)
		cna8 = cnv.CNV(1, 1, 0, 1, 10)
		cna8.index = 8
		cna8.phase = cons.A
		cna9 = cnv.CNV(1, 1, 0, 1, 10)
		my_cnas = [cna1, cna6, cna7, cna8, cna9]
		
		with self.assertRaises(eo.MyException):
			submarine.change_ssm_phasing_according_to_impact(impact_matrix=impact_matrix, my_ssms=my_ssms, my_cnas=my_cnas)

	def test_update_ADRs_according_impact_matrix(self):

		# doesn't work because lineage of SSM cannot be ancestor of CNA
		z_matrix = [[-1, 1, 1], [-1, -1, 0], [-1, -1, -1]]
		impact_matrix = np.ones(1).reshape(1,1)
		ssm1 = snp_ssm.SSM()
		ssm1.lineage = 2
		my_ssms = [ssm1]
		cna1 = cnv.CNV(1, 0, 0, 1, 10)
		cna1.lineage = 1
		my_cnas = [cna1]
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
				z_matrix, 3*3, 3)

		with self.assertRaises(eo.MyException):
			submarine.update_ADRs_according_impact_matrix(z_matrix=z_matrix, triplet_xys=triplet_xys, 
				triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, impact_matrix=impact_matrix, 
				my_ssms=my_ssms, my_cnas=my_cnas)

		# works
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, 1], [-1, -1, -1, -1]]
		impact_matrix = np.ones(1).reshape(1,1)
		impact_matrix = np.zeros(3*2).reshape(3,2)
		impact_matrix[2][1] = 1
		ssm1 = snp_ssm.SSM()
		ssm1.lineage = 1
		ssm2 = snp_ssm.SSM()
		ssm2.lineage = 1
		ssm3 = snp_ssm.SSM()
		ssm3.lineage = 1
		my_ssms = [ssm1, ssm2, ssm3]
		cna1 = cnv.CNV(1, 0, 0, 1, 10)
		cna1.lineage = 2
		cna2 = cnv.CNV(1, 0, 0, 1, 10)
		cna2.lineage = 2
		my_cnas = [cna1, cna2]
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
				z_matrix, 4*4, 4)

		submarine.update_ADRs_according_impact_matrix(z_matrix=z_matrix, triplet_xys=triplet_xys, 
			triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, impact_matrix=impact_matrix, 
			my_ssms=my_ssms, my_cnas=my_cnas)

		self.assertEqual(1, z_matrix[1][2])
		self.assertEqual(1, z_matrix[1][3])

		# update fails because relation is already set
		z_matrix = [[-1, 1, 1, 1], [-1, -1, -1, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		impact_matrix = np.ones(1).reshape(1,1)
		impact_matrix = np.zeros(3*2).reshape(3,2)
		impact_matrix[2][1] = 1
		ssm1 = snp_ssm.SSM()
		ssm1.lineage = 1
		ssm2 = snp_ssm.SSM()
		ssm2.lineage = 1
		ssm3 = snp_ssm.SSM()
		ssm3.lineage = 1
		my_ssms = [ssm1, ssm2, ssm3]
		cna1 = cnv.CNV(1, 0, 0, 1, 10)
		cna1.lineage = 2
		cna2 = cnv.CNV(1, 0, 0, 1, 10)
		cna2.lineage = 2
		my_cnas = [cna1, cna2]
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
				z_matrix, 4*4, 4)

		with self.assertRaises(eo.MyException):
			submarine.update_ADRs_according_impact_matrix(z_matrix=z_matrix, triplet_xys=triplet_xys, 
				triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, impact_matrix=impact_matrix, 
				my_ssms=my_ssms, my_cnas=my_cnas)

	def test_propagate_negative_SSM_phasing_according_to_impact_matrix(self):

		# error: eq. 14: different phase of SSM
		ssm1 = snp_ssm.SSM()
		ssm1.index = 0
		ssm1.lineage = 1
		ssm1.seg_index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.index = 1
		ssm2.lineage = 1
		ssm2.seg_index = 0
		ssm3 = snp_ssm.SSM()
		ssm3.index = 2
		ssm3.phase = cons.A
		ssm3.lineage = 1
		ssm3.seg_index = 0
		my_ssms = [ssm1, ssm2, ssm3]
		cna1 = cnv.CNV(1, 0, 0, 1, 10)
		cna1.index = 0
		cna1.lineage = 1
		cna1.seg_index = 10
		cna2 = cnv.CNV(1, 0, 0, 1, 10)
		cna2.index = 1
		cna2.phase = cons.A
		cna2.lineage = 2
		cna2.seg_index = 0
		my_cnas = [cna1, cna2]
		impact_matrix = np.zeros(3*2).reshape(3,2)
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 1, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		my_lins = None

		with self.assertRaises(eo.MyException):
			submarine.propagate_negative_SSM_phasing_according_to_impact_matrix(my_ssms, my_cnas, z_matrix, impact_matrix, 
				my_lins)

		# error: eq. 7: deletion in ancestor, wrong phase
		ssm1 = snp_ssm.SSM()
		ssm1.index = 0
		ssm1.lineage = 2
		ssm1.seg_index = 0
		ssm1.phase = cons.A
		my_ssms = [ssm1]
		cna1 = cnv.CNV(-1, 0, 0, 1, 10)
		cna1.index = 0
		cna1.lineage = 1
		cna1.seg_index = 0
		cna1.phase = cons.A
		my_cnas = [cna1]
		impact_matrix = np.zeros(1).reshape(1,1)
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 1, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		lin0 = lineage.Lineage([1, 2, 3], [1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], [0.5], [cna1], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.3], [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [0.2], [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]

		with self.assertRaises(eo.MyException):
			submarine.propagate_negative_SSM_phasing_according_to_impact_matrix(my_ssms, my_cnas, z_matrix, impact_matrix, 
				my_lins)

		# error: eq. 7: deletion in current lineage, wrong phase
		ssm1 = snp_ssm.SSM()
		ssm1.index = 0
		ssm1.lineage = 2
		ssm1.seg_index = 0
		ssm1.phase = cons.B
		my_ssms = [ssm1]
		cna1 = cnv.CNV(-1, 0, 0, 1, 10)
		cna1.index = 0
		cna1.lineage = 1
		cna1.seg_index = 0
		cna1.phase = cons.B
		my_cnas = [cna1]
		impact_matrix = np.zeros(1).reshape(1,1)
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 1, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		lin0 = lineage.Lineage([1, 2, 3], [1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], [0.5], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.3], [], [cna1], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [0.2], [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]

		with self.assertRaises(eo.MyException):
			submarine.propagate_negative_SSM_phasing_according_to_impact_matrix(my_ssms, my_cnas, z_matrix, impact_matrix, 
				my_lins)

		# working
		ssm1 = snp_ssm.SSM()
		ssm1.index = 0
		ssm1.lineage = 1
		ssm1.seg_index = 0
		ssm2 = snp_ssm.SSM()
		ssm2.index = 1
		ssm2.lineage = 1
		ssm2.seg_index = 0
		ssm3 = snp_ssm.SSM()
		ssm3.index = 2
		ssm3.lineage = 2
		ssm3.seg_index = 1
		my_ssms = [ssm1, ssm2, ssm3]
		cna1 = cnv.CNV(1, 0, 0, 1, 10)
		cna1.index = 0
		cna1.lineage = 2
		cna1.seg_index = 0
		cna1.phase = cons.B
		cna2 = cnv.CNV(-1, 0, 0, 1, 10)
		cna2.index = 1
		cna2.phase = cons.A
		cna2.lineage = 2
		cna2.seg_index = 1
		my_cnas = [cna1, cna2]
		impact_matrix = np.zeros(3*2).reshape(3,2)
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 1, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		lin0 = lineage.Lineage([1, 2, 3], [1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], [0.5], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.3], [cna2], [cna1], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [0.2], [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]

		submarine.propagate_negative_SSM_phasing_according_to_impact_matrix(my_ssms, my_cnas, z_matrix, impact_matrix, 
			my_lins)

		self.assertEqual(ssm1.phase, cons.A)
		self.assertEqual(ssm2.phase, cons.A)
		self.assertEqual(ssm3.phase, cons.B)



	def test_new_dfs(self):

		# testing correct iteration through all settings
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		lin0 = lineage.Lineage([1, 2, 3], [1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], [0.5], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.3], [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [0.2], [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2, lin3]
		seg_num = 1
		all_options = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1,], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]]

		all_options2 = submarine.new_dfs(z_matrix, my_lineages, seg_num, test_iteration=True)
		self.assertEqual(all_options, all_options2)

		# 4 lineages, no mutations, all valid Z-matrices possible
		# two Z-matrices are not possible because of tree rules, some values already get updated at earlier undefined entries
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		lin0 = lineage.Lineage([1, 2, 3], [1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], [0.5], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.3], [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [0.2], [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2, lin3]
		seg_num = 1

		reconstructions, total_count, valid_count = submarine.new_dfs(z_matrix, my_lineages, seg_num, test_reconstructions=True)

		self.assertEqual(total_count, 8)
		self.assertEqual(valid_count, 6)
		self.assertEqual(len(reconstructions), 6)
		z_matrix_1 = np.asarray([[-1, 1, 1, 1], [-1, -1, 1, 1], [-1, -1, -1, 1], [-1, -1, -1, -1]])
		z_matrix_2 = np.asarray([[-1, 1, 1, 1], [-1, -1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
		z_matrix_3 = np.asarray([[-1, 1, 1, 1], [-1, -1, 1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
		z_matrix_4 = np.asarray([[-1, 1, 1, 1], [-1, -1, -1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
		z_matrix_5 = np.asarray([[-1, 1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, 1], [-1, -1, -1, -1]])
		z_matrix_6 = np.asarray([[-1, 1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
		self.assertTrue(np.array_equal(reconstructions[0].zmco.z_matrix, z_matrix_1))
		self.assertTrue(np.array_equal(reconstructions[1].zmco.z_matrix, z_matrix_2))
		self.assertTrue(np.array_equal(reconstructions[2].zmco.z_matrix, z_matrix_3))
		self.assertTrue(np.array_equal(reconstructions[3].zmco.z_matrix, z_matrix_4))
		self.assertTrue(np.array_equal(reconstructions[4].zmco.z_matrix, z_matrix_5))
		self.assertTrue(np.array_equal(reconstructions[5].zmco.z_matrix, z_matrix_6))

		# same input as before but this time only ambiguity analysis is tested
		total_count, valid_count, output = submarine.new_dfs(z_matrix, my_lineages, seg_num, analyze_ambiguity_during_runtime=True)

		self.assertEqual(output, "True\n")

		# 4 lineages, no mutations, only one valid Z-matrices possible
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, -1], [-1, -1, -1, 1], [-1, -1, -1, -1]]
		lin0 = lineage.Lineage([1, 2, 3], [1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], [0.5], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.3], [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [0.2], [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2, lin3]
		seg_num = 1

		total_count, valid_count, output = submarine.new_dfs(z_matrix, my_lineages, seg_num, analyze_ambiguity_during_runtime=True)

		self.assertEqual(output, "False, 1, 2, False, True\n")

		# 3 lineages, no mutations, two valid Z-matrices with noise threshold possible
		z_matrix = [[-1, 1, 1], [-1, -1, 0], [-1, -1, -1]]
		lin0 = lineage.Lineage([1, 2], [1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], [0.8], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.3], [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2]
		seg_num = 1

		total_count, valid_count, output = submarine.new_dfs(z_matrix, my_lineages, seg_num, analyze_ambiguity_during_runtime=True,
			noise_threshold=0.1)

		self.assertEqual(output, "True\n")
		self.assertEqual(total_count, 2)
		self.assertEqual(valid_count, 2)

		# more tests at test_compute_number_ambiguous_recs_and_new_dfs

	def test_create_ID_ordering_mapping(self):

		# positive example
		sorted_indices = [4, 2, 3, 1, 0]
		lin_ids = [0, 1, 2, 3, 4]

		mapping = submarine.create_ID_ordering_mapping(sorted_indices, lin_ids)

		self.assertEqual(mapping[4], 1)
		self.assertEqual(mapping[2], 2)
		self.assertEqual(mapping[3], 3)
		self.assertEqual(mapping[1], 4)
		self.assertEqual(mapping[0], 5)

		# negative example, same ID twice
		sorted_indices = [4, 2, 3, 1, 0]
		lin_ids = [0, 1, 2, 3, 3]

		with self.assertRaises(eo.MyException):
			mapping = submarine.create_ID_ordering_mapping(sorted_indices, lin_ids)

	def test_go_basic_version(self):
		# no user constraints, works
		freq_file = "testdata/unittests/frequencies2.csv"

		my_lins, z_matrix, avFreqs, ppm = submarine.go_basic_version(freq_file=freq_file, use_logging=False)

		real_z = [
			[0, 1, 1, 1, 1],
			[0, 0, 1, 1, 1],
			[0, 0, 0, 0, -1],
			[0, 0, 0, 0, -1],
			[0, 0, 0, 0, 0]
			]

		real_ppm = [
			[0, 0, 0, 0, 0],
			[1, 0, 0, 0, 0],
			[0, 1, 0, 0, 0],
			[0, 1, 0, 0, 0],
			[0, 0, 1, 1, 0],
			]

		self.assertEqual(z_matrix, real_z)
		self.assertTrue((real_ppm == ppm).all())
		self.assertEqual(my_lins[1].sublins, [2, 3, 4])
		self.assertEqual(my_lins[2].sublins, [])
		self.assertEqual(my_lins[3].sublins, [])

		# with user constraints, works
		freq_file = "testdata/unittests/frequencies2.csv"
		userZ_file = "testdata/unittests/userZ.csv"
		output_prefix = "testdata/unittests/out_result4"

		my_lins, z_matrix, avFreqs, ppm = submarine.go_basic_version(freq_file=freq_file, userZ_file=userZ_file, use_logging=True, 
			output_prefix=output_prefix, overwrite=True)

		real_z = [
			[0, 1, 1, 1, 1],
			[0, 0, 1, 1, 1],
			[0, 0, 0, 0, 1],
			[0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0]
			]

		real_ppm = [
			[0, 0, 0, 0, 0],
			[1, 0, 0, 0, 0],
			[0, 1, 0, 0, 0],
			[0, 1, 0, 0, 0],
			[0, 0, 1, 0, 0],
			]

		self.assertEqual(z_matrix, real_z)
		self.assertTrue((real_ppm == ppm).all())
		self.assertEqual(my_lins[1].sublins, [2, 3, 4])
		self.assertEqual(my_lins[2].sublins, [4])
		self.assertEqual(my_lins[3].sublins, [])

		# with user constraints, doesn't work
		freq_file = "testdata/unittests/frequencies2.csv"
		userZ_file = "testdata/unittests/userZ_2.csv"

		with self.assertRaises(eo.MyException):
			my_lins, z_matrix, avFreqs, ppm = submarine.go_basic_version(freq_file=freq_file, userZ_file=userZ_file)

		# no user constraints, doesn't work
		freq_file = "testdata/unittests/frequencies3.csv"

		error_message = submarine.go_basic_version(freq_file=freq_file)
		message = "There are no possible parents for subclone 4 with frequencies of 0.400,0.310, because subclone 0 has only available frequencies of 0.200,0.200, subclone 1 has only available frequencies of 0.000,0.000.\nCurrent tree with definite children: 0->1,1->2,1->3."
		self.assertEqual(message, error_message)

		# allows noise (#1)
		my_lins, z_matrix_for_output, avFreqs, ppm = submarine.go_basic_version(freq_file=freq_file, allow_noise=True)

		real_ppm = np.asarray([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0],])
		real_avFreqs = np.asarray([[-0.2, -0.11], [0, 0], [0.3, 0.5], [0.5, 0.3], [0.4, 0.31]])
		real_z_matrix_for_output = [[0, 1, 1, 1, 1], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

		self.assertTrue((ppm == real_ppm).all())
		self.assertTrue(np.isclose(avFreqs, real_avFreqs).all())
		self.assertEqual(z_matrix_for_output, real_z_matrix_for_output)

		# allows noise, given theshold is large enough (#2)
		my_lins, z_matrix_for_output, avFreqs, ppm = submarine.go_basic_version(freq_file=freq_file, allow_noise=True, noise_threshold=0.3, do_binary_search=False)

		real_ppm = np.asarray([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0],])
		real_avFreqs = np.asarray([[0.2, 0.2], [0.8, 0.8], [0.3, 0.5], [0.5, 0.3], [0.4, 0.31]])
		real_z_matrix_for_output = [[0, 1, 1, 1, 1], [0, 0, -1, -1, -1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

		self.assertTrue((ppm == real_ppm).all())
		self.assertTrue(np.isclose(avFreqs, real_avFreqs).all())
		self.assertEqual(z_matrix_for_output, real_z_matrix_for_output)

		# maximal noise threshold too small (#3)
		freq_file = "testdata/unittests/frequencies3.csv"

		error_message = submarine.go_basic_version(freq_file=freq_file, allow_noise=True, maximal_noise=0.1)
		self.assertEqual("There are no possible parents for subclone 4 with frequencies of 0.400,0.310, because subclone 0 has only available frequencies of 0.200,0.200, subclone 1 has only available frequencies of 0.000,0.000.\nCurrent tree with definite children: 0->1,1->2,1->3.", error_message)

		# allows noise, threshold found in second round (#4)
		freq_file = "testdata/unittests/frequencies4.csv"
		userZ_file = "testdata/unittests/userZ_3.csv"

		my_lins, z_matrix_for_output, avFreqs, ppm = submarine.go_basic_version(freq_file=freq_file, allow_noise=True, userZ_file=userZ_file)

		real_ppm = np.asarray([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
		real_avFreqs = np.asarray([[-0.4, 0.1], [-0.3, -0.1], [0.6, 0.5], [0.5, 0.6], [0.5, 0]])
		real_z_matrix_for_output = [[0, 1, 1, 1, 1], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]


	def test_get_lineages_from_freqs(self):

		# everything already sorted correctly
		freqs = [[1, 0.8], [0.8, 0.7], [0.9, 0.1]]
		freq_num = 2
		lin_num = 4
		lin_ids = [1, 2, 3]

		lins, mapping = submarine.get_lineages_from_freqs(freqs=freqs, freq_num=freq_num, 
			lin_num=lin_num, lin_ids=lin_ids)
		self.assertEqual(lins[0].freq, [1.0, 1.0])
		self.assertEqual(lins[1].freq, [1.0, 0.8])
		self.assertEqual(lins[2].freq, [0.8, 0.7])
		self.assertEqual(lins[3].freq, [0.9, 0.1])
		self.assertEqual(lins[0].sublins, [1, 2, 3])
		self.assertEqual(lins[1].sublins, [])
		self.assertEqual(lins[2].sublins, [])
		self.assertEqual(lins[3].sublins, [])


		# sorting needed
		freqs = [[1, 0.8], [0.8, 0.1], [0.9, 0.1], [1, 1]]
		freq_num = 2
		lin_num = 5
		lin_ids = [1, 2, 3, 4]

		lins, mapping = submarine.get_lineages_from_freqs(freqs=freqs, freq_num=freq_num, 
			lin_num=lin_num, lin_ids=lin_ids)
		self.assertEqual(lins[1].freq, [1.0, 1.0])
		self.assertEqual(lins[2].freq, [1.0, 0.8])
		self.assertEqual(lins[3].freq, [0.9, 0.1])
		self.assertEqual(lins[4].freq, [0.8, 0.1])


		# same average freqs but other individual frequencies
		freqs = [[0.8, 0.2, 1.0], [0.1, 0.2, 0.3], [0.5, 0.5, 1.0], [0.9, 0.8, 0.9],
			[1.0, 0.3, 0.7], [0.4, 0.2, 0.2]]
		freq_num = 3
		lin_num = 7
		lin_ids = [1, 2, 3, 4, 5, 6]
		lins, mapping = submarine.get_lineages_from_freqs(freqs=freqs, freq_num=freq_num, 
			lin_num=lin_num, lin_ids=lin_ids)

		self.assertEqual(lins[1].freq, [0.9, 0.8, 0.9])
		self.assertEqual(lins[4].freq, [0.8, 0.2, 1.0])
		self.assertEqual(lins[3].freq, [0.5, 0.5, 1.0])
		self.assertEqual(lins[2].freq, [1.0, 0.3, 0.7])
		self.assertEqual(lins[6].freq, [0.1, 0.2, 0.3])
		self.assertEqual(lins[5].freq, [0.4, 0.2, 0.2])


		# same frequencies
		freqs = [[1.0, 0.1], [0.8, 0.7], [1.0, 0.1]]
		freq_num = 2
		lin_num = 3
		lin_ids = [1, 2, 3]

		lins, mapping = submarine.get_lineages_from_freqs(freqs=freqs, 
			freq_num=freq_num, lin_num=lin_num, lin_ids=lin_ids)

		self.assertEqual(3, len(lins))


	def test_convert_zmatrix_for_internal_use(self):

		z_matrix = [[0, 1, 1, 1], [0, 0, "?", 1], [0, 0, 0, "?"], [0, 0, 0, 0]]
		new_z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 1], [-1, -1, -1, 0], [-1, -1, -1, -1]]

		submarine.convert_zmatrix_for_internal_use(z_matrix)
		self.assertEqual(z_matrix, new_z_matrix)

	def test_convert_zmatrix_0_m1(self):

		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 1], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		new_z_matrix = [[0, 1, 1, 1], [0, 0, -1, 1], [0, 0, 0, -1], [0, 0, 0, 0]]
		self.assertEqual(submarine.convert_zmatrix_0_m1(z_matrix), new_z_matrix)

	def test_count_ambiguous_relationships(self):

		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]

		self.assertEqual(3, submarine.count_ambiguous_relationships(z_matrix))

	def test_get_lineages_from_input_files(self):

		parents_file = "submarine_example/ex1_parents.csv"
		freq_file = "submarine_example/ex1_frequencies.csv"
		cna_file = "submarine_example/ex1_cnas.csv"
		ssm_file = "submarine_example/ex1_ssms.csv"

		my_lins = submarine.get_lineages_from_input_files(parents_file, freq_file, cna_file, ssm_file)

		self.assertEqual(len(my_lins), 4)
		self.assertEqual(len(my_lins[1].ssms), 1)
		self.assertEqual(len(my_lins[2].ssms_a), 1)
		self.assertEqual(len(my_lins[3].ssms_b), 1)
		self.assertEqual(len(my_lins[2].cnvs_a), 1)
		self.assertEqual(len(my_lins[3].cnvs_b), 1)
		self.assertEqual(my_lins[1].sublins, [2, 3])
		self.assertEqual(my_lins[2].freq, [0.7, 0.15])

	def test_check_and_update_complete_Z_matrix_from_matrix(self):
		
		# lin. 1 and lin. 2 are ancestors of lin. 3
		# lin. 1 has an SSM that should not get influenced by lin. 2
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 1], [-1, -1, -1, 1], [-1, -1, -1, -1]]
		z_matrix2 = [[-1, 1, 1, 1], [-1, -1, 0, 1], [-1, -1, -1, 1], [-1, -1, -1, -1]]
		zero_count = 1
		lineage_num = 4
		lin0 = lineage.Lineage([1, 2, 3], [1.0], [], [], [], [], [], [], [], [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 0
		ssm1.pos = 1
		ssm1.seg_index = 0
		lin1 = lineage.Lineage([3], [1.0], [], [], [], [], [], [ssm1], [], [])
		cnv1 = cnv.CNV(1, 0, 0, 0, 10)
		lin2 = lineage.Lineage([3], [1.0], [cnv1], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [1.0], [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2, lin3]
		gain_num = []
		loss_num = []
		CNVs = []
		present_ssms = []
		ssm_infl_cnv_same_lineage = []
		seg_num = 1
		submarine.get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
			ssm_infl_cnv_same_lineage)

		submarine.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, zero_count, lineage_num, CNVs, present_ssms, z_matrix2)

		self.assertEqual(present_ssms[0][cons.B][1], True)
		self.assertEqual(present_ssms[0][cons.UNPHASED][1], False)

	def test_update_ancestry_w_preprocessing(self):
		lin0 = lineage.Lineage([1, 2, 3], [1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], [0.7], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([3], [0.2], [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [0.1], [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2, lin3]
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, 1], [-1, -1, -1, -1]]
		ppm = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0]]
		seg_num = 1
		value = 1
		k = 1
		kprime = 2

		submarine.update_ancestry_w_preprocessing(my_lineages, z_matrix, ppm, seg_num, value, k, kprime)

		self.assertEqual(z_matrix[1][2], 1)
		self.assertEqual(z_matrix[1][3], 1)
		self.assertEqual(ppm[2][0], 0)
		

	def test_is_reconstruction_valid(self):

		# valid example
		lin0 = lineage.Lineage([1, 2, 3], [1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], [0.7], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([3], [0.2], [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [0.1], [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2, lin3]
		z_matrix = [[-1, 1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, 1], [-1, -1, -1, -1]]
		ppm = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]]
		seg_num = 1

		self.assertTrue(submarine.is_reconstruction_valid(my_lineages, z_matrix, ppm, seg_num))

		# invalid example, Z-matrix
		lin0 = lineage.Lineage([1, 2, 3], [1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([2,3], [0.7], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([3], [0.2], [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [0.1], [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2, lin3]
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 1, 0], [-1, -1, -1, 1], [-1, -1, -1, -1]]
		ppm = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
		seg_num = 1
		self.assertFalse(submarine.is_reconstruction_valid(my_lineages, z_matrix, ppm, seg_num))
		
		# different ppm
		ppm = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
		self.assertFalse(submarine.is_reconstruction_valid(my_lineages, z_matrix, ppm, seg_num))

		# valid example with noise
		lin0 = lineage.Lineage([1, 2], [1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], [0.7], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.7], [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2]
		z_matrix = [[-1, 1, 1], [-1, -1, -1], [-1, -1, -1]]
		ppm = [[0, 0, 0], [1, 0, 0], [1, 0, 0]]
		seg_num = 1

		self.assertTrue(submarine.is_reconstruction_valid(my_lineages, z_matrix, ppm, seg_num, noise_threshold=0.4))


	def test_get_definite_parents_available_frequencies(self):
		ppm = np.asarray([
			[0, 0, 0, 0, 0, 0],
			[1, 0, 0, 0, 0, 0],
			[1, 0, 0, 0, 0, 0],
			[0, 1, 1, 0, 0, 0],
			[0, 0, 1, 0, 0, 0],
			[1, 0, 1, 0, 0, 0]
			])
		freqs = np.asarray([[1, 1], [0.5, 0.51], [0.3, 0.31], [0.2, 0.19], [0.1, 0.1], [0.05, 0.07]])

		true_defp = [-1, 0, 0, -1, 2, -1]
		true_avFreqs = np.asarray([[0.2, 0.18], [0.5, 0.51], [0.2, 0.21], [0.2, 0.19], [0.1, 0.1], [0.05, 0.07]])

		defp, avFreqs = submarine.get_definite_parents_available_frequencies( freqs, ppm)

		self.assertTrue((defp == true_defp).all())
		self.assertTrue(np.allclose(avFreqs, true_avFreqs))
	

	def test_build_sublin_lists_from_parental_info(self):

		true_lin0 = lineage.Lineage([1, 2, 3, 4, 5, 6, 7], [], [], [], [], [], [], [], [], [])
		true_lin1 = lineage.Lineage([2, 3, 4, 5, 6, 7], [], [], [], [], [], [], [], [], [])
		true_lin2 = lineage.Lineage([3, 6, 7], [], [], [], [], [], [], [], [], [])
		true_lin3 = lineage.Lineage([], [], [], [], [], [], [], [], [], [])
		true_lin4 = lineage.Lineage([5], [], [], [], [], [], [], [], [], [])
		true_lin5 = lineage.Lineage([], [], [], [], [], [], [], [], [], [])
		true_lin6 = lineage.Lineage([7], [], [], [], [], [], [], [], [], [])
		true_lin7 = lineage.Lineage([], [], [], [], [], [], [], [], [], [])
		true_lins = [true_lin0, true_lin1, true_lin2, true_lin3, true_lin4, true_lin5, true_lin6, true_lin7]

		lin0 = lineage.Lineage([1, 2, 3, 4, 5, 6, 7], [], [], [], [], [], [], [], [], [])
		lin1= lineage.Lineage([], [], [], [], [], [], [], [], [], [])
		lin2= lineage.Lineage([], [], [], [], [], [], [], [], [], [])
		lin3= lineage.Lineage([], [], [], [], [], [], [], [], [], [])
		lin4= lineage.Lineage([], [], [], [], [], [], [], [], [], [])
		lin5= lineage.Lineage([], [], [], [], [], [], [], [], [], [])
		lin6= lineage.Lineage([], [], [], [], [], [], [], [], [], [])
		lin7= lineage.Lineage([], [], [], [], [], [], [], [], [], [])
		lins = [lin0, lin1, lin2, lin3, lin4, lin5, lin6, lin7]

		parental_info = [0, 1, 2, 1, 4, 2, 6]

		submarine.build_sublin_lists_from_parental_info(lins, parental_info)
		self.assertEqual(true_lins, lins)


	def test_get_lca(self):

		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, -1], [-1, -1, -1, 1],
			[-1, -1, -1, -1]]
		self.assertEqual(submarine.get_lca(2, 3, z_matrix), 2)
		self.assertEqual(submarine.get_lca(1, 3, z_matrix), 0)

	def test_get_lca_from_multiple_lineages(self):

		z_matrix = [[-1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, -1, -1, -1],
			[-1, -1, -1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, 1, 1, 0, 0],
			[-1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1],
			[-1, -1, -1, -1, -1, -1, -1, 1], [-1, -1, -1, -1, -1, -1, -1, -1]]
		self.assertEqual(submarine.get_lca_from_multiple_lineages([4, 5, 6, 7], z_matrix),
			2)

	def test_get_possible_ancestors(self):
		zmatrix = [[-1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, 1, -1], 
			[-1, -1, -1, -1, 0, 0], [-1, -1, -1, -1, 0, 1],
			[-1, -1, -1, -1, -1, 0], [-1, -1, -1, -1, -1, -1]]
		k = 5
		kstar = 4

		self.assertEqual(submarine.get_possible_ancestors(zmatrix, k, kstar).tolist(),
			[0, 2, 3])

	def test_get_possible_descendants(self):

		zmatrix = [[-1, 1, 1, 1], [-1, -1, 0, 1], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		self.assertEqual([1, 2], submarine.get_possible_descendants(zmatrix, 0, 3).tolist())
		self.assertEqual([2], submarine.get_possible_descendants(zmatrix, 1, 3).tolist())

		zmatrix = np.asarray(zmatrix)
		self.assertEqual([3], submarine.get_possible_descendants(zmatrix, 2, 4).tolist())
		self.assertEqual([], submarine.get_possible_descendants(zmatrix, 3, 4).tolist())

	def test_get_possible_children_smaller_last(self):

		defparent = [-1, 0, -1, -1]
		ppm = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]])
		self.assertEqual([2], submarine.get_possible_children_smaller_last(ppm, 0, 3, defparent))

	def test_get_possible_children_smaller_last_greater_kprime(self):

		defparent = [-1, 0, 1, -1, -1, -1]
		ppm = np.asarray([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0],
			[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]])
		self.assertEqual([3, 4], submarine.get_possible_children_smaller_last_greater_kprime(ppm, 1, 2, 5, defparent))

		defparent = [-1, 0, 1, 1, -1, -1]
		ppm = np.asarray([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
			[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]])
		self.assertEqual([4], submarine.get_possible_children_smaller_last_greater_kprime(ppm, 1, 2, 5, defparent))

	def test_check_untightess_zmatrix(self):
		# 1) threshold prevented building of all matrices
		z_matrix = [[-1, 1, 1], [-1, -1, 0], [-1, -1, -1]]
		z_matrix_list = [z_matrix]
		total_number = 2

		self.assertTrue("False" in submarine.check_untightess_zmatrix(z_matrix, z_matrix_list, total_number))

		# 2) ambiguous entries are only present, not absent
		z_matrix = [[-1, 1, 1], [-1, -1, 0], [-1, -1, -1]]
		z_matrix_list = [[[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]], [[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]]]
		total_number = 2

		self.assertEqual(submarine.check_untightess_zmatrix(z_matrix, z_matrix_list, total_number), "False, 1, 2, 1.0, 0.0\n")

		# 2.5) ambiguous entries are only absent, not present
		z_matrix = [[-1, 1, 1], [-1, -1, 0], [-1, -1, -1]]
		z_matrix_list = [[[-1, 1, 1], [-1, -1, -1], [-1, -1, -1]], [[-1, 1, 1], [-1, -1, -1], [-1, -1, -1]]]
		total_number = 2

		self.assertEqual(submarine.check_untightess_zmatrix(z_matrix, z_matrix_list, total_number), "False, 1, 2, 0.0, 1.0\n")

		# 2.7) ambiguous entries are only absent, not present, two entries are not truly ambiguous
		z_matrix = [[-1, 1, 1], [-1, 0, 0], [-1, -1, -1]]
		z_matrix_list = [[[-1, 1, 1], [-1, -1, -1], [-1, -1, -1]], [[-1, 1, 1], [-1, -1, -1], [-1, -1, -1]]]
		total_number = 2

		self.assertEqual(submarine.check_untightess_zmatrix(z_matrix, z_matrix_list, total_number), "False, 1, 1, 0.0, 1.0\nFalse, 1, 2, 0.0, 1.0\n")

		# 3) ambiguous entries are really ambiguous
		z_matrix = [[-1, 1, 1], [-1, -1, 0], [-1, -1, -1]]
		z_matrix_list = [[[-1, 1, 1], [-1, -1, -1], [-1, -1, -1]], [[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]]]
		total_number = 2

		self.assertEqual(submarine.check_untightess_zmatrix(z_matrix, z_matrix_list, total_number), "True\n")


	def test_upper_bound_number_reconstructions(self):
		ppm = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]
		self.assertEqual(submarine.upper_bound_number_reconstructions(ppm), np.log(6))

	def test_compute_number_ambiguous_recs_and_new_dfs(self):
		# 1) 4, all setting are possible
		lin0 = lineage.Lineage([1, 2, 3], [1, 1], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], [0.6, 0.6], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.3, 0.3], [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [0.1, 0.1], [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1]]

		# recursive
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1]]
		self.assertEqual(submarine.compute_number_ambiguous_recs(copy.deepcopy(my_lins), seg_num, copy.deepcopy(z_matrix), recursive=True), 4)
		# iterative
		total_count, valid_count = submarine.new_dfs(z_matrix, my_lins, seg_num)
		self.assertEqual(total_count, 4)
		self.assertEqual(valid_count, 4)

		# 2) 6, only subset possible because of tree rule
		lin0 = lineage.Lineage([1, 2, 3], [1, 1], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], [0.6, 0.6], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.3, 0.3], [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [0.1, 0.1], [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]

		# recursive
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		self.assertEqual(submarine.compute_number_ambiguous_recs(copy.deepcopy(my_lins), seg_num, copy.deepcopy(z_matrix), recursive=True), 6)
		# iterative
		total_count, valid_count = submarine.new_dfs(z_matrix, my_lins, seg_num)
		self.assertEqual(total_count, 8)
		self.assertEqual(valid_count, 6)

		# 3) 2, only subset possible because of sum rule
		lin0 = lineage.Lineage([1, 2, 3], [1, 1], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], [0.5, 0.5], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.4, 0.4], [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [0.3, 0.3], [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1]]

		# recursive
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1]]
		self.assertEqual(submarine.compute_number_ambiguous_recs(copy.deepcopy(my_lins), seg_num, copy.deepcopy(z_matrix), recursive=True), 2)
		# iterative
		total_count, valid_count = submarine.new_dfs(z_matrix, my_lins, seg_num)
		self.assertEqual(total_count, 4)
		self.assertEqual(valid_count, 2)

		# 4) 3, only subset possible because of absence rule
		lin0 = lineage.Lineage([1, 2, 3], [1, 1], [], [], [], [], [], [], [], [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 0
		ssm1.pos = 1
		ssm1.seg_index = 0
		lin1 = lineage.Lineage([], [0.6, 0.6], [], [], [], [], [], [ssm1], [], [])
		cnv2 = cnv.CNV(1, 0, 0, 0, 10)
		lin2 = lineage.Lineage([], [0.3, 0.3], [cnv2], [], [], [], [], [], [], [])
		cnv3 = cnv.CNV(1, 0, 0, 0, 10)
		lin3 = lineage.Lineage([], [0.1, 0.1], [], [cnv3], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1]]

		# recursive
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1]]
		count = submarine.compute_number_ambiguous_recs(copy.deepcopy(my_lins), seg_num, copy.deepcopy(z_matrix), recursive=True)
		self.assertEqual(count, 3)
		# iterative
		total_count, valid_count = submarine.new_dfs(z_matrix, my_lins, seg_num)
		self.assertEqual(total_count, 4)
		self.assertEqual(valid_count, 3)

		# 5) 3, only subset possible because of absence rule, k lost allele, k' has SSM phased to this allele
		lin0 = lineage.Lineage([1, 2, 3], [1, 1], [], [], [], [], [], [], [], [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 0
		ssm1.pos = 1
		ssm1.seg_index = 0
		cnv2 = cnv.CNV(-1, 0, 0, 0, 10)
		lin1 = lineage.Lineage([], [0.6, 0.6], [cnv2], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.3, 0.3], [], [], [], [], [], [], [ssm1], [])
		lin3 = lineage.Lineage([], [0.1, 0.1], [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1]]

		# recursive
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1]]
		count = submarine.compute_number_ambiguous_recs(copy.deepcopy(my_lins), seg_num, copy.deepcopy(z_matrix), recursive=True)
		self.assertEqual(count, 2)
		# iterative
		total_count, valid_count = submarine.new_dfs(z_matrix, my_lins, seg_num)
		self.assertEqual(total_count, 3)
		self.assertEqual(valid_count, 2)

		# 6) 3, only subset possible because of absence rule, k lost both alleles, k' has unphased SSMs
		lin0 = lineage.Lineage([1, 2, 3], [1, 1], [], [], [], [], [], [], [], [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 0
		ssm1.pos = 1
		ssm1.seg_index = 0
		cnv2 = cnv.CNV(-1, 0, 0, 0, 10)
		cnv3 = cnv.CNV(-1, 0, 0, 0, 10)
		lin1 = lineage.Lineage([], [0.6, 0.6], [cnv2], [cnv3], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.3, 0.3], [], [], [], [], [], [ssm1], [], [])
		lin3 = lineage.Lineage([], [0.1, 0.1], [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1]]

		# recursive
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1]]
		count = submarine.compute_number_ambiguous_recs(copy.deepcopy(my_lins), seg_num, copy.deepcopy(z_matrix), recursive=True)
		self.assertEqual(count, 2)
		# iterative
		total_count, valid_count = submarine.new_dfs(z_matrix, my_lins, seg_num)
		self.assertEqual(total_count, 3)
		self.assertEqual(valid_count, 2)

		# 7) 3, only subset possible because of absence rule, k lost allele, k'' is descendat with other CNC, k' has unphased SSMs
		lin0 = lineage.Lineage([1, 2, 3], [1, 1], [], [], [], [], [], [], [], [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 0
		ssm1.pos = 1
		ssm1.seg_index = 0
		cnv2 = cnv.CNV(-1, 0, 0, 0, 10)
		cnv3 = cnv.CNV(-1, 0, 0, 0, 10)
		lin1 = lineage.Lineage([3], [0.6, 0.6], [cnv2], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.3, 0.3], [], [], [], [], [], [ssm1], [], [])
		lin3 = lineage.Lineage([], [0.1, 0.1], [], [cnv3], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 1], [-1, -1, -1, 0], [-1, -1, -1, -1]]

		# recursive
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 1], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		count = submarine.compute_number_ambiguous_recs(copy.deepcopy(my_lins), seg_num, copy.deepcopy(z_matrix), recursive=True)
		self.assertEqual(count, 2)
		# iterative
		total_count, valid_count = submarine.new_dfs(z_matrix, my_lins, seg_num)
		self.assertEqual(total_count, 4)
		self.assertEqual(valid_count, 2)

		# 8) 3, only subset possible because of absence rule, k and k' loose same allele in same segment
		lin0 = lineage.Lineage([1, 2, 3], [1, 1], [], [], [], [], [], [], [], [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 0
		ssm1.pos = 1
		ssm1.seg_index = 0
		cnv2 = cnv.CNV(-1, 0, 0, 0, 10)
		cnv3 = cnv.CNV(-1, 0, 0, 0, 10)
		lin1 = lineage.Lineage([3], [0.6, 0.6], [cnv2], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.3, 0.3], [cnv3], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [0.1, 0.1], [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 1], [-1, -1, -1, 0], [-1, -1, -1, -1]]

		# recursive
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 1], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		with self.assertRaises(eo.MyException):
			# input should be propagted further, thus exception is thrown
			count = submarine.compute_number_ambiguous_recs(copy.deepcopy(my_lins), seg_num, copy.deepcopy(z_matrix), 
				recursive=True, check_validity=True)
		# iterative
		total_count, valid_count = submarine.new_dfs(z_matrix, my_lins, seg_num)
		self.assertEqual(total_count, 3)
		self.assertEqual(valid_count, 1)

		# 9) input invalid
		lin0 = lineage.Lineage([1, 2, 3], [1, 1], [], [], [], [], [], [], [], [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 0
		ssm1.pos = 1
		ssm1.seg_index = 0
		cnv2 = cnv.CNV(-1, 0, 0, 0, 10)
		cnv3 = cnv.CNV(-1, 0, 0, 0, 10)
		lin1 = lineage.Lineage([3], [0.6, 0.6], [cnv2], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.3, 0.3], [cnv3], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [0.1, 0.1], [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 1, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]

		# iterative
		with self.assertRaises(eo.ReconstructionInvalid):
			total_count, valid_count = submarine.new_dfs(z_matrix, my_lins, seg_num)

		# 10) 3, different CNVs in linegaes 1 and 3, unphased SSM in 2, depending on phasing of 2, it can be ancestor of 3 or not
		lin0 = lineage.Lineage([1, 2, 3], [1, 1], [], [], [], [], [], [], [], [])
		ssm1 = snp_ssm.SSM()
		ssm1.chr = 0
		ssm1.pos = 1
		ssm1.seg_index = 0
		cnv2 = cnv.CNV(-1, 0, 0, 0, 10)
		cnv3 = cnv.CNV(1, 0, 0, 0, 10)
		lin1 = lineage.Lineage([3], [0.6, 0.6], [cnv2], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.3, 0.3], [], [], [], [], [], [ssm1], [], [])
		lin3 = lineage.Lineage([], [0.1, 0.1], [], [cnv3], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]
		seg_num = 1
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]

		# recursive
		count = submarine.compute_number_ambiguous_recs(copy.deepcopy(my_lins), seg_num, copy.deepcopy(z_matrix), 
			recursive=True, check_validity=True)
		self.assertEqual(count, 5)
		# iterative
		reconstructions, total_count, valid_count = submarine.new_dfs(z_matrix, my_lins, seg_num, test_reconstructions=True)
		self.assertEqual(total_count, 8)
		self.assertEqual(valid_count, 5)
		self.assertEqual(reconstructions[0].present_ssms[0][1][2], True)
		self.assertEqual(reconstructions[3].present_ssms[0][0][2], True)

	def test_check_sum_rule(self):
		lin0 = lineage.Lineage([1, 2, 3], [1, 1], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([2, 3], [0.8, 0.5], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.4, 0.3], [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [0.3, 0.7], [], [], [], [], [], [], [], [])
		my_lins = [lin0, lin1, lin2, lin3]

		z_matrix = [[-1, 1, 1, 1], [-1, -1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1]]

		self.assertFalse(submarine.check_sum_rule(my_lins, z_matrix))
		
		lin1.freq[1] = 1
		self.assertTrue(submarine.check_sum_rule(my_lins, z_matrix))

		lin1.freq[1] = 0.8
		self.assertTrue(submarine.check_sum_rule(my_lins, z_matrix, noise_threshold=0.2))

	def test_change_unnecessary_phasing(self):

		lin_num = 5

		CNVs_0 = {}
		CNVs_0[cons.GAIN] = {}
		CNVs_0[cons.GAIN][cons.A] = {}
		CNVs_0[cons.GAIN][cons.A][3] = "something"
		CNVs_1 = {}
		CNVs_1[cons.LOSS] = {}
		CNVs_1[cons.LOSS][cons.B] = {}
		CNVs_1[cons.LOSS][cons.B][1] = "something"
		CNVs_2 = {}
		CNVs_2[cons.GAIN] = {}
		CNVs_2[cons.GAIN][cons.A] = {}
		CNVs_2[cons.GAIN][cons.A][4] = "something"
		CNVs_3 = {}
		CNVs_3[cons.GAIN] = {}
		CNVs_3[cons.GAIN][cons.A] = {}
		CNVs_3[cons.GAIN][cons.A][2] = "something"
		CNVs_4 = {}
		CNVs = [CNVs_0, CNVs_1, CNVs_2, CNVs_3, CNVs_4]

		present_ssms = [
			[[False, True, True, True, True], [False, True, True, True, True], [False, False, False, False, False]],
			[[False, True, True, True, True], [False, True, True, True, True], [False, False, False, False, False]],
			[[False, False, False, False, False], [False, True, True, True, True], 
			[False, False, False, False, False]],
			[[False, True, True, True, True], [False, True, True, True, True], [False, False, False, False, False]],
			[[False, True, True, True, True], [False, True, True, True, True], [False, False, False, False, False]],
			]

		ssm_infl_cnv_same_lineage = [
			[[False, False, False, False, False], [False, False, False, False, False]],
			[[False, False, False, False, False], [False, False, False, False, False]],
			[[False, False, False, False, True], [False, False, False, False, False]],
			[[False, False, True, False, False], [False, False, False, False, False]],
			[[False, False, False, False, False], [False, False, False, False, False]],
			]

		z_matrix = [
			[-1, 1, 1, 1, 1],
			[-1, -1, 1, 1, 1],
			[-1, -1, -1, 0, 1],
			[-1, -1, -1, -1, 0],
			[-1, -1, -1, -1, -1]
			]

		seg_num = 5

		# function to test
		submarine.change_unnecessary_phasing(lin_num, CNVs, present_ssms, ssm_infl_cnv_same_lineage, z_matrix, seg_num)

		self.assertEqual(present_ssms[0], [[False, True, False, False, False], [False, True, False, False, False],
			[False, False, True, True, True]])
		self.assertEqual(present_ssms[1], [[False, True, True, True, True], [False, True, True, True, True],
			[False, False, False, False, False]])
		self.assertEqual(present_ssms[2], [[False, False, False, False, False], [False, True, True, False, False],
			[False, False, False, True, True]])
		self.assertEqual(present_ssms[3], [[False, True, True, False, False], [False, True, False, False, False],
			[False, False, True, True, True]])
		self.assertEqual(present_ssms[4], [[False, False, False, False, False], [False, False, False, False, False],
			[False, True, True, True, True]])


		# with user constraints
		lin_num = 5

		CNVs_0 = {}
		CNVs_0[cons.GAIN] = {}
		CNVs_0[cons.GAIN][cons.A] = {}
		CNVs_0[cons.GAIN][cons.A][3] = "something"
		CNVs_1 = {}
		CNVs_1[cons.LOSS] = {}
		CNVs_1[cons.LOSS][cons.B] = {}
		CNVs_1[cons.LOSS][cons.B][1] = "something"
		CNVs_2 = {}
		CNVs_2[cons.GAIN] = {}
		CNVs_2[cons.GAIN][cons.A] = {}
		CNVs_2[cons.GAIN][cons.A][4] = "something"
		CNVs_3 = {}
		CNVs_3[cons.GAIN] = {}
		CNVs_3[cons.GAIN][cons.A] = {}
		CNVs_3[cons.GAIN][cons.A][2] = "something"
		CNVs_4 = {}
		CNVs = [CNVs_0, CNVs_1, CNVs_2, CNVs_3, CNVs_4]

		present_ssms = [
			[[False, True, True, True, True], [False, True, True, True, True], [False, False, False, False, False]],
			[[False, True, True, True, True], [False, True, True, True, True], [False, False, False, False, False]],
			[[False, False, False, False, False], [False, True, True, True, True], 
			[False, False, False, False, False]],
			[[False, True, True, True, True], [False, True, True, True, True], [False, False, False, False, False]],
			[[False, True, True, True, True], [False, True, True, True, True], [False, False, False, False, False]],
			]

		ssm_infl_cnv_same_lineage = [
			[[False, False, False, False, False], [False, False, False, False, False]],
			[[False, False, False, False, False], [False, False, False, False, False]],
			[[False, False, False, False, False], [False, False, False, False, False]],
			[[False, False, True, False, False], [False, False, False, False, False]],
			[[False, False, False, False, False], [False, False, False, False, False]],
			]

		user_ssm = [
			[[False, False, False, False, False], [False, False, True, False, False]],
			[[False, False, False, False, False], [False, False, False, False, False]],
			[[False, False, False, False, False], [False, False, False, False, True]],
			[[False, False, False, True, False], [False, False, False, True, False]],
			[[False, True, False, False, False], [False, False, True, False, False]]
			]

		z_matrix = [
			[-1, 1, 1, 1, 1],
			[-1, -1, 1, 1, 1],
			[-1, -1, -1, 0, 1],
			[-1, -1, -1, -1, 0],
			[-1, -1, -1, -1, -1]
			]

		seg_num = 5

		# function to test
		submarine.change_unnecessary_phasing(lin_num, CNVs, present_ssms, ssm_infl_cnv_same_lineage, z_matrix, seg_num, user_ssm)

		self.assertEqual(present_ssms[0], [[False, True, False, False, False], [False, True, True, False, False],
			[False, False, True, True, True]])
		self.assertEqual(present_ssms[1], [[False, True, True, True, True], [False, True, True, True, True],
			[False, False, False, False, False]])
		self.assertEqual(present_ssms[2], [[False, False, False, False, False], [False, True, True, False, True],
			[False, False, False, True, False]])
		self.assertEqual(present_ssms[3], [[False, True, True, True, False], [False, True, False, True, False],
			[False, False, True, False, True]])
		self.assertEqual(present_ssms[4], [[False, True, False, False, False], [False, False, True, False, False],
			[False, True, True, True, True]])

	def test_is_CN_gain_in_k(self):
		CNVs = {}
		CNVs[cons.GAIN] = {}
		CNVs[cons.GAIN][cons.A] = {}
		CNVs[cons.GAIN][cons.A][1] = "something"
		CNVs[cons.GAIN][cons.B] = {}
		CNVs[cons.GAIN][cons.B][2] = "something"
		CNVs[cons.GAIN][cons.B][3] = "something"
		ssm_infl_cnv_same_lin_i = [
			[False, True, False, True],
			[False, False, True, False]
			]

		self.assertEqual(submarine.is_CN_gain_in_k(1, CNVs, ssm_infl_cnv_same_lin_i), (True, False))
		self.assertEqual(submarine.is_CN_gain_in_k(2, CNVs, ssm_infl_cnv_same_lin_i), (False, True))
		self.assertEqual(submarine.is_CN_gain_in_k(3, CNVs, ssm_infl_cnv_same_lin_i), (False, False))

	def test_do_descendants_have_CN_change(self):
		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][1] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"
		CNVs[cons.GAIN] = {}
		CNVs[cons.GAIN][cons.A] = {}
		CNVs[cons.GAIN][cons.A][3] = "something"
		CNVs[cons.GAIN][cons.B] = {}
		CNVs[cons.GAIN][cons.B][4] = "something"

		self.assertEqual(submarine.do_descendants_have_CN_change([0, 1], CNVs), True)
		self.assertEqual(submarine.do_descendants_have_CN_change([0, 2], CNVs), True)
		self.assertEqual(submarine.do_descendants_have_CN_change([0, 3], CNVs), True)
		self.assertEqual(submarine.do_descendants_have_CN_change([0, 4], CNVs), True)
		self.assertEqual(submarine.do_descendants_have_CN_change([0, 5], CNVs), False)

	def test_do_ancestors_have_CN_loss(self):

		CNVs = {}
		CNVs[cons.LOSS] = {}
		CNVs[cons.LOSS][cons.A] = {}
		CNVs[cons.LOSS][cons.A][1] = "something"
		CNVs[cons.LOSS][cons.B] = {}
		CNVs[cons.LOSS][cons.B][2] = "something"

		self.assertEqual(submarine.do_ancestors_have_CN_loss([0, 1], CNVs), True)
		self.assertEqual(submarine.do_ancestors_have_CN_loss([0, 2], CNVs), True)
		self.assertEqual(submarine.do_ancestors_have_CN_loss([3], CNVs), False)

	def test_sum_rule_algo_outer_loop(self):

		# 1) a simple example: 3 lineages, lineage 2 can be child only of 1
		# input
		z_matrix = [[-1, 1, 1], [-1, -1, 0], [-1, -1, -1]]
		linFreqs = np.asarray([[1.0, 1.0], [0.8, 0.6], [0.1, 0.5]])
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
				z_matrix, 3*3, 3)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=None,
				matrix_after_first_round=z_matrix)
		# supposed output
		avFreqs_true = np.asarray([[0.2, 0.4], [0.7, 0.1], [0.1, 0.5]])
		ppm_true = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
		new_z = [[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]]
		seg_num = 1
		zero_count = 10 # set to some number, doesn't need to make sense
		gain_num = [0]
		loss_num = [0]
		CNVs = [{}]
		present_ssms = None

		(mybool, avFreqs, ppm) = submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertTrue(mybool)
		self.assertTrue(np.array_equal(np.asarray(ppm_true), ppm))
		self.assertTrue(np.isclose(avFreqs_true, avFreqs).all())
		self.assertEqual(zmco.z_matrix, new_z)

		# 2) 4 lineages, lineage 2 has two potential parents 0 and 1, but as soon as lineage 3 becomes child of 0, 2 becomes child of 1
		#input
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]
		linFreqs = np.asarray([[1.0, 1.0], [0.7, 0.6], [0.1, 0.3], [0.05, 0.2]])
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
				z_matrix, 4*4, 4)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=None,
				matrix_after_first_round=z_matrix)
		seg_num = 1
		zero_count = 10 # set to some number, doesn't need to make sense
		gain_num = [0]
		loss_num = [0]
		CNVs = [{}]
		present_ssms = [[[False] * 4 for _ in range(3)]]
		# supposed output
		avFreqs_true = np.asarray([[0.25, 0.2], [0.6, 0.3], [0.1, 0.3], [0.05, 0.2]])
		ppm_true = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
		new_z = [[-1, 1, 1, 1], [-1, -1, 1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]

		(mybool, avFreqs, ppm) = submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertTrue(mybool)
		self.assertTrue(np.array_equal(np.asarray(ppm_true), ppm))
		self.assertTrue(np.isclose(avFreqs_true, avFreqs).all())
		self.assertEqual(zmco.z_matrix, new_z)

		# 3) 4 lineages, both 2 and 3 are potential children of 0 and 1
		# input
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1]]
		linFreqs = np.asarray([[1.0], [0.5], [0.4], [0.3]])
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
				z_matrix, 4*4, 4)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=None,
				matrix_after_first_round=z_matrix)
		seg_num = 1
		zero_count = 10 # set to some number, doesn't need to make sense
		gain_num = [1]
		loss_num = [1]
		CNVs = [{}]
		present_ssms = None
		# supposed output
		avFreqs_true = np.asarray([[0.5], [0.5], [0.4], [0.3]])
		ppm_true = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]
		new_z = copy.deepcopy(z_matrix)

		(mybool, avFreqs, ppm) = submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertTrue(mybool)
		self.assertTrue(np.array_equal(np.asarray(ppm_true), ppm))
		self.assertTrue(np.isclose(avFreqs_true, avFreqs).all())
		self.assertEqual(zmco.z_matrix, new_z)

		# 4) 4 lineages, 1 & 2 are children of 0, then there is no parent left for 3
		# input
		z_matrix = [[-1, 1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]
		linFreqs = np.asarray([[1.0], [0.5], [0.4], [0.3]])
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
				z_matrix, 4*4, 4)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=None,
				matrix_after_first_round=z_matrix)
		seg_num = 1
		zero_count = 10 # set to some number, doesn't need to make sense
		gain_num = [1]
		loss_num = [1]
		CNVs = [{}]
		present_ssms = None


		with self.assertRaises(eo.NoParentsLeftNoise) as e:
			submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)
		message = "There are no possible parents for subclone 3 with frequency of 0.300, because subclone 0 has only available frequency of 0.100.\nCurrent tree with definite children: 0->1,0->2."
		try:
			submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)
		except eo.NoParentsLeftNoise as e:
			self.assertEqual(e.k, 3)
			self.assertTrue(np.allclose(e.avFreqs_from_initial_pps[0], np.asarray([0.1])))
			self.assertEqual(e.message, message)

		# 4.1) 4 lineages, 1 & 2 are children of 0, in order for 3 to be a child of 0, the noise threshold has to be set up
		# input
		z_matrix = [[-1, 1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]
		linFreqs = np.asarray([[1.0], [0.5], [0.4], [0.3]])
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
				z_matrix, 4*4, 4)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=None,
				matrix_after_first_round=z_matrix)
		seg_num = 1
		zero_count = 10 # set to some number, doesn't need to make sense
		gain_num = [1]
		loss_num = [1]
		CNVs = [{}]
		present_ssms = None

		ppm_true = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
		avFreqs_true = np.asarray([[-0.2], [0.5], [0.4], [0.3]])
		new_z = z_matrix = [[-1, 1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]

		(mybool, avFreqs, ppm) = submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms,
			noise_threshold=0.2)
		self.assertTrue(mybool)
		self.assertTrue(np.array_equal(np.asarray(ppm_true), ppm))
		self.assertTrue(np.isclose(avFreqs_true, avFreqs).all())
		self.assertEqual(zmco.z_matrix, new_z)

		# 5) 8 lineages, Z-matrix and frequencies are given in a way that no reconstruction exists that fulfills the sum rule
		# when lineage 5 becomes a child of lineage 2, available frequency becomes 0
		# input
		z_matrix = [
				[-1, 1, 1, 1, 1, 1, 1, 1],
				[-1, -1, -1, 0, 1, 0, -1, -1],
				[-1, -1, -1, 0, -1, 0, 1, 1],
				[-1, -1, -1, -1, -1, -1, -1, -1],
				[-1, -1, -1, -1, -1, -1, -1, -1],
				[-1, -1, -1, -1, -1, -1, -1, -1],
				[-1, -1, -1, -1, -1, -1, -1, -1],
				[-1, -1, -1, -1, -1, -1, -1, -1]
				]
		linFreqs = np.asarray([[1.0, 1.0], [0.6, 0.6], [0.4, 0.4], [0.27, 0.08], [0.25, 0.05], [0.2, 0.03], [0.13, 0.13], [0.11, 0.11]])
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
				z_matrix, 8*8, 8)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=None,
				matrix_after_first_round=z_matrix)
		seg_num = 1
		zero_count = 10 # set to some number, doesn't need to make sense
		gain_num = [1]
		loss_num = [1]
		CNVs = [{}]
		present_ssms = None

		with self.assertRaises(eo.NoParentsLeftNoise) as e:
			submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		# 5.1) 8 lineages, Z-matrix and frequencies are given in a way that no reconstruction exists that fulfills the sum rule
		# when lineage 5 becomes a child of lineage 2, available frequency becomes 0
		# test returned exception
		# input
		z_matrix = [
				[-1, 1, 1, 1, 1, 1, 1, 1],
				[-1, -1, -1, 0, 1, 0, -1, -1],
				[-1, -1, -1, 0, -1, 0, 1, 1],
				[-1, -1, -1, -1, -1, -1, -1, -1],
				[-1, -1, -1, -1, -1, -1, -1, -1],
				[-1, -1, -1, -1, -1, -1, -1, -1],
				[-1, -1, -1, -1, -1, -1, -1, -1],
				[-1, -1, -1, -1, -1, -1, -1, -1]
				]
		linFreqs = np.asarray([[1.0, 1.0], [0.6, 0.6], [0.4, 0.4], [0.27, 0.08], [0.25, 0.05], [0.2, 0.03], [0.13, 0.13], [0.11, 0.11]])
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
				z_matrix, 8*8, 8)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=None,
				matrix_after_first_round=z_matrix)
		seg_num = 1
		zero_count = 10 # set to some number, doesn't need to make sense
		gain_num = [1]
		loss_num = [1]
		CNVs = [{}]
		present_ssms = None
		try:
			submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)
		except eo.NoParentsLeftNoise as e:
			self.assertEqual(e.k, 5)
			self.assertTrue(np.isclose(e.avFreqs_from_initial_pps[0], np.asarray([0, 0])).all())
			self.assertTrue(np.isclose(e.avFreqs_from_initial_pps[1], np.asarray([0.08, 0.47])).all())
			self.assertTrue(np.isclose(e.avFreqs_from_initial_pps[2], np.asarray([0.16, 0.16])).all())
			message = "There are no possible parents for subclone 5 with frequencies of 0.200,0.030, because subclone 0 has only available frequencies of 0.000,0.000, subclone 1 has only available frequencies of 0.080,0.470, subclone 2 has only available frequencies of 0.160,0.160.\nCurrent tree with definite children: 0->1,0->2,1->3,1->4,2->6,2->7."
			self.assertEqual(e.message, message)

		# 6) 4 lineages, 3 has two potential parents but because 2 becomes child of 1 and relationships are updated, it becomes child of 0
		# input
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, -1], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		linFreqs = np.asarray([[1.0], [0.7], [0.4], [0.2]])
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
				z_matrix, 4*4, 4)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=None,
				matrix_after_first_round=z_matrix)
		seg_num = 1
		zero_count = 10 # set to some number, doesn't need to make sense
		gain_num = [0]
		loss_num = [0]
		CNVs = [{}]
		present_ssms = [[[False] * 4 for _ in range(3)]]
		# supposed output
		avFreqs_true = np.asarray([[0.1], [0.3], [0.4], [0.2]])
		ppm_true = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]] 
		new_z = copy.deepcopy(z_matrix)
		new_z[1][2] = 1
		new_z[2][3] = -1
		
		(mybool, avFreqs, ppm) = submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertTrue(mybool)
		self.assertTrue(np.array_equal(np.asarray(ppm_true), ppm))
		self.assertTrue(np.isclose(avFreqs_true, avFreqs).all())
		self.assertEqual(zmco.z_matrix, new_z)

		# 7) 4 lineages, possible parent-child relationship is removed, leading to a forbidden ADR in Z-matrix
		# * as soon as 3 becomes child of 1, 2 becomes child of 0, no ADR with 1 anymore
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 1], [-1, -1, -1, -1], [-1, -1, -1, -1]]
		linFreqs = np.asarray([[1.0], [0.5], [0.4], [0.3]])
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
				z_matrix, 4*4, 4)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=None,
				matrix_after_first_round=z_matrix)
		seg_num = 1
		zero_count = 10 # set to some number, doesn't need to make sense
		gain_num = [1]
		loss_num = [1]
		CNVs = [{}]
		present_ssms = None
		# supposed output
		avFreqs_true = np.asarray([[0.1], [0.2], [0.4], [0.3]])
		ppm_true = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]] 
		new_z = copy.deepcopy(z_matrix)
		new_z[1][2] = -1

		(mybool, avFreqs, ppm) = submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertTrue(mybool)
		self.assertTrue(np.array_equal(np.asarray(ppm_true), ppm))
		self.assertTrue(np.isclose(avFreqs_true, avFreqs).all())
		self.assertEqual(zmco.z_matrix, new_z)

		# 8) 5 lineages, although possible parent-child relationship is removed ambiguous ADR stays
		# * 4 becomes child of 1, so 3 cannot be a potential child of 1 anymore, however is still stays in an ambiguous ADR because of 
		#   relationship to lineage 2
		z_matrix = [[-1, 1, 1, 1, 1], [-1, -1, 1, 0, 1], [-1, -1, -1, 0, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]]
		linFreqs = np.asarray([[1.0], [0.5], [0.3], [0.25], [0.2]])
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
					z_matrix, 5*5, 5)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=None,
				matrix_after_first_round=z_matrix)
		seg_num = 1
		zero_count = 10 # set to some number, doesn't need to make sense
		gain_num = [1]
		loss_num = [1]
		CNVs = [{}]
		present_ssms = None
		# supposed output
		avFreqs_true = np.asarray([[0.5], [0], [0.3], [0.25], [0.2]])
		#ppm_true = [[0, 0, 0, 0, 0] for i in range(5)]
		ppm_true = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 0, 0]]
		new_z = copy.deepcopy(z_matrix)

		(mybool, avFreqs, ppm) = submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertTrue(mybool)
		self.assertTrue(np.array_equal(np.asarray(ppm_true), ppm))
		self.assertTrue(np.isclose(avFreqs_true, avFreqs).all())
		self.assertEqual(zmco.z_matrix, new_z)

		# 9) update of relationships takes away a pp and leaves a lineage with a single pp in back update
		# * 4 can only be child of 1, then 2 can't be child of 1, becomes child of 0
		# * with Z_1,2 = 0, Z_2,3 becomes 0, thus 3 has one pp less, becomes child of 1
		z_matrix = [[-1, 1, 1, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, -1, 0, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]]
		linFreqs = np.asarray([[1.0], [0.6], [0.4], [0.3], [0.25]])
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
					z_matrix, 5*5, 5)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=None,
				matrix_after_first_round=z_matrix)
		seg_num = 1
		zero_count = 10 # set to some number, doesn't need to make sense
		gain_num = [1]
		loss_num = [1]
		CNVs = [{}]
		present_ssms = None
		# supposed output
		avFreqs_true = np.asarray([[0], [0.05], [0.4], [0.3], [0.25]])
		ppm_true = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]
		new_z = copy.deepcopy(z_matrix)
		new_z[1][2] = -1
		new_z[2][3] = -1

		(mybool, avFreqs, ppm) = submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertTrue(mybool)
		self.assertTrue(np.array_equal(np.asarray(ppm_true), ppm))
		self.assertTrue(np.isclose(avFreqs_true, avFreqs).all())
		self.assertEqual(zmco.z_matrix, new_z)

		# 10) 4 lineages, lin. 1 has unphased SSM, lineages 2 and 3 CN losses on different segments
		# given frequency, only linear phylogeny is possible but this is not allowed by SSM phasing
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		linFreqs = np.asarray([[1.0], [0.9], [0.8], [0.7]])
		lin0 = lineage.Lineage([1, 2, 3], 1.0, [], [], [], [], [], [], [], [])
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		lin1 = lineage.Lineage([], 0.9, [], [], [], [], [], [ssm1], [], [])
		cnv2 = cnv.CNV(-1, 0, 1, 1, 100)
		lin2 = lineage.Lineage([], 0.8, [cnv2], [], [], [], [], [], [], [])
		cnv3 = cnv.CNV(-1, 0, 1, 1, 100)
		lin3 = lineage.Lineage([], 0.7, [], [cnv3], [], [], [], [], [], [])
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
					z_matrix, 4*4, 4)
		seg_num = 1
		gain_num = []
		loss_num = []
		CNVs = []
		present_ssms = []
		lineage_num = 4
		my_lineages = [lin0, lin1, lin2, lin3]
		ssm_infl_cnv_same_lineage = []
		submarine.get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
			ssm_infl_cnv_same_lineage)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=present_ssms,
				matrix_after_first_round=copy.deepcopy(z_matrix))

		with self.assertRaises(eo.NoParentsLeftNoise) as e:
			submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)
		
		# 11) 4 lineages, lin. 2 has unphased SSM, lineages 1 and 3 CN losses on different segments
		# given frequency, only linear phylogeny is possible but this is not allowed by SSM phasing
		z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]]
		linFreqs = np.asarray([[1.0], [0.9], [0.8], [0.7]])
		lin0 = lineage.Lineage([1, 2, 3], 1.0, [], [], [], [], [], [], [], [])
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		cnv2 = cnv.CNV(-1, 0, 1, 1, 100)
		lin1 = lineage.Lineage([], 0.9, [cnv2], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.8, [], [], [], [], [], [ssm1], [], [])
		cnv3 = cnv.CNV(-1, 0, 1, 1, 100)
		lin3 = lineage.Lineage([], 0.7, [], [cnv3], [], [], [], [], [], [])
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
					z_matrix, 4*4, 4)
		seg_num = 1
		gain_num = []
		loss_num = [2]
		CNVs = []
		present_ssms = []
		lineage_num = 4
		my_lineages = [lin0, lin1, lin2, lin3]
		ssm_infl_cnv_same_lineage = []
		submarine.get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
			ssm_infl_cnv_same_lineage)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=present_ssms,
				matrix_after_first_round=copy.deepcopy(z_matrix))

		with self.assertRaises(eo.NoParentsLeftNoise) as e:
			submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)


		# 12) 6 lineages, lin. 5 can only be a child of lin. 1, however, during the algorithm Z_{2,5} is first not set to 0
		#	because lin. 3 is a potential parent of lin. 5 and a descendant of lin. 2
		z_matrix = [[-1, 1, 1, 1, 1, 1], [-1, -1, 0, 0, 0, 0], [-1, -1, -1, 0, 0, 0], [-1, -1, -1, -1, 0, 0],
			[-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]]
		linFreqs = np.asarray([[1.0],[1.0], [0.8], [0.7], [0.6], [0.2]])
		lin0 = lineage.Lineage([1, 2, 3, 4, 5], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 1.0, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.8, [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0.7, [], [], [], [], [], [], [], [])
		lin4 = lineage.Lineage([], 0.6, [], [], [], [], [], [], [], [])
		lin5 = lineage.Lineage([], 0.2, [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2, lin3, lin4, lin5]
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, 6*6, 6)
		seg_num = 1
		gain_num = []
		loss_num = []
		CNVs = []
		present_ssms = []
		lineage_num = 6
		ssm_infl_cnv_same_lineage = []
		submarine.get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
			ssm_infl_cnv_same_lineage)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=present_ssms,
			matrix_after_first_round=copy.deepcopy(z_matrix))

		ppm_true = [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
			[0, 1, 0, 0, 0, 0]]

		(mybool, avFreqs, ppm) = submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertEqual(ppm.tolist(), ppm_true)
		self.assertEqual(zmco.z_matrix[2][5], -1)
		self.assertEqual(zmco.z_matrix[3][5], -1)

		# 13) 7 lineages, lin. 6 can be a child of lin. 1 and 2 but not of lin. 3 or 4, 
		# however, during the algorithm Z_{3,5} is first not set to 0
		#	because lin. 4 is a potential parent of lin. 6 and a descendant of lin. 3
		z_matrix = [[-1, 1, 1, 1, 1, 1, 1], [-1, -1, 0, 0, 0, 0, 0], [-1, -1, -1, 0, 0, 0, 0], [-1, -1, -1, -1, 0, 0, 0],
			[-1, -1, -1, -1, -1, 0, 0], [-1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1]]
		linFreqs = np.asarray([[1.0],[1.0], [0.7], [0.6], [0.55], [0.5], [0.1]])
		lin0 = lineage.Lineage([1, 2, 3, 4, 5, 6], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 1.0, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.7, [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0.6, [], [], [], [], [], [], [], [])
		lin4 = lineage.Lineage([], 0.55, [], [], [], [], [], [], [], [])
		lin5 = lineage.Lineage([], 0.5, [], [], [], [], [], [], [], [])
		lin6 = lineage.Lineage([], 0.1, [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2, lin3, lin4, lin5, lin6]
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, 7*7, 7)
		seg_num = 1
		gain_num = []
		loss_num = []
		CNVs = []
		present_ssms = []
		lineage_num = 7
		ssm_infl_cnv_same_lineage = []
		submarine.get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
			ssm_infl_cnv_same_lineage)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=present_ssms,
			matrix_after_first_round=copy.deepcopy(z_matrix))

		ppm_true = [[0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, 0], [0, 1, 1, 0, 0, 0, 0]]

		(mybool, avFreqs, ppm) = submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertEqual(ppm.tolist(), ppm_true)
		self.assertEqual(zmco.z_matrix[2][6], 0)
		self.assertEqual(zmco.z_matrix[4][6], -1)
		self.assertEqual(zmco.z_matrix[3][6], -1)

		# 14) 10 lineages, lin. 9 can be a child of lin. 1 and 2, 
		# however, during the algorithm Z_{3,9} and Z_{6,9} are first not set to 0
		# because the descendant of 3 and 6 are potential parents of 9
		z_matrix = [
			[-1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
			[-1, -1, 0, 0, 0, 0, -1, -1, -1, 0],
			[-1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
			[-1, -1, -1, -1, 0, 0, -1, -1, -1, 0],
			[-1, -1, -1, -1, -1, 0, -1, -1, -1, 0],
			[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
			[-1, -1, -1, -1, -1, -1, -1, 0, 0, 0],
			[-1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
			[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
			[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
			]
		linFreqs = np.asarray([[1.0],[0.9], [0.7], [0.6], [0.55], [0.5], [0.1], [0.09], [0.08], [0.07]])
		lin0 = lineage.Lineage([1, 2, 3, 4, 5, 6, 7, 8, 9], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.9, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.7, [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0.6, [], [], [], [], [], [], [], [])
		lin4 = lineage.Lineage([], 0.55, [], [], [], [], [], [], [], [])
		lin5 = lineage.Lineage([], 0.5, [], [], [], [], [], [], [], [])
		lin6 = lineage.Lineage([], 0.1, [], [], [], [], [], [], [], [])
		lin7 = lineage.Lineage([], 0.09, [], [], [], [], [], [], [], [])
		lin8 = lineage.Lineage([], 0.08, [], [], [], [], [], [], [], [])
		lin9 = lineage.Lineage([], 0.07, [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2, lin3, lin4, lin5, lin6, lin7, lin8, lin9]
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, 10*10, 10)
		seg_num = 1
		gain_num = []
		loss_num = []
		CNVs = []
		present_ssms = []
		lineage_num = 10
		ssm_infl_cnv_same_lineage = []
		submarine.get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
			ssm_infl_cnv_same_lineage)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=present_ssms,
			matrix_after_first_round=copy.deepcopy(z_matrix))

		ppm_true = [
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
			[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
			]

		(mybool, avFreqs, ppm) = submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertEqual(ppm.tolist(), ppm_true)
		self.assertEqual(zmco.z_matrix[1][9], 1)
		self.assertEqual(zmco.z_matrix[2][9], 0)
		self.assertEqual(zmco.z_matrix[4][9], -1)
		self.assertEqual(zmco.z_matrix[3][9], -1)
		self.assertEqual(zmco.z_matrix[7][9], -1)
		self.assertEqual(zmco.z_matrix[6][9], -1)

		# 15) 7 lineages, lin. 5 can only be child of lins 1 or 4
		# but initially Z_{2,5} and Z_{3,5} are not set to 0 because of possible descendant
		# that can be parents of lin 5
		z_matrix = [
			[-1, 1, 1, 1, 1, 1, 1],
		[-1, -1, 1, 1, 0, 0, 0],
		[-1, -1, -1, 1, 0, 0, 0],
		[-1, -1, -1, -1, 0, 0, 1],
		[-1, -1, -1, -1, -1, 0, -1],
		[-1, -1, -1, -1, -1, -1, -1],
		[-1, -1, -1, -1, -1, -1, -1]
			]
		linFreqs = np.asarray([[1.0],[0.3], [0.2], [0.16], [0.15], [0.1], [0.09]])
		lin0 = lineage.Lineage([1, 2, 3, 4, 5, 6], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 0.3, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.2, [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0.16, [], [], [], [], [], [], [], [])
		lin4 = lineage.Lineage([], 0.15, [], [], [], [], [], [], [], [])
		lin5 = lineage.Lineage([], 0.1, [], [], [], [], [], [], [], [])
		lin6 = lineage.Lineage([], 0.09, [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2, lin3, lin4, lin5, lin6]
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, 7*7, 7)
		seg_num = 1
		gain_num = []
		loss_num = []
		CNVs = []
		present_ssms = []
		lineage_num = 7
		ssm_infl_cnv_same_lineage = []
		submarine.get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
			ssm_infl_cnv_same_lineage)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=present_ssms,
			matrix_after_first_round=copy.deepcopy(z_matrix))

		ppm_true = [
			[0, 0, 0, 0, 0, 0, 0],
		[1, 0, 0, 0, 0, 0, 0],
		[0, 1, 0, 0, 0, 0, 0],
		[0, 0, 1, 0, 0, 0, 0],
		[1, 0, 0, 0, 0, 0, 0],
		[1, 1, 0, 0, 1, 0, 0],
		[0, 0, 0, 1, 0, 0, 0]
			]

		(mybool, avFreqs, ppm) = submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertEqual(ppm.tolist(), ppm_true)
		self.assertEqual(zmco.z_matrix[3][5], -1)
		self.assertEqual(zmco.z_matrix[2][5], -1)

		# 16) 5 lineages, 4 has two potential parents 1 and 3 and is
		# ancestor of 1
		z_matrix = [[-1, 1, 1, 1, 1],
			[-1, -1, 0, 0, 0],
			[-1, -1, -1, 0, 0],
			[-1, -1, -1, -1, 0],
			[-1, -1, -1, -1, -1]]
		linFreqs = np.asarray([[1.0], [1.0], [0.8], [0.8], [0.2]])
		lin0 = lineage.Lineage([1, 2, 3, 4], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 1.0, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.8, [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0.8, [], [], [], [], [], [], [], [])
		lin4 = lineage.Lineage([], 0.2, [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2, lin3, lin4]
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, 5*5, 5)
		seg_num = 1
		gain_num = []
		loss_num = []
		CNVs = []
		present_ssms = []
		lineage_num = 5
		ssm_infl_cnv_same_lineage = []
		submarine.get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
			ssm_infl_cnv_same_lineage)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=present_ssms,
			matrix_after_first_round=copy.deepcopy(z_matrix))

		ppm_true = [
			[0, 0, 0, 0, 0],
			[1, 0, 0, 0, 0],
			[0, 1, 0, 0, 0],
			[0, 0, 1, 0, 0],
			[0, 1, 0, 1, 0]
			]

		(mybool, avFreqs, ppm) = submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertEqual(ppm.tolist(), ppm_true)
		self.assertEqual(zmco.z_matrix[1][4], 1)

		# 17) 3 lineages, 2 can be child of 0 because of noise threshold
		# tests outer loop of sum rule and make_def_child first mention of threshold
		z_matrix = [[-1, 1, 1], [-1, -1, -1], [-1, -1, -1]]
		linFreqs = np.asarray([[1.0], [1.0], [0.2]])
		lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 1.0, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.2, [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2]
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, 3*3, 3)
		seg_num = 1
		gain_num = []
		loss_num = []
		CNVs = []
		present_ssms = []
		lineage_num = 3
		ssm_infl_cnv_same_lineage = []
		submarine.get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
			ssm_infl_cnv_same_lineage)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=present_ssms,
			matrix_after_first_round=copy.deepcopy(z_matrix))

		ppm_true = [[0, 0, 0], [1, 0, 0], [1, 0 , 0]]
		avFreqs_true = np.asarray([[-0.2], [1.0], [0.2]])

		(mybool, avFreqs, ppm) = submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms,
			noise_threshold=0.2)

		self.assertEqual(ppm.tolist(), ppm_true)
		self.assertTrue(np.isclose(avFreqs_true, avFreqs).all())

		# 18) 3 lineages, 2 can be child of both 0 and 1 when noise threshold is high enough
		z_matrix = [[-1, 1, 1], [-1, -1, 0], [-1, -1, -1]]
		linFreqs = np.asarray([[1.0], [1.0], [0.2]])
		lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], 1.0, [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], 0.2, [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2]
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, 3*3, 3)
		seg_num = 1
		gain_num = []
		loss_num = []
		CNVs = []
		present_ssms = []
		lineage_num = 3
		ssm_infl_cnv_same_lineage = []
		submarine.get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
			ssm_infl_cnv_same_lineage)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=present_ssms,
			matrix_after_first_round=copy.deepcopy(z_matrix))

		ppm_true = [[0, 0, 0], [1, 0, 0], [1, 1 , 0]]
		avFreqs_true = np.asarray([[0], [1.0], [0.2]])

		(mybool, avFreqs, ppm) = submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms,
			noise_threshold=0.2)

		self.assertEqual(ppm.tolist(), ppm_true)
		self.assertTrue(np.isclose(avFreqs_true, avFreqs).all())


		# 19) example why it needs to be checked that subclone doesn't have other _possible_ descendants that are possible parents before setting Z to 0
		# 5 can only be child of 1
		# thne 3 becomes child of 2, and 4 of 3
		# the order in which this update happens requires 2 to have possible descendants that are possible possible parents of 4,
		# otherwise Z(2,4) would be set to 0
		z_matrix = [
			[-1, 1, 1, 1, 1, 1],
			[-1, -1, 0, 0, 0, 0],
			[-1, -1, -1, 0, 0, -1],
			[-1, -1, -1, -1, 0, -1],
			[-1, -1, -1, -1, -1, -1],
			[-1, -1, -1, -1, -1, -1,]
			]
		linFreqs = np.asarray([[1.0, 1.0], [1.0, 1.0], [0.45, 0.5], [0.4, 0.4], [0.3, 0.3], [0.5, 0.1]])
		lin0 = lineage.Lineage([1, 2, 3, 4, 5], [1.0, 1.0], [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([], [1.0, 1.0], [], [], [], [], [], [], [], [])
		lin2 = lineage.Lineage([], [0.45,0.5], [], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], [0.4, 0.4], [], [], [], [], [], [], [], [])
		lin4 = lineage.Lineage([], [0.3, 0.3], [], [], [], [], [], [], [], [])
		lin5 = lineage.Lineage([], [0.5, 0.1], [], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2, lin3, lin4, lin5]
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, 6*6, 6)
		seg_num = 1
		gain_num = []
		loss_num = []
		CNVs = []
		present_ssms = []
		lineage_num = 6
		ssm_infl_cnv_same_lineage = []
		submarine.get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
			ssm_infl_cnv_same_lineage)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=present_ssms,
			matrix_after_first_round=copy.deepcopy(z_matrix))

		ppm_true = np.asarray([
			[0, 0, 0, 0, 0, 0],
			[1, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0],
			[0, 0, 1, 0, 0, 0],
			[0, 0, 0, 1, 0, 0],
			[0, 1, 0, 0, 0, 0]
			])

		(mybool, avFreqs, ppm) = submarine.sum_rule_algo_outer_loop(linFreqs, zmco, seg_num, zero_count, gain_num, loss_num, CNVs, present_ssms)

		self.assertTrue((ppm == ppm_true).all())
		

	def test_get_all_possible_z_matrices_with_lineages_new(self):

		# some simple example
		lin0 = lineage.Lineage([1, 2, 3, 4], [1.0, 1.0], [], [], [], [], [], [], [], [])
		cnv1 = cnv.CNV(-1, 0, 1, 1, 100)
		lin1 = lineage.Lineage([], [0.6, 0.5], [cnv1], [], [], [], [], [], [], [])
		cnv2 = cnv.CNV(-1, 0, 1, 1, 100)
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 0
		lin2 = lineage.Lineage([3], [0.3, 0.4], [], [], [], [], [], [], [ssm3], [])
		lin3 = lineage.Lineage([], [0.25, 0.3], [cnv2], [], [], [], [], [], [], [])
		ssm4 = snp_ssm.SSM()
		ssm4.seg_index = 0
		lin4 = lineage.Lineage([], [0.2, 0.2], [], [], [], [], [], [], [ssm4], [])
		my_lins = [lin0, lin1, lin2, lin3, lin4]
		seg_num = 1

		# supposed results
		z_matrix = [[-1, 1, 1, 1, 1], [-1, -1, -1, -1, 0], [-1, -1, -1, 1, 0], [-1, -1, -1, -1, 0], [-1, -1, -1, -1, -1]]

		my_lineages, z_matrix_returned, avFreqs, ppm = submarine.get_all_possible_z_matrices_with_lineages_new(my_lins, seg_num)

		self.assertEqual(z_matrix, z_matrix_returned.tolist())
		self.assertEqual(my_lineages[4].ssms, [ssm4])
		self.assertEqual(ppm[4].tolist(), [0, 1, 0, 1, 0])

		# example that tests SSM phasing
		lin0 = lineage.Lineage([1, 2, 3, 4, 5], [1.0], [], [], [], [], [], [], [], [])
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		lin1 = lineage.Lineage([2], [0.8], [], [], [], [], [], [], [], [ssm1])
		cnv2 = cnv.CNV(1, 0, 1, 1, 100)
		lin2 = lineage.Lineage([], [0.6], [cnv2], [], [], [], [], [], [], [])
		ssm3 = snp_ssm.SSM()
		ssm3.seg_index = 0
		lin3 = lineage.Lineage([4,5], [0.2], [], [], [], [], [], [], [ssm3], [])
		ssm4 = snp_ssm.SSM()
		ssm4.seg_index = 0
		cnv4 = cnv.CNV(1, 0, 1, 1, 100)
		lin4 = lineage.Lineage([5], [0.19], [cnv4], [], [], [], [], [], [], [ssm4])
		cnv5 = cnv.CNV(1, 0, 1, 1, 100)
		lin5 = lineage.Lineage([], [0.05], [cnv5], [], [], [], [], [], [], [])
		my_lineages = [lin0, lin1, lin2, lin3, lin4, lin5]
		my_lineages_true = copy.deepcopy(my_lineages)
		seg_num = 1

		(my_lineages, z_matrix, avFreqs, ppm) = submarine.get_all_possible_z_matrices_with_lineages_new(my_lineages, seg_num)

		self.assertEqual(my_lineages, my_lineages_true)


	def test_des_are_potential_parents(self):

		zmatrix = [[-1, 1, 1, 1], [-1, -1, 1, 1], [-1, -1 , -1, -1], [-1, -1 , -1, -1]]
		ppm = [[0, 0, 0, 0]] * 4

		self.assertFalse(submarine.des_are_potential_parents(1, 3, zmatrix, ppm))

		
		zmatrix = [[-1, 1, 1, 1], [-1, -1, 1, 1], [-1, -1 , -1, 0], [-1, -1 , -1, -1]]
		ppm = [[0, 0, 0, 0] for i in range(4)]
		ppm[3][2] = 1

		self.assertTrue(submarine.des_are_potential_parents(1, 3, zmatrix, ppm))

	def test_make_def_child(self):

		# 1) k cannot be a child of k* because available frequency is too small
		kstar = 0
		k = 2
		# Z-matrix
		z_matrix = [[-1, 1, 1], [-1, -1, -1], [-1, -1, -1]]
		# possible parent matrix
		ppm = np.asarray([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
		# lineage frequencies
		linFreqs = np.asarray([[1.0, 1.0], [0.9, 0.8], [0.7, 0.2]])
		# available Frequencies
		avFreqs = np.asarray([[0.1, 0.2], [0.9, 0.8], [0.7, 0.2]])
		# Z-matrix & co object
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=None, triplet_ysx=None, triplet_xsy=None, present_ssms=None,
			matrix_after_first_round=z_matrix)
		seg_num = 1
		zero_count = 0
		gain_num = None
		loss_num = None
		CNVs = None
		present_ssms = None
		lin_num = 3
		defparent = [-1, -1, -1]
		initial_pps_for_all = [[], [0], [0]]
			
		with self.assertRaises(eo.NoParentsLeftNoise) as e:
			submarine.make_def_child(kstar, k, k, ppm, defparent, linFreqs, avFreqs, zmco, seg_num, zero_count, gain_num, 
				loss_num, CNVs, present_ssms, initial_pps_for_all=initial_pps_for_all)
		
		# 2) recursive call of make_def_child doesn't work because of available frequency
		kstar = 0
		k = 4
		# Z-matrix
		z_matrix = [[-1, 1, 1, 1, 1], [-1, -1, 0, 0, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]]
		# possible parent matrix
		ppm = np.asarray([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
		# lineage frequencies
		linFreqs = np.asarray([[1.0, 1.0], [0.5, 0.5], [0.4, 0.4], [0.3, 0.3], [0.25, 0.25]])
		# available Frequencies
		avFreqs = np.asarray([[0.5, 0.5], [0.5, 0.5], [0.4, 0.4], [0.3, 0.3]])
		# Z-matrix & co object
		zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
			z_matrix, 5*5, 5)
		zmco = submarine.Z_Matrix_Co(z_matrix=z_matrix, triplet_xys=triplet_xys, triplet_ysx=triplet_ysx, triplet_xsy=triplet_xsy, present_ssms=None,
				matrix_after_first_round=z_matrix)
		seg_num = 1
		zero_count = 0
		gain_num = None
		loss_num = None
		CNVs = None
		present_ssms = None
		lin_num = 5
		defparent = [-1, 0, -1, -1, -1]
		initial_pps_for_all = [[], [0], [0, 1], [0, 1], [0]]
		# check for conflicht with 2
			
		with self.assertRaises(eo.NoParentsLeftNoise) as e:
			submarine.make_def_child(kstar, k, k, ppm, defparent, linFreqs, avFreqs, zmco, seg_num, zero_count, 
				gain_num, loss_num, CNVs, present_ssms, initial_pps_for_all=initial_pps_for_all)
		
		#TODO test trasitivity update which fails

	def test_update_possible_parents_per_child(self):
		z_matrix = [[-1, 1, 1, 1, 1], [-1, -1, -1, 1, 0], [-1, -1, -1, -1, -1],
			[-1, -1, -1, -1, 0], [-1, -1, -1, -1, -1]]
		ppm = np.asarray([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
			[0, 1, 0, 0, 0], [1, 1, 0, 1, 0]])
		k = 1
		kprime = 4

		submarine.update_possible_parents_per_child(z_matrix, ppm, k, kprime)

		self.assertEqual(ppm[4].tolist(), [0, 1, 0, 1, 0])

	def test_get_possible_parents(self):

			# given Z-matrix
			z_matrix = [
				[-1, 1, 1, 1, 1],
				[-1, -1, -1, 1, 0],
				[-1, -1, -1, 0, 1],
				[-1, -1, -1, -1, 0],
				[-1, -1, -1, -1, -1]
				]

			ppm_true = [
				[0, 0, 0, 0, 0],
				[1, 0, 0, 0, 0],
				[1, 0, 0, 0, 0],
				[0, 1, 1, 0, 0],
				[0, 0, 1, 1, 0]
				]

			ppm = submarine.get_possible_parents(z_matrix)

			self.assertTrue(np.array_equal(np.asarray(ppm_true), ppm))

	def test_get_children(self):

			z_matrix = [
				[-1, 1, 1, 1, 1, 1, 1],
				[-1, -1, 1, 1, 0, 1, 1],
				[-1, -1, -1, 1, 0, 1, 0],
				[-1, -1, -1, -1, 0, 1, 0],
				[-1, -1, -1, -1, -1, 0, 0],
				[-1, -1, -1, -1, -1, -1, 0],
				[-1, -1, -1, -1, -1, -1, -1]
				]
			self.assertEqual([2,6], submarine.get_children(z_matrix, 1))
			self.assertEqual([1,4], submarine.get_children(z_matrix, 0))
			self.assertEqual([], submarine.get_children(z_matrix, 5))

	def test_get_0_number_in_z_matrix(self):

			z_matrix = [[-1, 1, 1 ,1], [-1, -1, 0, 1], [-1, -1, -1, 0], [-1, -1, -1, -1]]
			self.assertEqual(2, submarine.get_0_number_in_z_matrix(z_matrix))

	def test_check_CN_influence_user_constraints(self):

		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		cnv1 = cnv.CNV("+1", 0, 1, 1, 10)
		lin0 = lineage.Lineage([1, 2, 3], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([2, 3], 0.3, [], [], [], [], [], [], [ssm1], [])
		lin2 = lineage.Lineage([3], 0.2, [cnv1], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([], 0.1, [], [], [], [], [], [], [], [])
		lineages = [lin0, lin1, lin2, lin3]

		z_matrix = submarine.get_Z_matrix(lineages)[0]
		self.assertEqual(z_matrix, [[-1, 1, 1, 1], [-1, -1, 1, 1], [-1, -1, -1, 1], [-1, -1, -1, -1]])

		submarine.check_CN_influence_user_constraints(z_matrix, lineages)
		self.assertEqual(z_matrix, [[-1, 1, 1, 1], [-1, -1, 1, 0], [-1, -1, -1, 0], [-1, -1, -1, -1]])

		# with user constrainst
		ssm1 = snp_ssm.SSM()
		ssm1.seg_index = 0
		cnv1 = cnv.CNV("+1", 0, 1, 1, 10)
		lin0 = lineage.Lineage([1, 2, 3, 4, 5], 1.0, [], [], [], [], [], [], [], [])
		lin1 = lineage.Lineage([2, 3, 4], 0.3, [], [], [], [], [], [], [ssm1], [])
		lin2 = lineage.Lineage([3, 4], 0.2, [cnv1], [], [], [], [], [], [], [])
		lin3 = lineage.Lineage([4], 0.1, [], [], [], [], [], [], [], [])
		lin4 = lineage.Lineage([], 0.1, [], [], [], [], [], [], [], [])
		lin5 = lineage.Lineage([], 0.05, [], [], [], [], [], [], [], [])
		lineages = [lin0, lin1, lin2, lin3, lin4, lin5]
		user_z = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1], [0, 0, 0, 1, 0, -1], [0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, -1],
		[0, 0, 0, 0, 0, 0]]

		z_matrix = submarine.get_Z_matrix(lineages)[0]
		self.assertEqual(z_matrix, [[-1, 1, 1, 1, 1, 1], [-1, -1, 1, 1, 1, 0], [-1, -1, -1, 1, 1, 0], [-1, -1, -1, -1, 1, 0], 
		[-1, -1, -1, -1, -1, 0], [-1, -1, -1, -1, -1, -1]])

		submarine.check_CN_influence_user_constraints(z_matrix, lineages, user_z)
		self.assertEqual(z_matrix, [[-1, 1, 1, 1, 1, 1], [-1, -1, 1, 0, 0, -1], [-1, -1, -1, 1, 0, -1], [-1, -1, -1, -1, 0, -1],
		[-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]])

	def test_is_CN_influence_present(self):

			k = 1
			k_prime = 2

			# not present
			CN_changes_hash = {}
			CN_changes_hash[k_prime] = {}
			CN_changes_hash[k_prime][0] = [cons.A]
			CN_changes_hash[k_prime][1] = [cons.B]
			SSMs_hash = {}
			SSMs_hash[k] = {}
			SSMs_hash[k][0] = [cons.B]

			self.assertFalse(submarine.is_CN_influence_present(k, k_prime, CN_changes_hash, SSMs_hash))

			# for lineages that don't contain mutations
			self.assertFalse(submarine.is_CN_influence_present(3, 4, CN_changes_hash, SSMs_hash))
			self.assertFalse(submarine.is_CN_influence_present(k, 4, CN_changes_hash, SSMs_hash))

			# influence is present, on A
			CN_changes_hash = {}
			CN_changes_hash[k_prime] = {}
			CN_changes_hash[k_prime][0] = [cons.A]
			SSMs_hash = {}
			SSMs_hash[k] = {}
			SSMs_hash[k][0] = [cons.A]
			self.assertTrue(submarine.is_CN_influence_present(k, k_prime, CN_changes_hash, SSMs_hash))

			# influence is present, on B, but not on first segment
			CN_changes_hash = {}
			CN_changes_hash[k_prime] = {}
			CN_changes_hash[k_prime][0] = [cons.B]
			CN_changes_hash[k_prime][1] = [cons.A, cons.B]
			SSMs_hash = {}
			SSMs_hash[k] = {}
			SSMs_hash[k][0] = [cons.A]
			SSMs_hash[k][1] = [cons.B]
			self.assertTrue(submarine.is_CN_influence_present(k, k_prime, CN_changes_hash, SSMs_hash))


	def test_create_CN_changes_and_SSM_hash_for_LDR(self):

			ssm1 = snp_ssm.SSM()
			ssm1.seg_index = 0
			cnv1 = cnv.CNV("+1", 0, 1, 1, 10)
			ssm2 = snp_ssm.SSM()
			ssm2.seg_index = 1
			cnv2 = cnv.CNV("+1", 2, 1, 1, 10)

			lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
			lin1 = lineage.Lineage([], 0.5, [cnv1, cnv2], [cnv1], [], [], [], [], [ssm1], [ssm1, ssm2])
			lin2 = lineage.Lineage([], 0.4, [], [cnv2], [], [], [], [], [ssm2], [])

			CN_changes, SSMs = submarine.create_CN_changes_and_SSM_hash_for_LDR([lin0, lin1, lin2])
			
			self.assertEqual(len(CN_changes), 2)
			self.assertEqual(CN_changes[1][0], [cons.A, cons.B])
			self.assertEqual(CN_changes[1][2], [cons.A])
			self.assertEqual(CN_changes[2][2], [cons.B])
			self.assertEqual(len(SSMs), 2)
			self.assertEqual(SSMs[1][0], [cons.A, cons.B])
			self.assertEqual(SSMs[1][1], [cons.B])
			self.assertEqual(SSMs[2][1], [cons.A])
			


	def test_add_SSM_appearence_to_hash(self):

			seg_index = 0
			ssm1 = snp_ssm.SSM()
			ssm1.seg_index = seg_index
			SSMs = {}
			lin_index = 1
			phase = cons.A

			# add as first entry
			submarine.add_SSM_appearence_to_hash(SSMs, [ssm1], phase, lin_index)
			self.assertEqual(len(SSMs), 1)
			self.assertEqual(SSMs[lin_index][seg_index], [phase])

			# add something twice, not happens
			submarine.add_SSM_appearence_to_hash(SSMs, [ssm1], phase, lin_index)
			self.assertEqual(len(SSMs), 1)
			self.assertEqual(SSMs[lin_index][seg_index], [phase])

			# add second thing
			phase_2 = cons.B
			submarine.add_SSM_appearence_to_hash(SSMs, [ssm1], phase_2, lin_index)
			self.assertEqual(len(SSMs), 1)
			self.assertEqual(SSMs[lin_index][seg_index], [phase, phase_2])

	def test_add_CN_changes_to_hash(self):

			seg_index = 0
			cnv1 = cnv.CNV("+1", seg_index, 1, 1, 10)
			CN_changes = {}
			lin_index = 1
			phase = cons.A

			# add as first entry
			submarine.add_CN_changes_to_hash(CN_changes, [cnv1], phase, lin_index)
			self.assertEqual(len(CN_changes), 1)
			self.assertEqual(CN_changes[lin_index][seg_index], [phase])

			# add something twice, not possible
			with self.assertRaises(eo.MyException):
				submarine.add_CN_changes_to_hash(CN_changes, [cnv1], phase, lin_index)

			# add second thing
			phase_2 = cons.B
			submarine.add_CN_changes_to_hash(CN_changes, [cnv1], phase_2, lin_index)
			self.assertEqual(len(CN_changes), 1)
			self.assertEqual(CN_changes[lin_index][seg_index], [phase, phase_2])


	def test_adapt_lineages_after_Z_matrix_update(self):
			
			# no forking, Z-matrix wasn't change after first round
			ssm0 = snp_ssm.SSM()
			ssm0.seg_index = 0
			ssm1 = snp_ssm.SSM()
			ssm1.seg_index = 0
			ssm2 = snp_ssm.SSM()
			ssm2.seg_index = 2
			ssm3 = snp_ssm.SSM()
			ssm3.seg_index = 2
			ssm4 = snp_ssm.SSM()
			ssm4.seg_index = 3
			lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
			lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [ssm0, ssm1, ssm3], [], [ssm4])
			lin2 = lineage.Lineage([], 0.6, [], [], [], [], [], [ssm2], [], [])
			my_lineages = [lin0, lin1, lin2]
			z_matrix_fst_rnd = np.asarray([[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]])
			z_matrix_list = [np.asarray([[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]])]

			my_lins, new_lineages_list = submarine.adapt_lineages_after_Z_matrix_update(my_lineages, 
				z_matrix_fst_rnd, z_matrix_list, 
				None, None)

			self.assertEqual(len(new_lineages_list), 1)
			self.assertEqual(new_lineages_list[0], my_lineages)
			self.assertEqual(my_lins, my_lineages)

			# Z-matrix was changed after first round
			ssm0 = snp_ssm.SSM()
			ssm0.seg_index = 0
			ssm1 = snp_ssm.SSM()
			ssm1.seg_index = 0
			ssm2 = snp_ssm.SSM()
			ssm2.seg_index = 2
			ssm3 = snp_ssm.SSM()
			ssm3.seg_index = 2
			ssm4 = snp_ssm.SSM()
			ssm4.seg_index = 3
			lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
			# ssm3 is unphased at beginning, segment 2
			lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [ssm0, ssm1, ssm3], [], [ssm4])
			lin2 = lineage.Lineage([], 0.6, [], [], [], [], [], [ssm2], [], [])
			my_lineages = [lin0, lin1, lin2]
			my_lin1 = copy.deepcopy(lin1)
			my_lin1.ssms = [ssm3]
			my_lin1.ssms_a = [ssm0, ssm1]
			my_lin1.sublins = [2]
			my_lin2 = copy.deepcopy(lin2)
			my_lin2.ssms = []
			my_lin2.ssms_b = [ssm2]
			my_right_lineages = [lin0, my_lin1, my_lin2]
			my_lineages = [lin0, lin1, lin2]
			my_lin2_1 = copy.deepcopy(lin1)
			my_lin2_1.ssms = [ssm3]
			my_lin2_1.ssms_b = [ssm0, ssm1, ssm4]
			my_lin2_1.sublins = [2]
			my_lin2_2 = copy.deepcopy(lin2)
			my_lin2_2.ssms = []
			my_lin2_2.ssms_a = [ssm2]
			my_right_lineages2 = [lin0, my_lin2_1, my_lin2_2]
			lin_num = 3
			seg_num = 4
			# [segment][lineage][A, B, unphased]
			origin_present_ssms = [[[False] * lin_num for _ in range(3)] for x in range(seg_num)]
			origin_present_ssms[0][cons.UNPHASED][1] = True
			origin_present_ssms[2][cons.UNPHASED][2] = True
			origin_present_ssms[2][cons.UNPHASED][1] = True
			origin_present_ssms[3][cons.B][1] = True
			current_ssms_list = copy.deepcopy(origin_present_ssms)
			current_ssms_list[0][cons.UNPHASED][1]= False
			current_ssms_list[0][cons.A][1] = True
			current_ssms_list[2][cons.UNPHASED][2] = False
			current_ssms_list[2][cons.B][2] = True
			current_ssms_list2 = copy.deepcopy(origin_present_ssms)
			current_ssms_list2[0][cons.UNPHASED][1] = False
			current_ssms_list2[0][cons.B][1] = True
			current_ssms_list2[2][cons.UNPHASED][2] = False
			current_ssms_list2[2][cons.A][2] = True
			present_ssms_list = [current_ssms_list, current_ssms_list2]
			i = 0
			present_ssms_list = [current_ssms_list, current_ssms_list2]
			z_matrix_list = [np.asarray([[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]]), 
				np.asarray([[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]])]
			z_matrix_fst_rnd = [np.asarray([[-1, 1, 1], [-1, -1, 0], [-1, -1, -1]])]

			my_lins, new_lineages_list = submarine.adapt_lineages_after_Z_matrix_update(my_lineages, 
				z_matrix_fst_rnd, z_matrix_list, origin_present_ssms, present_ssms_list)

			self.assertEqual(new_lineages_list[0], my_right_lineages)
			self.assertEqual(new_lineages_list[1], my_right_lineages2)
	

	def test_create_updates_lineages(self):
			ssm0 = snp_ssm.SSM()
			ssm0.seg_index = 0
			ssm1 = snp_ssm.SSM()
			ssm1.seg_index = 0
			ssm2 = snp_ssm.SSM()
			ssm2.seg_index = 2
			ssm3 = snp_ssm.SSM()
			ssm3.seg_index = 2
			ssm4 = snp_ssm.SSM()
			ssm4.seg_index = 3
			lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
			lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [ssm0, ssm1, ssm3], [], [ssm4])
			lin2 = lineage.Lineage([], 0.6, [], [], [], [], [], [ssm2], [], [])
			my_lineages = [lin0, lin1, lin2]
			my_lin1 = copy.deepcopy(lin1)
			my_lin1.ssms = [ssm3]
			my_lin1.ssms_a = [ssm0, ssm1]
			my_lin1.sublins = [2]
			my_lin2 = copy.deepcopy(lin2)
			my_lin2.ssms = []
			my_lin2.ssms_b = [ssm2]
			my_right_lineages = [lin0, my_lin1, my_lin2]
			lin_num = 3
			seg_num = 4
			origin_present_ssms = [[[False] * lin_num for _ in range(3)] for x in range(seg_num)]
			origin_present_ssms[0][cons.UNPHASED][1] = True
			origin_present_ssms[2][cons.UNPHASED][2] = True
			origin_present_ssms[2][cons.UNPHASED][1] = True
			origin_present_ssms[3][cons.B][1] = True
			current_ssms_list = copy.deepcopy(origin_present_ssms)
			current_ssms_list[0][cons.UNPHASED][1] = False
			current_ssms_list[0][cons.A][1] = True
			current_ssms_list[2][cons.UNPHASED][2] = False
			current_ssms_list[2][cons.B][2] = True
			i = 5
			present_ssms_list = [[], [], [], [], [], current_ssms_list]
			z_matrix_list = [[], [], [], [], [], np.asarray([[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]])]

			new_lineages = submarine.create_updates_lineages(my_lineages, i, z_matrix_list, origin_present_ssms, 
				present_ssms_list)

			self.assertEqual(new_lineages, my_right_lineages)
			self.assertEqual(my_lineages[1].sublins, [])

	def test_update_SSM_phasing_after_Z_matrix_update(self):

			ssm0 = snp_ssm.SSM()
			ssm0.seg_index = 0
			ssm1 = snp_ssm.SSM()
			ssm1.seg_index = 0
			ssm2 = snp_ssm.SSM()
			ssm2.seg_index = 2
			ssm3 = snp_ssm.SSM()
			ssm3.seg_index = 2
			ssm4 = snp_ssm.SSM()
			ssm4.seg_index = 3
			lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
			lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [ssm0, ssm1, ssm3], [], [ssm4])
			lin2 = lineage.Lineage([], 0.6, [], [], [], [], [], [ssm2], [], [])
			current_lineages = [lin0, lin1, lin2]
			my_lin1 = copy.deepcopy(lin1)
			my_lin1.ssms = [ssm3]
			my_lin1.ssms_a = [ssm0, ssm1]
			my_lin2 = copy.deepcopy(lin2)
			my_lin2.ssms = []
			my_lin2.ssms_b = [ssm2]
			my_right_lineages = [lin0, my_lin1, my_lin2]
			lin_num = 3
			seg_num = 4
			origin_present_ssms = [[[False] * lin_num for _ in range(3)] for x in range(seg_num)]
			origin_present_ssms[0][cons.UNPHASED][1] = True
			origin_present_ssms[2][cons.UNPHASED][2] = True
			origin_present_ssms[2][cons.UNPHASED][1] = True
			origin_present_ssms[3][cons.B][1] = True
			current_ssms_list = copy.deepcopy(origin_present_ssms)
			current_ssms_list[0][cons.UNPHASED][1] = False
			current_ssms_list[0][cons.A][1] = True
			current_ssms_list[2][cons.UNPHASED][2] = False
			current_ssms_list[2][cons.B][2] = True

			submarine.update_SSM_phasing_after_Z_matrix_update(current_lineages, origin_present_ssms, current_ssms_list)

			self.assertEqual(current_lineages, my_right_lineages)

	def test_get_updated_SSM_list(self):
			lin_index = 1
			phase = cons.B
			ssm0 = snp_ssm.SSM()
			ssm0.seg_index = 0
			ssm1 = snp_ssm.SSM()
			ssm1.seg_index = 1
			ssm2 = snp_ssm.SSM()
			ssm2.seg_index = 2
			ssms_per_segments = [[[[], []], [[], []], [[], []]], [[[], []], [[ssm0], [ssm1, ssm2]], [[], []]]]

			new_list = submarine.get_updated_SSM_list(lin_index, phase, ssms_per_segments)

			self.assertEqual(new_list, [ssm0, ssm1, ssm2])


	def test_move_SSMs_in_list_per_segment(self):
			seg_index = 1
			lin_index = 0
			new_phase = cons.A
			old_phase = cons.B
			ssm0 = snp_ssm.SSM()
			ssm0.seg_index = 1
			ssm0.chr = 1
			ssm0.pos = 3
			ssm1 = snp_ssm.SSM()
			ssm1.seg_index = 1
			ssm1.chr = 1
			ssm1.pos = 1
			ssm2 = snp_ssm.SSM()
			ssm2.seg_index = 1
			ssm2.chr = 1
			ssm2.pos = 5
			ssms_per_segments = [[[[], [ssm0]], [[], [ssm1, ssm2]], [[], []]], [[[], []], [[], []], [[], []]]]

			submarine.move_SSMs_in_list_per_segment(ssms_per_segments, seg_index, lin_index, new_phase, old_phase)

			self.assertEqual(ssms_per_segments[lin_index][new_phase][seg_index], [ssm1, ssm0, ssm2])
			self.assertEqual(ssms_per_segments[lin_index][old_phase][seg_index], [])


	def test_get_ssms_per_segments(self):
			ssm0 = snp_ssm.SSM()
			ssm0.seg_index = 0
			ssm1 = snp_ssm.SSM()
			ssm1.seg_index = 0
			ssm2 = snp_ssm.SSM()
			ssm2.seg_index = 2
			ssm3 = snp_ssm.SSM()
			ssm3.seg_index = 2
			ssm4 = snp_ssm.SSM()
			ssm4.seg_index = 3
			lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
			lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [ssm0, ssm1], [], [ssm3, ssm4])
			lin2 = lineage.Lineage([], 0.6, [], [], [], [], [], [], [ssm2], [])
			current_lineages = [lin0, lin1, lin2]
			seg_num = 4

			my_ssms_per_segments = [
				[[[], [], [], []], [[], [], [], []], [[], [], [], []]], 
				[[[], [], [], []], [[], [], [ssm3], [ssm4]], [[ssm0, ssm1], [], [], []]], 
				[[[], [], [ssm2], []], [[], [], [], []], [[], [], [], []]]
				]

			ssms_per_segments = submarine.get_ssms_per_segments(current_lineages, seg_num)

			self.assertEqual(ssms_per_segments, my_ssms_per_segments)
			

	def test_get_ssms_per_segments_lineage_phase(self):
			ssm0 = snp_ssm.SSM()
			ssm0.seg_index = 0
			ssm1 = snp_ssm.SSM()
			ssm1.seg_index = 0
			ssm2 = snp_ssm.SSM()
			ssm2.seg_index = 2
			my_ssms = [ssm0, ssm1, ssm2]
			seg_num = 4

			ssms_per_segment_tmp = submarine.get_ssms_per_segments_lineage_phase(my_ssms, seg_num)

			self.assertEqual(ssms_per_segment_tmp, [[ssm0, ssm1], [], [ssm2], []])

	def test_update_sublineages_after_Z_matrix_update(self):
			lin0 = lineage.Lineage([1, 2], 1.0, [], [], [], [], [], [], [], [])
			lin1 = lineage.Lineage([], 0.8, [], [], [], [], [], [], [], [])
			lin2 = lineage.Lineage([], 0.6, [], [], [], [], [], [], [], [])
			current_lineages = [lin0, lin1, lin2]
			lin_num = 3
			current_z_matrix = np.asarray([
				[-1, 1, 1],
				[-1, -1, 1],
				[-1, -1, -1]
				])

			submarine.update_sublineages_after_Z_matrix_update(current_lineages, current_z_matrix)

			self.assertEqual(lin0.sublins, [1, 2])
			self.assertEqual(lin1.sublins, [2])
			self.assertEqual(lin2.sublins, [])

	def test_has_SSMs_in_segment(self):

			lin_num = 3
			seg_num = 2
			present_ssms_1 = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms_1[0][cons.A][1] = True
			present_ssms_1[0][cons.UNPHASED][2] = True
			present_ssms_2 = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms_2[0][cons.UNPHASED][1] = True
			present_ssms_2[0][cons.B][2] = True
			present_ssms_list = [present_ssms_1, present_ssms_2]

			self.assertTrue(submarine.has_SSMs_in_segment(present_ssms_list, 1, 0))
			self.assertTrue(submarine.has_SSMs_in_segment(present_ssms_list, 2, 0))
			self.assertFalse(submarine.has_SSMs_in_segment(present_ssms_list, 1, 1))
			self.assertFalse(submarine.has_SSMs_in_segment(present_ssms_list, 2, 1))
			
	def test_phasing_allows_relation(self):

			# ancestor-descendant relation between lineages given in matrix after first round
			k = 1
			k_prime = 2
			lin_num = 3
			seg_num = 4
			matrix_after_first_round = [[-1] * lin_num for _ in range(lin_num)]
			matrix_after_first_round[0][1] = 1
			matrix_after_first_round[0][2] = 1
			matrix_after_first_round[k][k_prime] = 1
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			CNVs = []
			CNV_tmp = {}
			CNVs.append(CNV_tmp)

			self.assertTrue(submarine.phasing_allows_relation(k, k_prime, matrix_after_first_round, present_ssms, CNVs, 1))

			# ancestor-descendant relation between lineages between lineages is possible
			k = 1
			k_prime = 2
			lin_num = 3
			seg_num = 4
			matrix_after_first_round = [[-1] * lin_num for _ in range(lin_num)]
			matrix_after_first_round[0][1] = 1
			matrix_after_first_round[0][2] = 1
			matrix_after_first_round[k][k_prime] = 0
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[0][cons.UNPHASED][k] = True
			present_ssms[1][cons.UNPHASED][k] = True
			present_ssms[2][cons.UNPHASED][k] = True
			present_ssms[3][cons.UNPHASED][k] = True
			present_ssms[0][cons.UNPHASED][k_prime] = True
			present_ssms[1][cons.UNPHASED][k_prime] = True
			present_ssms[2][cons.UNPHASED][k_prime] = True
			present_ssms[3][cons.UNPHASED][k_prime] = True
			CNVs = []
			CNV_tmp = {}
			CNVs.append(CNV_tmp)

			self.assertTrue(submarine.phasing_allows_relation(k, k_prime, matrix_after_first_round, present_ssms, CNVs, 1))

			# ancestor-descendant relation between lineages not possible because of SSMs in k and
			# CNVs in k_prime in  A
			k = 1
			k_prime = 2
			lin_num = 3
			seg_num = 4
			matrix_after_first_round = [[-1] * lin_num for _ in range(lin_num)]
			matrix_after_first_round[0][1] = 1
			matrix_after_first_round[0][2] = 1
			matrix_after_first_round[k][k_prime] = 0
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[0][cons.A][k] = True
			present_ssms[1][cons.UNPHASED][k] = True
			present_ssms[2][cons.UNPHASED][k] = True
			present_ssms[3][cons.UNPHASED][k] = True
			present_ssms[0][cons.UNPHASED][k_prime] = True
			present_ssms[1][cons.UNPHASED][k_prime] = True
			present_ssms[2][cons.UNPHASED][k_prime] = True
			present_ssms[3][cons.UNPHASED][k_prime] = True
			CNV_0 = {}
			CNV_0[cons.LOSS] = {}
			CNV_0[cons.LOSS][cons.A] = {}
			CNV_0[cons.LOSS][cons.A][k_prime] = True
			CNV_1 = {}
			CNV_2 = {}
			CNV_3 = {}
			CNVs = [CNV_0, CNV_1, CNV_2, CNV_3]

			with self.assertRaises(eo.ADRelationNotPossible):
				submarine.phasing_allows_relation(k, k_prime, matrix_after_first_round, present_ssms, CNVs, 1)

			# ancestor-descendant relation between lineages not possible because of SSMs in k and
			# CNVs in k_prime in  B
			k = 1
			k_prime = 2
			lin_num = 3
			seg_num = 4
			matrix_after_first_round = [[-1] * lin_num for _ in range(lin_num)]
			matrix_after_first_round[0][1] = 1
			matrix_after_first_round[0][2] = 1
			matrix_after_first_round[k][k_prime] = 0
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[0][cons.UNPHASED][k] = True
			present_ssms[1][cons.B][k] = True
			present_ssms[2][cons.UNPHASED][k] = True
			present_ssms[3][cons.UNPHASED][k] = True
			present_ssms[0][cons.UNPHASED][k_prime] = True
			present_ssms[1][cons.UNPHASED][k_prime] = True
			present_ssms[2][cons.UNPHASED][k_prime] = True
			present_ssms[3][cons.UNPHASED][k_prime] = True
			CNV_0 = {}
			CNV_1 = {}
			CNV_1[cons.LOSS] = {}
			CNV_1[cons.LOSS][cons.B] = {}
			CNV_1[cons.LOSS][cons.B][k_prime] = True
			CNV_2 = {}
			CNV_3 = {}
			CNVs = [CNV_0, CNV_1, CNV_2, CNV_3]

			with self.assertRaises(eo.ADRelationNotPossible):
				submarine.phasing_allows_relation(k, k_prime, matrix_after_first_round, present_ssms, CNVs, 1)

			# ancestor-descendant relation between lineages not possible because of SSMs in k_prime and
			# CNVs in k in  A
			k = 1
			k_prime = 2
			lin_num = 3
			seg_num = 4
			matrix_after_first_round = [[-1] * lin_num for _ in range(lin_num)]
			matrix_after_first_round[0][1] = 1
			matrix_after_first_round[0][2] = 1
			matrix_after_first_round[k][k_prime] = 0
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[0][cons.UNPHASED][k] = True
			present_ssms[1][cons.UNPHASED][k] = True
			present_ssms[2][cons.UNPHASED][k] = True
			present_ssms[3][cons.UNPHASED][k] = True
			present_ssms[0][cons.UNPHASED][k_prime] = True
			present_ssms[1][cons.UNPHASED][k_prime] = True
			present_ssms[2][cons.A][k_prime] = True
			present_ssms[3][cons.UNPHASED][k_prime] = True
			CNV_0 = {}
			CNV_1 = {}
			CNV_2 = {}
			CNV_2[cons.LOSS] = {}
			CNV_2[cons.LOSS][cons.A] = {}
			CNV_2[cons.LOSS][cons.A][k] = True
			CNV_3 = {}
			CNVs = [CNV_0, CNV_1, CNV_2, CNV_3]

			with self.assertRaises(eo.ADRelationNotPossible):
				submarine.phasing_allows_relation(k, k_prime, matrix_after_first_round, present_ssms, CNVs, 1)

			# ancestor-descendant relation between lineages not possible because of SSMs in k_prime and
			# CNVs in k in  B
			k = 1
			k_prime = 2
			lin_num = 3
			seg_num = 4
			matrix_after_first_round = [[-1] * lin_num for _ in range(lin_num)]
			matrix_after_first_round[0][1] = 1
			matrix_after_first_round[0][2] = 1
			matrix_after_first_round[k][k_prime] = 0
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[0][cons.UNPHASED][k] = True
			present_ssms[1][cons.UNPHASED][k] = True
			present_ssms[2][cons.UNPHASED][k] = True
			present_ssms[3][cons.UNPHASED][k] = True
			present_ssms[0][cons.UNPHASED][k_prime] = True
			present_ssms[1][cons.UNPHASED][k_prime] = True
			present_ssms[2][cons.UNPHASED][k_prime] = True
			present_ssms[3][cons.B][k_prime] = True
			CNV_0 = {}
			CNV_1 = {}
			CNV_2 = {}
			CNV_3 = {}
			CNV_3[cons.LOSS] = {}
			CNV_3[cons.LOSS][cons.B] = {}
			CNV_3[cons.LOSS][cons.B][k] = True
			CNVs = [CNV_0, CNV_1, CNV_2, CNV_3]

			with self.assertRaises(eo.ADRelationNotPossible):
				submarine.phasing_allows_relation(k, k_prime, matrix_after_first_round, present_ssms, CNVs, 1)

	def test_phasing_allows_relation_per_allele_lineage(self):

			# no phased SSMs
			lineage_ssm = 1
			lineage_cnv = 2
			lin_num = 4
			seg_num = 5
			seg_index = 3
			phase = cons.A
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[seg_index][cons.UNPHASED][lineage_ssm] = True
			CNVs = []
			CNV_tmp = {}
			CNVs.append(CNV_tmp)

			self.assertTrue(submarine.phasing_allows_relation_per_allele_lineage(lineage_ssm, lineage_cnv, 
				present_ssms, CNVs, phase, seg_index))

			# phased to other allele
			lineage_ssm = 1
			lineage_cnv = 2
			lin_num = 4
			seg_num = 5
			seg_index = 3
			phase = cons.A
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[seg_index][submarine.other_phase(phase)][lineage_ssm] = True
			CNVs = []
			CNV_tmp = {}
			CNVs.append(CNV_tmp)

			self.assertTrue(submarine.phasing_allows_relation_per_allele_lineage(lineage_ssm, lineage_cnv, 
				present_ssms, CNVs, phase, seg_index))

			# no CN change in same phase as SSMs but in other
			lineage_ssm = 1
			lineage_cnv = 2
			lin_num = 4
			seg_num = 5
			seg_index = 3
			phase = cons.A
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[seg_index][phase][lineage_ssm] = True
			CNVs = []
			CNV_0 = {}
			CNV_1 = {}
			CNV_2 = {}
			CNV_tmp = {}
			CNV_tmp[cons.GAIN] = {}
			CNV_tmp[cons.GAIN][submarine.other_phase(phase)] = {}
			CNV_tmp[cons.GAIN][submarine.other_phase(phase)][lineage_cnv] = True
			CNVs = [CNV_0, CNV_1, CNV_2, CNV_tmp]

			self.assertTrue(submarine.phasing_allows_relation_per_allele_lineage(lineage_ssm, lineage_cnv, 
				present_ssms, CNVs, phase, seg_index))

			# lineage_ssm < lineage_cnv, with gain
			lineage_ssm = 1
			lineage_cnv = 2
			lin_num = 4
			seg_num = 5
			seg_index = 3
			phase = cons.A
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[seg_index][phase][lineage_ssm] = True
			CNVs = []
			CNV_0 = {}
			CNV_1 = {}
			CNV_2 = {}
			CNV_tmp = {}
			CNV_tmp[cons.GAIN] = {}
			CNV_tmp[cons.GAIN][phase] = {}
			CNV_tmp[cons.GAIN][phase][lineage_cnv] = True
			CNVs = [CNV_0, CNV_1, CNV_2, CNV_tmp]

			with self.assertRaises(eo.ADRelationNotPossible):
				submarine.phasing_allows_relation_per_allele_lineage(lineage_ssm, lineage_cnv,
					present_ssms, CNVs, phase, seg_index)

			# lineage_ssm < lineage_cnv, with loss
			lineage_ssm = 1
			lineage_cnv = 2
			lin_num = 4
			seg_num = 5
			seg_index = 3
			phase = cons.A
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[seg_index][phase][lineage_ssm] = True
			CNVs = []
			CNV_0 = {}
			CNV_1 = {}
			CNV_2 = {}
			CNV_tmp = {}
			CNV_tmp[cons.LOSS] = {}
			CNV_tmp[cons.LOSS][phase] = {}
			CNV_tmp[cons.LOSS][phase][lineage_cnv] = True
			CNVs = [CNV_0, CNV_1, CNV_2, CNV_tmp]

			with self.assertRaises(eo.ADRelationNotPossible):
				submarine.phasing_allows_relation_per_allele_lineage(lineage_ssm, lineage_cnv,
					present_ssms, CNVs, phase, seg_index)

			# lineage_ssm > lineage_cnv, with loss
			lineage_ssm = 2
			lineage_cnv = 1
			lin_num = 4
			seg_num = 5
			seg_index = 3
			phase = cons.A
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[seg_index][phase][lineage_ssm] = True
			CNVs = []
			CNV_0 = {}
			CNV_1 = {}
			CNV_2 = {}
			CNV_tmp = {}
			CNV_tmp[cons.LOSS] = {}
			CNV_tmp[cons.LOSS][phase] = {}
			CNV_tmp[cons.LOSS][phase][lineage_cnv] = True
			CNVs = [CNV_0, CNV_1, CNV_2, CNV_tmp]

			with self.assertRaises(eo.ADRelationNotPossible):
				submarine.phasing_allows_relation_per_allele_lineage(lineage_ssm, lineage_cnv,
					present_ssms, CNVs, phase, seg_index)

			# lineage_ssm > lineage_cnv, with gain, ok
			lineage_ssm = 2
			lineage_cnv = 1
			lin_num = 4
			seg_num = 5
			seg_index = 3
			phase = cons.A
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[seg_index][phase][lineage_ssm] = True
			CNVs = []
			CNV_0 = {}
			CNV_1 = {}
			CNV_2 = {}
			CNV_tmp = {}
			CNV_tmp[cons.GAIN] = {}
			CNV_tmp[cons.GAIN][phase] = {}
			CNV_tmp[cons.GAIN][phase][lineage_cnv] = True
			CNVs = [CNV_0, CNV_1, CNV_2, CNV_tmp]

			self.assertTrue(submarine.phasing_allows_relation_per_allele_lineage(lineage_ssm, lineage_cnv,
				present_ssms, CNVs, phase, seg_index))

	def test_unphased_checking(self):

			# no unphased SSMs
			lineage_ssm = 2
			lineage_cnv = 3
			lin_num = 4
			seg_num = 5
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			CNVs = []
			CNV_tmp = {}
			CNVs.append(CNV_tmp)

			submarine.unphased_checking(lineage_ssm, lineage_cnv, present_ssms, CNVs)
			my_present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			self.assertEqual(my_present_ssms, present_ssms)

			# only one segment with unphased SSMs and no CNVs in this segment
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			seg_index = 0
			present_ssms[seg_index][cons.UNPHASED][lineage_ssm] = True
			CNVs = []
			CNV_tmp = {}
			CNVs.append(CNV_tmp)

			submarine.unphased_checking(lineage_ssm, lineage_cnv, present_ssms, CNVs)
			my_present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			my_present_ssms[seg_index][cons.UNPHASED][lineage_ssm] = True
			self.assertEqual(my_present_ssms, present_ssms)

			# four segments with unphased SSMs
			# influence of gain A, gain B, loss A and loss B
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[0][cons.UNPHASED][lineage_ssm] = True
			present_ssms[1][cons.UNPHASED][lineage_ssm] = True
			present_ssms[2][cons.UNPHASED][lineage_ssm] = True
			present_ssms[3][cons.UNPHASED][lineage_ssm] = True
			CNV_0 = {}
			CNV_0[cons.GAIN] = {}
			CNV_0[cons.GAIN][cons.A] = {}
			CNV_0[cons.GAIN][cons.A][lineage_cnv] = True
			CNV_1 = {}
			CNV_1[cons.GAIN] = {}
			CNV_1[cons.GAIN][cons.B] = {}
			CNV_1[cons.GAIN][cons.B][lineage_cnv] = True
			CNV_2 = {}
			CNV_2[cons.LOSS] = {}
			CNV_2[cons.LOSS][cons.A] = {}
			CNV_2[cons.LOSS][cons.A][lineage_cnv] = True
			CNV_3 = {}
			CNV_3[cons.LOSS] = {}
			CNV_3[cons.LOSS][cons.B] = {}
			CNV_3[cons.LOSS][cons.B][lineage_cnv] = True
			CNVs = [CNV_0, CNV_1, CNV_2, CNV_3]

			submarine.unphased_checking(lineage_ssm, lineage_cnv, present_ssms, CNVs)
			my_present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			my_present_ssms[0][cons.B][lineage_ssm] = True
			my_present_ssms[1][cons.A][lineage_ssm] = True
			my_present_ssms[2][cons.B][lineage_ssm] = True
			my_present_ssms[3][cons.A][lineage_ssm] = True
			self.assertEqual(my_present_ssms, present_ssms)

	def test_get_CNVs_of_lineage(self):
			lineage_cnv = 2
			lineage_ssm = 3
			seg_index = 0

			# no CNVs
			CNVs = []
			CNV_tmp = {}
			CNVs.append(CNV_tmp)

			with self.assertRaises(eo.no_CNVs):
				submarine.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)

			# one CN gain on A
			CNVs = []
			CNV_tmp = {}
			mutation_type = cons.GAIN
			phase = cons.A
			CNV_tmp[mutation_type] = {}
			CNV_tmp[mutation_type][phase] = {}
			CNV_tmp[mutation_type][phase][lineage_cnv] = True
			CNVs.append(CNV_tmp)

			loss_a, loss_b, gain_a, gain_b = submarine.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)
			self.assertEqual((False, False, True, False), (loss_a, loss_b, gain_a, gain_b))

			# one CN gain on B
			CNVs = []
			CNV_tmp = {}
			mutation_type = cons.GAIN
			phase = cons.B
			CNV_tmp[mutation_type] = {}
			CNV_tmp[mutation_type][phase] = {}
			CNV_tmp[mutation_type][phase][lineage_cnv] = True
			CNVs.append(CNV_tmp)

			loss_a, loss_b, gain_a, gain_b = submarine.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)
			self.assertEqual((False, False, False, True), (loss_a, loss_b, gain_a, gain_b))

			# one CN loss on A
			CNVs = []
			CNV_tmp = {}
			mutation_type = cons.LOSS
			phase = cons.A
			CNV_tmp[mutation_type] = {}
			CNV_tmp[mutation_type][phase] = {}
			CNV_tmp[mutation_type][phase][lineage_cnv] = True
			CNVs.append(CNV_tmp)

			loss_a, loss_b, gain_a, gain_b = submarine.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)
			self.assertEqual((True, False, False, False), (loss_a, loss_b, gain_a, gain_b))

			# one CN loss on B
			CNVs = []
			CNV_tmp = {}
			mutation_type = cons.LOSS
			phase = cons.B
			CNV_tmp[mutation_type] = {}
			CNV_tmp[mutation_type][phase] = {}
			CNV_tmp[mutation_type][phase][lineage_cnv] = True
			CNVs.append(CNV_tmp)

			loss_a, loss_b, gain_a, gain_b = submarine.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)
			self.assertEqual((False, True, False, False), (loss_a, loss_b, gain_a, gain_b))
	
			# two CN gains
			CNVs = []
			CNV_tmp = {}
			mutation_type = cons.GAIN
			phase = cons.A
			lineage_cnv = 3
			lineage_ssm = 2
			CNV_tmp[mutation_type] = {}
			CNV_tmp[mutation_type][phase] = {}
			CNV_tmp[mutation_type][phase][lineage_cnv] = True
			CNV_tmp[mutation_type][submarine.other_phase(phase)] = {}
			CNV_tmp[mutation_type][submarine.other_phase(phase)][lineage_cnv] = True
			CNVs.append(CNV_tmp)

			with self.assertRaises(eo.MyException):
				submarine.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)

			# two CN losses
			lineage_cnv = 2
			lineage_ssm = 3
			CNVs = []
			CNV_tmp = {}
			mutation_type = cons.LOSS
			phase = cons.A
			CNV_tmp[mutation_type] = {}
			CNV_tmp[mutation_type][phase] = {}
			CNV_tmp[mutation_type][phase][lineage_cnv] = True
			CNV_tmp[mutation_type][submarine.other_phase(phase)] = {}
			CNV_tmp[mutation_type][submarine.other_phase(phase)][lineage_cnv] = True
			CNVs.append(CNV_tmp)

			with self.assertRaises(eo.MyException):
				submarine.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)

			# loss and gain on A and B
			CNVs = []
			CNV_tmp = {}
			lineage_cnv = 3
			lineage_ssm = 2
			mutation_type = cons.GAIN
			mutation_type_2 = cons.LOSS
			phase = cons.A
			CNV_tmp[mutation_type] = {}
			CNV_tmp[mutation_type][phase] = {}
			CNV_tmp[mutation_type][phase][lineage_cnv] = True
			CNV_tmp[mutation_type_2] = {}
			CNV_tmp[mutation_type_2][submarine.other_phase(phase)] = {}
			CNV_tmp[mutation_type_2][submarine.other_phase(phase)][lineage_cnv] = True
			CNVs.append(CNV_tmp)

			with self.assertRaises(eo.MyException):
				submarine.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)

			# loss and gain on B and A
			lineage_cnv = 3
			lineage_ssm = 2
			CNVs = []
			CNV_tmp = {}
			mutation_type = cons.GAIN
			mutation_type_2 = cons.LOSS
			phase = cons.B
			CNV_tmp[mutation_type] = {}
			CNV_tmp[mutation_type][phase] = {}
			CNV_tmp[mutation_type][phase][lineage_cnv] = True
			CNV_tmp[mutation_type_2] = {}
			CNV_tmp[mutation_type_2][submarine.other_phase(phase)] = {}
			CNV_tmp[mutation_type_2][submarine.other_phase(phase)][lineage_cnv] = True
			CNVs.append(CNV_tmp)

			with self.assertRaises(eo.MyException):
				submarine.get_CNVs_of_lineage(lineage_cnv, CNVs, seg_index, lineage_ssm)

	def test_has_CNV_in_phase(self):
			seg_index = 0
			mutation_type = cons.GAIN
			phase = cons.A
			CNVs = []
			CNVs_tmp = {}
			CNVs_tmp[mutation_type] = {}
			CNVs_tmp[mutation_type][phase] = {}
			CNVs_tmp[mutation_type][phase][2] = True
			CNVs_tmp[mutation_type][phase][4] = True
			CNVs.append(CNVs_tmp)
	
			# lineage index contained
			lineage_cnv = 2
			self.assertTrue(submarine.has_CNV_in_phase(lineage_cnv, CNVs, seg_index, mutation_type, phase))

			# lineage index not contained
			lineage_cnv = 3
			self.assertFalse(submarine.has_CNV_in_phase(lineage_cnv, CNVs, seg_index, mutation_type, phase))

			# wrong phase
			phase = cons.B
			self.assertFalse(submarine.has_CNV_in_phase(lineage_cnv, CNVs, seg_index, mutation_type, phase))

			# wrong mutation type
			mutation_type = cons.LOSS
			self.assertFalse(submarine.has_CNV_in_phase(lineage_cnv, CNVs, seg_index, mutation_type, phase))

	def test_cn_change_influences_ssms(self):

			# lineage_ssm < lineage_cnv, mutation = cons.LOSS
			# there is change!
			lin_num = 4
			seg_num = 6
			lineage_ssm = 2
			lineage_cnv = 3
			seg_index = 3
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[seg_index][cons.UNPHASED][lineage_ssm] = True
			phase = cons.A
			mutation = cons.LOSS

			submarine.cn_change_influences_ssms(lineage_ssm, lineage_cnv, present_ssms, seg_index, mutation, phase)
			my_present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			my_present_ssms[seg_index][submarine.other_phase(phase)][lineage_ssm] = True
			self.assertEqual(present_ssms, my_present_ssms)
			
			# lineage_ssm < lineage_cnv, mutation = cons.GAIN
			# there is change!
			lin_num = 4
			seg_num = 6
			lineage_ssm = 2
			lineage_cnv = 3
			seg_index = 3
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[seg_index][cons.UNPHASED][lineage_ssm] = True
			phase = cons.A
			mutation = cons.GAIN

			submarine.cn_change_influences_ssms(lineage_ssm, lineage_cnv, present_ssms, seg_index, mutation, phase)
			my_present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			my_present_ssms[seg_index][submarine.other_phase(phase)][lineage_ssm] = True
			self.assertEqual(present_ssms, my_present_ssms)

			# lineage_ssm > lineage_cnv, mutation = cons.LOSS
			# there is change!
			lin_num = 4
			seg_num = 6
			lineage_ssm = 3
			lineage_cnv = 2
			seg_index = 3
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[seg_index][cons.UNPHASED][lineage_ssm] = True
			phase = cons.A
			mutation = cons.LOSS

			submarine.cn_change_influences_ssms(lineage_ssm, lineage_cnv, present_ssms, seg_index, mutation, phase)
			my_present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			my_present_ssms[seg_index][submarine.other_phase(phase)][lineage_ssm] = True
			self.assertEqual(present_ssms, my_present_ssms)

			# lineage_ssm > lineage_cnv, mutation = cons.GAIN
			# there is change!
			lin_num = 4
			seg_num = 6
			lineage_ssm = 3
			lineage_cnv = 2
			seg_index = 3
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[seg_index][cons.UNPHASED][lineage_ssm] = True
			phase = cons.A
			mutation = cons.GAIN

			submarine.cn_change_influences_ssms(lineage_ssm, lineage_cnv, present_ssms, seg_index, mutation, phase)
			my_present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			my_present_ssms[seg_index][cons.UNPHASED][lineage_ssm] = True
			self.assertEqual(present_ssms, my_present_ssms)

	def test_move_unphased_SSMs(self):

			lin_num = 4
			seg_num = 6
			current_lin = 2
			seg_index = 3
			present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			present_ssms[seg_index][cons.UNPHASED][current_lin] = True
			phase = cons.A

			submarine.move_unphased_SSMs(present_ssms, seg_index, current_lin, phase)
			my_present_ssms = [[[False] * lin_num for _ in range(3)] for _ in range(seg_num)]
			my_present_ssms[seg_index][phase][current_lin] = True
			self.assertEqual(present_ssms, my_present_ssms)

			with self.assertRaises(eo.MyException):
				present_ssms[seg_index][submarine.other_phase(phase)][current_lin] = True
				submarine.move_unphased_SSMs(present_ssms, seg_index, current_lin, phase)
			

	def test_get_CN_changes_SSM_apperance(self):
			# 4 lineages, 3 segments
			# lin1 only has unphased SSM in seg0
			# lin2 only one loss in seg1, lin3 also one in seg1
			# seg2: CN gains in all lineages
			lin0 = lineage.Lineage([1, 2, 3], 1.0, None, None, None, None, None, None, None, None)
			ssm1 = snp_ssm.SNP_SSM()
			ssm1.pos = 1
			ssm1.seg_index = 0
			ssm11 = snp_ssm.SNP_SSM()
			ssm11.pos = 11
			ssm11.seg_index = 1
			ssm11.infl_cnv_same_lin = True
			cnv12 = cnv.CNV(1, 2, 1, 21, 30)
			cnv122 = cnv.CNV(1, 2, 1, 21, 30)
			lin1 = lineage.Lineage([2, 3], 1.0, [cnv12], [cnv122], None, None, None, [ssm1], [ssm11], None)
			cnv2 = cnv.CNV(-1, 1, 1, 11, 20)
			ssm21 = snp_ssm.SNP_SSM()
			ssm21.pos = 12
			ssm21.seg_index = 1
			cnv22 = cnv.CNV(1, 2, 1, 21, 30)
			lin2 = lineage.Lineage([3], 1.0, None, [cnv2, cnv22], None, None, None, None, None, [ssm21])
			cnv3 = cnv.CNV(-1, 1, 1, 11, 20)
			cnv32 = cnv.CNV(1, 2, 1, 21, 30)
			cnv322 = cnv.CNV(1, 2, 1, 21, 30)
			lin3 = lineage.Lineage([], 1.0, [cnv3, cnv32], [cnv322], None, None, None, None, None, None)

			seg_num = 3
			gain_num = []
			loss_num = []
			CNVs = [] 
			present_ssms = [] 
			ssm_infl_cnv_same_lineage = []
			my_lineages = [lin0, lin1, lin2, lin3]
			lineage_num = len(my_lineages)

			submarine.get_CN_changes_SSM_apperance(seg_num, gain_num, loss_num, CNVs, present_ssms, lineage_num, my_lineages,
				ssm_infl_cnv_same_lineage)

			self.assertEqual(gain_num, [0, 0, 5])
			self.assertEqual(loss_num, [0, 2, 0])
			self.assertEqual(present_ssms[0], [[False, False, False, False], [False, False, False, False], 
				[False, True, False, False]])
			self.assertEqual(present_ssms[1], [[False, True, False, False], [False, False, True, False], 
				[False, False, False, False]])
			self.assertEqual(present_ssms[2], [[False, False, False, False], [False, False, False, False],
				[False, False, False, False]])
			self.assertEqual(ssm_infl_cnv_same_lineage[0], [[False, False, False, False], [False, False, False, False]])
			self.assertEqual(ssm_infl_cnv_same_lineage[1], [[False, True, False, False], [False, False, False, False]])
			self.assertEqual(ssm_infl_cnv_same_lineage[2], [[False, False, False, False], [False, False, False, False]])
			self.assertEqual(len(CNVs[0].keys()), 0)
			self.assertEqual(len(CNVs[1][cons.LOSS].keys()), 2)
			self.assertEqual(len(CNVs[2][cons.GAIN].keys()), 2)
			self.assertEqual(sorted(CNVs[2][cons.GAIN][cons.A].keys()), [1, 3])


	def test_check_crossing_rule_function(self):
			# crossing rule fulfilled
			lin0 = lineage.Lineage([1, 2], [1.0, 1.0, 1.0], None, None, None, None, None, None, None, None)
			lin1 = lineage.Lineage([], [1.0, 0.8, 1.0], None, None, None, None, None, None, None, None)
			lin2 = lineage.Lineage([], [0.8, 0.7, 0.9], None, None, None, None, None, None, None, None)
			lins = [lin0, lin1, lin2]
			z_matrix = [[-1, 1, 1], [-1, -1, 0], [-1, -1, -1]]
			z_matrix_ori = copy.deepcopy(z_matrix)
			zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
					z_matrix, 1, 3)
			
			zero_count = submarine.check_crossing_rule_function(lins, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy)

			self.assertEqual(zero_count, 1)
			self.assertEqual(z_matrix, z_matrix_ori)

			# crossing rule violated
			lin0 = lineage.Lineage([1, 2], [1.0, 1.0, 1.0], None, None, None, None, None, None, None, None)
			lin1 = lineage.Lineage([], [1.0, 0.8, 1.0], None, None, None, None, None, None, None, None)
			lin2 = lineage.Lineage([], [0.8, 0.81, 0.9], None, None, None, None, None, None, None, None)
			lins = [lin0, lin1, lin2]
			z_matrix = [[-1, 1, 1], [-1, -1, 0], [-1, -1, -1]]
			zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
					z_matrix, 1, 3)
			
			zero_count = submarine.check_crossing_rule_function(lins, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy)

			self.assertEqual(zero_count, 0)
			self.assertEqual(z_matrix[1][2], -1)

			# crossing rule violated, leads to Z-matrix update
			lin0 = lineage.Lineage([1, 2], [1.0, 1.0, 1.0], None, None, None, None, None, None, None, None)
			lin1 = lineage.Lineage([], [1.0, 0.8, 1.0], None, None, None, None, None, None, None, None)
			lin2 = lineage.Lineage([2], [0.8, 0.81, 0.9], None, None, None, None, None, None, None, None)
			lin3 = lineage.Lineage([], [0.6, 0.5, 0.4], None, None, None, None, None, None, None, None)
			lins = [lin0, lin1, lin2, lin3]
			z_matrix = [[-1, 1, 1, 1], [-1, -1, 0, 0], [-1, -1, -1, 1], [-1, -1, -1, -1]]
			zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
					z_matrix, 2, 4)
			
			zero_count = submarine.check_crossing_rule_function(lins, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy)

			self.assertEqual(zero_count, 0)
			self.assertEqual(z_matrix[1][2], -1)
			self.assertEqual(z_matrix[1][3], -1)

			# crossing rule violated, entry was already set to 1
			lin0 = lineage.Lineage([1, 2], [1.0, 1.0, 1.0], None, None, None, None, None, None, None, None)
			lin1 = lineage.Lineage([2], [1.0, 0.8, 1.0], None, None, None, None, None, None, None, None)
			lin2 = lineage.Lineage([], [0.8, 0.81, 0.9], None, None, None, None, None, None, None, None)
			lins = [lin0, lin1, lin2]
			z_matrix = [[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]]
			zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
					z_matrix, 2, 3)
			
			
			with self.assertRaises(eo.MyException):
				zero_count = submarine.check_crossing_rule_function(lins, z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy)

	def test_check_and_update_complete_Z_matrix(self):
			# minimal filled Z matrix
			# will get filled completely with 1's
			lin0 = lineage.Lineage([1, 2, 3, 4, 5, 6, 7], 0, None, None, None, None, None, None, None, None)
			lin1 = lineage.Lineage([2], 0, None, None, None, None, None, None, None, None)
			lin2 = lineage.Lineage([3], 0, None, None, None, None, None, None, None, None)
			lin3 = lineage.Lineage([4], 0, None, None, None, None, None, None, None, None)
			lin4 = lineage.Lineage([5], 0, None, None, None, None, None, None, None, None)
			lin5 = lineage.Lineage([6], 0, None, None, None, None, None, None, None, None)
			lin6 = lineage.Lineage([7], 0, None, None, None, None, None, None, None, None)
			lin7 = lineage.Lineage([], 0, None, None, None, None, None, None, None, None)
			lins = [lin0, lin1, lin2, lin3, lin4, lin5, lin6, lin7]

			z_matrix = submarine.get_Z_matrix(lins)[0]
			zero_count = submarine.get_0_number_in_z_matrix(z_matrix)
			lineage_num = len(lins)
			zero_count, triplet_xys, triplet_ysx, triplet_xsy = submarine.check_and_update_complete_Z_matrix_from_matrix(
				z_matrix, zero_count, lineage_num)
			self.assertEqual(z_matrix, [[-1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, 1, 1, 1, 1, 1, 1],
				[-1, -1, -1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, 1, 1, 1],
				[-1, -1, -1, -1, -1, -1, 1, 1], [-1, -1, -1, -1, -1, -1, -1, 1], [-1, -1, -1, -1, -1, -1, -1, -1]])
			self.assertEqual(zero_count, 0)
			self.assertEqual(triplet_xys, {})
			self.assertEqual(triplet_ysx, {})
			self.assertEqual(triplet_xsy, {})
			

	def test_update_Z_matrix_iteratively(self):
			# no triplets in hashes, no influence by changed index pair
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			z_matrix = [[0] * 4 for _ in range(4)]
			my_z_matrix = [[0] * 4 for _ in range(4)]
			zero_count = 0
			index_pair = (2, 3)

			zero_count = submarine.update_Z_matrix_iteratively(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy, 
				index_pair)
			self.assertEqual(z_matrix, my_z_matrix)
			self.assertEqual(zero_count, 0)

			# triplets with 0's exist, but non are influences by the changed index pair
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			submarine.update_triplet_hash(triplet_xys, 1, 2, 3)
			submarine.update_triplet_hash(triplet_ysx, 2, 3, 1)
			submarine.update_triplet_hash(triplet_xsy, 1, 3, 2)
			z_matrix = [[0] * 12 for _ in range(12)]
			my_z_matrix = [[0] * 12 for _ in range(12)]
			zero_count = 1
			index_pair = (5,7)

			zero_count = submarine.update_Z_matrix_iteratively(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy, 
				index_pair)
			self.assertEqual(z_matrix, my_z_matrix)
			self.assertEqual(zero_count, 1)

			# 4 triplets, at least one in each triplet category
			# 1 triplet in each category is changed
			# 1 triplet is not changed
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			submarine.update_triplet_hash(triplet_xys, 5, 7, 8)
			submarine.update_triplet_hash(triplet_xsy, 5, 7, 6)
			submarine.update_triplet_hash(triplet_ysx, 5, 7, 1)
			submarine.update_triplet_hash(triplet_ysx, 5, 7, 2)
			z_matrix = [[0] * 12 for _ in range(12)]
			z_matrix[5][7] = 1
			z_matrix[5][8] = -1
			z_matrix[6][7] = 1
			z_matrix[1][7] = 1
			my_z_matrix = [[0] * 12 for _ in range(12)]
			my_z_matrix[5][7] = 1
			my_z_matrix[5][8] = -1
			my_z_matrix[6][7] = 1
			my_z_matrix[1][7] = 1
			my_z_matrix[7][8] = -1
			my_z_matrix[5][6] = 1
			my_z_matrix[1][5] = 1
			zero_count = 4
			index_pair = (5,7)

			zero_count = submarine.update_Z_matrix_iteratively(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy,
				index_pair)
			self.assertEqual(z_matrix, my_z_matrix)
			self.assertEqual(zero_count, 1)
			self.assertEqual(triplet_xys, {})
			self.assertEqual(triplet_xsy, {})
			self.assertEqual(triplet_ysx[5][7][2], True)
			self.assertEqual(len(triplet_ysx.keys()), 1)

			# iterative update
			# 3 triplets, influence each other iterativly
			# 4th triplet includes value that gets changes before triplet is processed
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			submarine.update_triplet_hash(triplet_xys, 5, 7, 9)
			submarine.update_triplet_hash(triplet_ysx, 5, 7, 4)
			submarine.update_triplet_hash(triplet_ysx, 5, 9, 1)
			submarine.update_triplet_hash(triplet_xsy, 1, 5, 4)
			z_matrix = [[0] * 12 for _ in range(12)]
			z_matrix[5][7] = 1
			z_matrix[7][9] = 1
			z_matrix[1][9] = 1
			z_matrix[1][4] = -1
			z_matrix[4][7] = -1
			my_z_matrix = [[0] * 12 for _ in range(12)]
			my_z_matrix[5][7] = 1
			my_z_matrix[7][9] = 1
			my_z_matrix[1][9] = 1
			my_z_matrix[1][4] = -1
			my_z_matrix[4][7] = -1
			my_z_matrix[5][9] = 1
			my_z_matrix[1][5] = 1
			my_z_matrix[4][5] = -1
			zero_count = 4
			index_pair = (5,7)

			zero_count = submarine.update_Z_matrix_iteratively(z_matrix, zero_count, triplet_xys, triplet_ysx, triplet_xsy,
				index_pair)
			self.assertEqual(z_matrix, my_z_matrix)
			self.assertEqual(zero_count, 1)
			self.assertEqual(triplet_xys, {})
			self.assertEqual(triplet_xsy, {})
			self.assertEqual(triplet_ysx, {})

	def test_remove_triplet_from_all_hashes(self):
			# add one entry to all three hashes manually
			# then remove them all with the function to be tested
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			x = 1
			y = 2
			s = 3
			submarine.update_triplet_hash(triplet_xys, x, y, s)
			submarine.update_triplet_hash(triplet_ysx, y, s, x)
			submarine.update_triplet_hash(triplet_xsy, x, s, y)

			submarine.remove_triplet_from_all_hashes(triplet_xys, triplet_ysx, triplet_xsy, x, y, s)
			self.assertEqual(triplet_xys, {})
			self.assertEqual(triplet_ysx, {})
			self.assertEqual(triplet_xsy, {})

	def test_remove_triplet_from_hash(self):
			# hash with more entries at second index
			my_hash = {}
			submarine.update_triplet_hash(my_hash, 1, 2, 3)
			submarine.update_triplet_hash(my_hash, 1, 2, 4)

			submarine.remove_triplet_from_hash(my_hash, 1, 2, 4)
			self.assertEqual(list(my_hash[1][2].keys()), [3])

			# hash with only one entry, is empty afterwards
			submarine.remove_triplet_from_hash(my_hash, 1, 2, 3)
			self.assertEqual(my_hash, {})

	def test_update_triplet_hash(self):
			my_hash = {}
			submarine.update_triplet_hash(my_hash, 1, 2, 3)
			self.assertTrue(my_hash[1][2][3])

	def test_check_1f_2d_2g_2j_losses_gains(self):
			# two losses on two alleles in two lineages with relation
			# lower lineages can't be descendants if they have SSMs
			# lineage has SSMs
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][1] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][2] = "something"
			loss_num = 2
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[1][2] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[1][2] = 1
			my_z_matrix[2][3] = -1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][3] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.LOSS)
			self.assertEqual(zero_count, 1)
			self.assertEqual(z_matrix, my_z_matrix)

			# loss and gain on two alleles in two lineages with relation
			# lower lineages can be descendants
			# lineage has SSMs
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][1] = "something"
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.B] = {}
			CNVs[cons.GAIN][cons.B][2] = "something"
			loss_num = 1
			gain_num = 1
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[1][2] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[1][2] = 1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][3] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.LOSS, mut_num_B=gain_num, mutations_B=cons.GAIN)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# loss and gain on two alleles in two lineages with relation
			# lower lineages can be descendants
			# lineage has SSMs
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][1] = "something"
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][2] = "something"
			loss_num = 1
			gain_num = 1
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[1][2] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[1][2] = 1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][3] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(gain_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.GAIN, mut_num_B=loss_num, mutations_B=cons.LOSS)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# loss and gain on two alleles in two lineages with relation
			# lower lineages can be descendants
			# lineage has SSMs
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][1] = "something"
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][2] = "something"
			loss_num = 1
			gain_num = 1
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[1][2] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[1][2] = 1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][3] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(gain_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.GAIN, mut_num_B=loss_num, mutations_B=cons.LOSS)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# loss and gain on two alleles in two lineages with relation
			# lower lineages can be descendants
			# lineage has SSMs
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.B] = {}
			CNVs[cons.GAIN][cons.B][1] = "something"
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][2] = "something"
			loss_num = 1
			gain_num = 1
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[1][2] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[1][2] = 1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][3] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.LOSS, mut_num_B=gain_num, mutations_B=cons.GAIN)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# two losses on two alleles in two lineages with relation
			# lower lineages can't be descendants if they have SSMs
			# lineage has no SSMs
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][1] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][2] = "something"
			loss_num = 2
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[1][2] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[1][2] = 1
			present_ssms = [[False] * lin_num for _ in range(3)]
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.LOSS)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# two losses on two alleles in two lineages without known relation
			# nothing can be said about lower lineages
			# lineage has SSMs
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][1] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][2] = "something"
			loss_num = 2
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][3] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.LOSS)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# two losses on two alleles in two lineages without relation
			# nothing can be said about lower lineages
			# lineage has SSMs
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][1] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][2] = "something"
			loss_num = 2
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[1][2] = -1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[1][2] = -1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][3] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.LOSS)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# two losses on two alleles in two lineages with relation
			# higher lineage has SSM
			# can't be ancestor of both
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][2] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][3] = "something"
			loss_num = 2
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[2][3] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[2][3] = 1
			my_z_matrix[1][2] = -1
			my_z_matrix[1][3] = -1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][1] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.LOSS)
			self.assertEqual(zero_count, 0)
			self.assertEqual(z_matrix, my_z_matrix)

			# two losses on two alleles in two lineages with relation
			# higher lineage has SSM
			# is only ancestor of one --> not possible
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][2] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][3] = "something"
			loss_num = 2
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[2][3] = 1
			z_matrix[1][3] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[2][3] = 1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][1] = True
			zero_count = 2
			first_run = True

			with self.assertRaises(eo.MyException):
				submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, present_ssms, 
					triplet_xys, triplet_ysx, triplet_xsy,
					first_run, mutations=cons.LOSS)

			# two losses on two alleles in two lineages with relation
			# higher lineage has no SSMs
			# no result can be drawn
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][2] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][3] = "something"
			loss_num = 2
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[2][3] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[2][3] = 1
			present_ssms = [[False] * lin_num for _ in range(3)]
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.LOSS)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# two gains on two alleles in two lineages with relation
			# higher lineage has SSM
			# can't be ancestor of both
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][2] = "something"
			CNVs[cons.GAIN][cons.B] = {}
			CNVs[cons.GAIN][cons.B][3] = "something"
			gain_num = 2
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[2][3] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[2][3] = 1
			my_z_matrix[1][2] = -1
			my_z_matrix[1][3] = -1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][1] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(gain_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy, 
				first_run, mutations=cons.GAIN)
			self.assertEqual(zero_count, 0)
			self.assertEqual(z_matrix, my_z_matrix)

			# loss and gain on two alleles in two lineages with relation
			# higher lineage has SSM
			# can't be ancestor of both
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][2] = "something"
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][3] = "something"
			gain_num = 1
			loss_num = 1
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[2][3] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[2][3] = 1
			my_z_matrix[1][2] = -1
			my_z_matrix[1][3] = -1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][1] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(gain_num, CNVs, z_matrix, 
			zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy, 
				first_run, mutations=cons.GAIN, mut_num_B=loss_num, 
			mutations_B=cons.LOSS)
			self.assertEqual(zero_count, 0)
			self.assertEqual(z_matrix, my_z_matrix)

			# loss and gain on two alleles in two lineages with relation
			# higher lineage has SSM
			# can't be ancestor of both
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.B] = {}
			CNVs[cons.GAIN][cons.B][2] = "something"
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][3] = "something"
			gain_num = 1
			loss_num = 1
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[2][3] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[2][3] = 1
			my_z_matrix[1][2] = -1
			my_z_matrix[1][3] = -1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][1] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, 
				zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy, 
				first_run, mutations=cons.LOSS, mut_num_B=gain_num, 
			mutations_B=cons.GAIN)
			self.assertEqual(zero_count, 0)
			self.assertEqual(z_matrix, my_z_matrix)

		# loss and gain on two alleles in two lineages with relation
			# higher lineage has SSM
			# can't be ancestor of both
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][2] = "something"
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.B] = {}
			CNVs[cons.GAIN][cons.B][3] = "something"
			gain_num = 1
			loss_num = 1
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[2][3] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[2][3] = 1
			my_z_matrix[1][2] = -1
			my_z_matrix[1][3] = -1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][1] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, 
				zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy, 
				first_run, mutations=cons.LOSS, mut_num_B=gain_num, 
			mutations_B=cons.GAIN)
			self.assertEqual(zero_count, 0)
			self.assertEqual(z_matrix, my_z_matrix)

			# loss and gain on two alleles in two lineages with relation
			# higher lineage has SSM
			# can't be ancestor of both
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][2] = "something"
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][3] = "something"
			gain_num = 1
			loss_num = 1
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[2][3] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[2][3] = 1
			my_z_matrix[1][2] = -1
			my_z_matrix[1][3] = -1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][1] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(gain_num, CNVs, z_matrix, 
				zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy, 
				first_run, mutations=cons.GAIN, mut_num_B=loss_num, 
			mutations_B=cons.LOSS)
			self.assertEqual(zero_count, 0)
			self.assertEqual(z_matrix, my_z_matrix)

			# two losses on two alleles in two lineages with relation
			# middle lineages can't be in ancestor-descendant relation if they have unphased SSMs
			# lineage has unphased SSMs
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][1] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][3] = "something"
			loss_num = 2
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[1][3] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[1][3] = 1
			my_z_matrix[2][3] = -1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][2] = True
			zero_count = 1
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.LOSS)
			self.assertEqual(zero_count, 0)
			self.assertEqual(z_matrix, my_z_matrix)

			# loss and gain on two alleles in two lineages with relation
			# middle lineages can be in ancestor-descendant relation if they have unphased SSMs
		# and lower lineager has loss
			# lineage has unphased SSMs
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][1] = "something"
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][3] = "something"
			loss_num = 1
			gain_num = 1
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[1][3] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[1][3] = 1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][2] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(gain_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.GAIN, mut_num_B=loss_num, mutations_B=cons.LOSS)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# loss and gain on two alleles in two lineages with relation
			# middle lineages can be in ancestor-descendant relation if they have unphased SSMs
			# and lower lineager has loss
			# lineage has unphased SSMs
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.B] = {}
			CNVs[cons.GAIN][cons.B][1] = "something"
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][3] = "something"
			loss_num = 1
			gain_num = 1
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[1][3] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[1][3] = 1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][2] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.LOSS, mut_num_B=gain_num, mutations_B=cons.GAIN)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)
			
			# loss and gain on two alleles in two lineages with relation
			# middle lineages can't be in ancestor-descendant relation if they have unphased SSMs
			# and higher lineager has loss
			# lineage has unphased SSMs
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][1] = "something"
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.B] = {}
			CNVs[cons.GAIN][cons.B][3] = "something"
			loss_num = 1
			gain_num = 1
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[1][3] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[1][3] = 1
			my_z_matrix[2][3] = -1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][2] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.LOSS, mut_num_B=gain_num, mutations_B=cons.GAIN)
			self.assertEqual(zero_count, 1)
			self.assertEqual(z_matrix, my_z_matrix)

			# loss and gain on two alleles in two lineages with relation
			# middle lineages can't be in ancestor-descendant relation if they have unphased SSMs
			# and higher lineager has loss
			# lineage has unphased SSMs
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][1] = "something"
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][3] = "something"
			loss_num = 1
			gain_num = 1
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[1][3] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[1][3] = 1
			my_z_matrix[2][3] = -1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][2] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(gain_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.GAIN, mut_num_B=loss_num, mutations_B=cons.LOSS)
			self.assertEqual(zero_count, 1)
			self.assertEqual(z_matrix, my_z_matrix)

			# two losses on two alleles in two lineages with relation
			# middle lineages can't be in ancestor-descendant relation if they have unphased SSMs
			# lineage has no unphased SSMs
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][1] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][3] = "something"
			loss_num = 2
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[1][3] = 1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[1][3] = 1
			present_ssms = [[False] * lin_num for _ in range(3)]
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.LOSS)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# two losses on two alleles in two lineages without known relation
			# nothing can be said about middle lineages
			# lineage has unphased SSMs
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][1] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][3] = "something"
			loss_num = 2
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][2] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.LOSS)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# two losses on two alleles in two lineages without relation
			# nothing can be said about middle lineages
			# lineage has unphased SSMs
			# first run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][1] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][3] = "something"
			loss_num = 2
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[1][3] = -1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[1][3] = -1
			present_ssms = [[False] * lin_num for _ in range(3)]
			present_ssms[cons.UNPHASED][2] = True
			zero_count = 2
			first_run = True

			zero_count = submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, z_matrix, zero_count, 
				present_ssms, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.LOSS)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# two losses on two alleles in two lineages without relation
			# nothing can be said about lower lineages
			# lineage has SSMs
			# second run
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][1] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][2] = "something"
			loss_num = 2
			lin_num = 4
			z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			z_matrix[1][2] = -1
			my_z_matrix = [[0] * (lin_num) for _ in range(lin_num)]
			my_z_matrix[1][2] = -1
			present_ssms = [[[False] * lin_num for _ in range(3)]]
			present_ssms[0][cons.UNPHASED][3] = True
			zero_count = 2
			first_run = False
			z_matrix_fst_rnd = copy.deepcopy(z_matrix)
			z_matrix_list = [z_matrix]
			triplets_list = [[triplet_xys, triplet_ysx, triplet_xsy]]
			present_ssms_list = [present_ssms]
			CNVs_all = [CNVs]
			# function to test
			submarine.check_1f_2d_2g_2j_losses_gains(loss_num, CNVs, None, zero_count,
				None, triplet_xys, triplet_ysx, triplet_xsy,
				first_run, mutations=cons.LOSS, z_matrix_fst_rnd=z_matrix_fst_rnd, z_matrix_list=z_matrix_list,
				triplets_list=triplets_list, present_ssms_list=present_ssms_list, seg_index=0, CNVs_all=CNVs_all)

			self.assertEqual(len(z_matrix_list), 1)
			self.assertEqual(len(triplets_list), 1)
			self.assertEqual(len(present_ssms_list), 1)

	def test_check_2i_phased_changes(self):
			# no influence of upstream lineages
			# current example is not possible in practice but should test everything
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.GAIN] = {} 
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][3] = "something"
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][2] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			z_matrix[1][2] = 1
			my_z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix[1][2] = 1 
			my_z_matrix[1][3] = -1
			my_z_matrix[2][3] = -1
			present_ssms = [[False] * 4 for _ in range(3)]
			present_ssms[cons.A][1] = True
			present_ssms[cons.B][1] = True
			present_ssms[cons.A][2] = True
			zero_count = 2

			zero_count = submarine.check_2i_phased_changes(CNVs, z_matrix, zero_count, present_ssms, 
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 0)
			self.assertEqual(z_matrix, my_z_matrix)

			# some influence to upstream lineages
			# loss in A
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][3] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			z_matrix[1][3] = 1
			my_z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix[1][3] = 1
			my_z_matrix[2][3] = -1
			present_ssms = [[False] * 4 for _ in range(3)]
			present_ssms[cons.A][1] = True
			present_ssms[cons.A][2] = True
			zero_count = 2

			zero_count = submarine.check_2i_phased_changes(CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 1)
			self.assertEqual(z_matrix, my_z_matrix)

			# loss in B
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][3] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			z_matrix[1][3] = 1
			my_z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix[1][3] = 1
			my_z_matrix[2][3] = -1
			present_ssms = [[False] * 4 for _ in range(3)]
			present_ssms[cons.B][1] = True
			present_ssms[cons.B][2] = True
			zero_count = 2

			zero_count = submarine.check_2i_phased_changes(CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 1)
			self.assertEqual(z_matrix, my_z_matrix)

			# gain in A
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][3] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			z_matrix[1][3] = 1
			my_z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix[1][3] = 1
			my_z_matrix[2][3] = -1
			present_ssms = [[False] * 4 for _ in range(3)]
			present_ssms[cons.A][1] = True
			present_ssms[cons.A][2] = True
			zero_count = 2

			zero_count = submarine.check_2i_phased_changes(CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 1)
			self.assertEqual(z_matrix, my_z_matrix)

			# gain in B
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.B] = {}
			CNVs[cons.GAIN][cons.B][3] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			z_matrix[1][3] = 1
			my_z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix[1][3] = 1
			my_z_matrix[2][3] = -1
			present_ssms = [[False] * 4 for _ in range(3)]
			present_ssms[cons.B][1] = True
			present_ssms[cons.B][2] = True
			zero_count = 2

			zero_count = submarine.check_2i_phased_changes(CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 1)
			self.assertEqual(z_matrix, my_z_matrix)

			# gain in B
			# adding a -1 to the matrix has an influence on further triplets and thus fields in the matrix
			triplet_xys = {}
			submarine.update_triplet_hash(triplet_xys, 2, 3, 4)
			triplet_ysx = {}
			triplet_xsy = {}
			CNVs = {}
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.B] = {}
			CNVs[cons.GAIN][cons.B][3] = "something"
			z_matrix = [[0] * (5) for _ in range(5)]
			z_matrix[1][3] = 1
			z_matrix[3][4] = 1
			my_z_matrix = [[0] * (5) for _ in range(5)]
			my_z_matrix[1][3] = 1
			my_z_matrix[2][3] = -1
			my_z_matrix[3][4] = 1
			my_z_matrix[2][4] = -1
			present_ssms = [[False] * 4 for _ in range(3)]
			present_ssms[cons.B][1] = True
			present_ssms[cons.B][2] = True
			zero_count = 2

			zero_count = submarine.check_2i_phased_changes(CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 0)
			self.assertEqual(z_matrix, my_z_matrix)

	def test_check_2h_LOH(self):
			# loss and gain in different lineages
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			gain_num = 1
			loss_num = 1
			CNVs = {}
			CNVs[cons.GAIN] = {} 
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][3] = "something"
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][2] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			present_ssms = [[False] * 4 for _ in range(3)]
			zero_count = 2

			submarine.check_2h_LOH(loss_num, gain_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 2)

			# LOH but no SSMs
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			gain_num = 1
			loss_num = 1
			CNVs = {}
			CNVs[cons.GAIN] = {} 
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][3] = "something"
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][3] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix = [[0] * (4) for _ in range(4)]
			present_ssms = [[False] * 4 for _ in range(3)]
			zero_count = 2

			zero_count = submarine.check_2h_LOH(loss_num, gain_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# LOH with upstream SSMs, some are already in relation
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			gain_num = 1
			loss_num = 1
			CNVs = {}
			CNVs[cons.GAIN] = {} 
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][3] = "something"
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][3] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			z_matrix[1][3] = 1
			my_z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix[1][3] = 1
			my_z_matrix[2][3] = -1
			present_ssms = [[False] * 4 for _ in range(3)]
			present_ssms[cons.A][1] = True
			present_ssms[cons.B][2] = True
			zero_count = 2

			zero_count = submarine.check_2h_LOH(loss_num, gain_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 1)
			self.assertEqual(z_matrix, my_z_matrix)

			# LOH in different lineages with upstream SSMs
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			gain_num = 3
			loss_num = 3
			CNVs = {}
			CNVs[cons.GAIN] = {} 
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][1] = "something"
			CNVs[cons.GAIN][cons.A][2] = "something"
			CNVs[cons.GAIN][cons.A][3] = "something"
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][2] = "something"
			CNVs[cons.LOSS][cons.B][3] = "something"
			CNVs[cons.LOSS][cons.B][4] = "something"
			z_matrix = [[0] * (5) for _ in range(5)]
			my_z_matrix = [[0] * (5) for _ in range(5)]
			my_z_matrix[1][3] = -1
			my_z_matrix[1][2] = -1
			present_ssms = [[False] * 5 for _ in range(3)]
			present_ssms[cons.A][1] = True
			zero_count = 6

			zero_count = submarine.check_2h_LOH(loss_num, gain_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 4)
			self.assertEqual(z_matrix, my_z_matrix)


	def test_check_2f_CN_gains(self):
			# CN gains on both alleles and in different lineages
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			gain_num = 2
			CNVs = {}
			CNVs[cons.GAIN] = {} 
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][3] = "something"
			CNVs[cons.GAIN][cons.B] = {}
			CNVs[cons.GAIN][cons.B][2] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			present_ssms = [[False] * 4 for _ in range(3)]
			zero_count = 2

			zero_count = submarine.check_2f_CN_gains(gain_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 2)

			# CN on both alleles and in the same lineages, no upstream SSMs
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			gain_num = 2
			CNVs = {}
			CNVs[cons.GAIN] = {} 
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][3] = "something"
			CNVs[cons.GAIN][cons.B] = {}
			CNVs[cons.GAIN][cons.B][3] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix = [[0] * (4) for _ in range(4)]
			present_ssms = [[False] * 4 for _ in range(3)]
			zero_count = 2

			zero_count = submarine.check_2f_CN_gains(gain_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# CN on one alleles and in the different lineages, no upstream SSMs
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			gain_num = 2
			CNVs = {}
			CNVs[cons.GAIN] = {} 
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][3] = "something"
			CNVs[cons.GAIN][cons.A][1] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix = [[0] * (4) for _ in range(4)]
			present_ssms = [[False] * 4 for _ in range(3)]
			zero_count = 2

			zero_count = submarine.check_2f_CN_gains(gain_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# CN on both alleles and in the same lineages, multiple upstream SSMs
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			gain_num = 2
			CNVs = {}
			CNVs[cons.GAIN] = {} 
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][3] = "something"
			CNVs[cons.GAIN][cons.B] = {}
			CNVs[cons.GAIN][cons.B][3] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			z_matrix[1][3] = 1
			my_z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix[1][3] = 1
			my_z_matrix[2][3] = -1
			
			present_ssms = [[False] * 4 for _ in range(3)]
			present_ssms[cons.UNPHASED][1] = True
			present_ssms[cons.UNPHASED][2] = True
			zero_count = 2

			zero_count = submarine.check_2f_CN_gains(gain_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 1)
			self.assertEqual(z_matrix, my_z_matrix)

	def test_check_1d_2c_CN_losses(self):

			# CN losses on both alleles and different lineages
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			loss_num = 2
			CNVs = {}
			CNVs[cons.LOSS] = {} 
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][3] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][2] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			present_ssms = [[False] * 4 for _ in range(3)]
			zero_count = 2

			zero_count = submarine.check_1d_2c_CN_losses(loss_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 2)

			# CN losses on both alleles and in same lineage, no lower or higher lineage with SSMs
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			loss_num = 2
			CNVs = {}
			CNVs[cons.LOSS] = {} 
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][2] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][2] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix = [[0] * (4) for _ in range(4)] 
			present_ssms = [[False] * 4 for _ in range(3)]
			zero_count = 2

			zero_count = submarine.check_1d_2c_CN_losses(loss_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)


			# CN losses on both alleles and in same lineage, lower lineages have SSMs
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			loss_num = 2
			CNVs = {}
			CNVs[cons.LOSS] = {} 
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][2] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][2] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix = [[0] * (4) for _ in range(4)] 
			my_z_matrix[2][3] = -1
			present_ssms = [[False] * 4 for _ in range(3)]
			present_ssms[cons.UNPHASED][3] = True
			zero_count = 2

			zero_count = submarine.check_1d_2c_CN_losses(loss_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 1)
			self.assertEqual(z_matrix, my_z_matrix)

			# CN losses on both alleles and in same lineage, lower lineages have SSMs as well as higher
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			loss_num = 2
			CNVs = {}
			CNVs[cons.LOSS] = {} 
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][2] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][2] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix = [[0] * (4) for _ in range(4)] 
			my_z_matrix[2][3] = -1
			my_z_matrix[1][2] = -1
			present_ssms = [[False] * 4 for _ in range(3)]
			present_ssms[cons.UNPHASED][3] = True
			present_ssms[cons.UNPHASED][1] = True
			zero_count = 2

			zero_count = submarine.check_1d_2c_CN_losses(loss_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 0)
			self.assertEqual(z_matrix, my_z_matrix)

			# CN losses on both alleles and in same lineage, higher lineage has SSMs but is phased
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			loss_num = 2
			CNVs = {}
			CNVs[cons.LOSS] = {} 
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][2] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][2] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			z_matrix[1][2] = 1
			my_z_matrix = [[0] * (4) for _ in range(4)] 
			my_z_matrix[1][2] = 1
			present_ssms = [[False] * 4 for _ in range(3)]
			present_ssms[cons.UNPHASED][1] = True
			zero_count = 2

			zero_count = submarine.check_1d_2c_CN_losses(loss_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# CN losses on only one allele and same lineage
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			loss_num = 2
			CNVs = {}
			CNVs[cons.LOSS] = {} 
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][3] = "something"
			CNVs[cons.LOSS][cons.A][3] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			present_ssms = [[False] * 4 for _ in range(3)]
			zero_count = 2

			zero_count = submarine.check_1d_2c_CN_losses(loss_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 2)
	
	def test_check_1c_CN_loss(self):

			# already lowest lineage
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			loss_num = 1
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][3] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix = [[0] * (4) for _ in range(4)]
			zero_count = 2
			phase = cons.A
			present_ssms = [[False] * 4 for _ in range(3)]

			zero_count = submarine.check_1c_CN_loss(loss_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# lower lineage has no SSMs phased to A
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			loss_num = 1
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][1] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix = [[0] * (4) for _ in range(4)]
			zero_count = 2
			phase = cons.A
			present_ssms = [[False] * 4 for _ in range(3)]
			present_ssms[cons.B][2] = True
			present_ssms[cons.UNPHASED][2] = True

			zero_count = submarine.check_1c_CN_loss(loss_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 2)
			self.assertEqual(z_matrix, my_z_matrix)

			# lower lineage 2 has SSMs phased to B, lineage 3 doesn't have SSMs
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			loss_num = 1
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][1] = "something"
			z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix = [[0] * (4) for _ in range(4)]
			my_z_matrix[1][2] = -1
			zero_count = 2
			phase = cons.B
			present_ssms = [[False] * 4 for _ in range(3)]
			present_ssms[cons.B][2] = True
			present_ssms[cons.UNPHASED][2] = True

			zero_count = submarine.check_1c_CN_loss(loss_num, CNVs, z_matrix, zero_count, present_ssms,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 1)
			self.assertEqual(z_matrix, my_z_matrix)


	def test_check_1a_CN_LOSS(self):

			# only one loss
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			loss_num = 1
			CNVs = {}
			z_matrix = None
			zero_count = 1

			zero_count = submarine.check_1a_CN_LOSS(loss_num, CNVs, z_matrix, zero_count,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 1)

			# 2 losses, same lineage, different alleles
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			loss_num = 2
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][3] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][3] = "something" 
			z_matrix = None
			zero_count = 1

			zero_count = submarine.check_1a_CN_LOSS(loss_num, CNVs, z_matrix, zero_count,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 1)

			# 2 losses, different lineages, different allele
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			loss_num = 2
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][3] = "something"
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][5] = "something" 
			z_matrix = None
			zero_count = 1

			zero_count = submarine.check_1a_CN_LOSS(loss_num, CNVs, z_matrix, zero_count,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 1)

			# 2 losses, different lineages, same allele
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			loss_num = 2
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.A] = {}
			CNVs[cons.LOSS][cons.A][1] = "something"
			CNVs[cons.LOSS][cons.A][2] = "something" 
			z_matrix = [[0] * 3 for _ in range(3)]
			my_z_matrix = [[0] * 3 for _ in range(3)]
			my_z_matrix[1][2] = -1
			zero_count = 2

			zero_count = submarine.check_1a_CN_LOSS(loss_num, CNVs, z_matrix, zero_count,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 1)
			self.assertEqual(z_matrix, my_z_matrix)

			# 3 losses, three pairs, three pairs lead to -1, whereas one already has a -1
			triplet_xys = {}
			triplet_ysx = {}
			triplet_xsy = {}
			loss_num = 3
			CNVs = {}
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][1] = "something"
			CNVs[cons.LOSS][cons.B][2] = "something" 
			CNVs[cons.LOSS][cons.B][3] = "something" 
			z_matrix = [[0] * 4 for _ in range(4)]
			z_matrix[2][3] = -1
			my_z_matrix = [[0] * 4 for _ in range(4)]
			my_z_matrix[2][3] = -1
			my_z_matrix[1][2] = -1
			my_z_matrix[1][3] = -1
			zero_count = 3

			zero_count = submarine.check_1a_CN_LOSS(loss_num, CNVs, z_matrix, zero_count,
				triplet_xys, triplet_ysx, triplet_xsy)
			self.assertEqual(zero_count, 1)
			self.assertEqual(z_matrix, my_z_matrix)

	def test_get_present_ssms(self):
			ssm_1 = snp_ssm.SNP_SSM()
			ssm_1.seg_index = 0
			ssm_2 = snp_ssm.SNP_SSM()
			ssm_2.seg_index = 0
			ssm_a_1 = snp_ssm.SNP_SSM()
			ssm_a_1.seg_index = 0
			ssm_b_1 = snp_ssm.SNP_SSM()
			ssm_b_1.seg_index = 3
			ssm_b_2 = snp_ssm.SNP_SSM()
			ssm_b_2.seg_index = 6

			lin1 = lineage.Lineage([], 0, None, None, None, None, None, [ssm_1, ssm_2], None, 
				[ssm_b_1, ssm_b_2])
			lin2 = lineage.Lineage([], 0, None, None, None, None, None, None, [ssm_a_1], None)
			lins = [lin1, lin2]

			# unphased SSMs are present in segment 0 and the first lineage
			onctos_present_ssms = [[False] * 2 for _ in range(3)]
			lin_index = 0
			seg_index = 0
			phase = cons.UNPHASED
			ssms_index_list = [0, 0]
			submarine.get_present_ssms(onctos_present_ssms, lin_index, lins, seg_index, phase, 
				ssms_index_list)
			my_present_ssms = [[False] * 2 for _ in range(3)]
			my_present_ssms[cons.UNPHASED][0] = True
			self.assertEqual(onctos_present_ssms, my_present_ssms)
			self.assertEqual(ssms_index_list, [2, 0])

			# no unphased SSMs are present for this position in ssm index list
			onctos_present_ssms = [[False] * 2 for _ in range(3)]
			lin_index = 0
			seg_index = 1
			phase = cons.UNPHASED
			ssms_index_list = [10, 0]
			submarine.get_present_ssms(onctos_present_ssms, lin_index, lins, seg_index, phase, 
				ssms_index_list)
			my_present_ssms = [[False] * 2 for _ in range(3)]
			self.assertEqual(onctos_present_ssms, my_present_ssms)
			self.assertEqual(ssms_index_list, [10, 0])

			# phased to A SSMs are present in second lineage
			onctos_present_ssms = [[False] * 2 for _ in range(3)]
			lin_index = 1
			seg_index = 0
			phase = cons.A
			ssms_index_list = [0, 0]
			submarine.get_present_ssms(onctos_present_ssms, lin_index, lins, seg_index, phase, 
				ssms_index_list)
			my_present_ssms = [[False] * 2 for _ in range(3)]
			my_present_ssms[cons.A][1] = True
			self.assertEqual(onctos_present_ssms, my_present_ssms)
			self.assertEqual(ssms_index_list, [0, 1])

			# phased to B in first lineage, other segment
			onctos_present_ssms = [[False] * 2 for _ in range(3)]
			lin_index = 0
			seg_index = 6
			phase = cons.B
			ssms_index_list = [1, 0]
			submarine.get_present_ssms(onctos_present_ssms, lin_index, lins, seg_index, phase, 
				ssms_index_list)
			my_present_ssms = [[False] * 2 for _ in range(3)]
			my_present_ssms[cons.B][0] = True
			self.assertEqual(onctos_present_ssms, my_present_ssms)
			self.assertEqual(ssms_index_list, [2, 0])


	def test_is_it_LOH(self):

			cnv1 = cnv.CNV(1, 0, 1, 1, 1)
			cnv2 = cnv.CNV(-1, 0, 1, 1, 1)

			# proper LOH
			CNVs = {}
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][1] = cnv1
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][1] = cnv2
			gain_num = 1
			loss_num = 1
			self.assertTrue(submarine.is_it_LOH(gain_num, loss_num, CNVs))

			# only CN changes in one direction
			gain_num = 1
			loss_num = 0
			self.assertFalse(submarine.is_it_LOH(gain_num, loss_num, CNVs))

			# too many CN changes
			gain_num = 1
			loss_num = 2
			with self.assertRaises(eo.NotProperLOH):
				submarine.is_it_LOH(gain_num, loss_num, CNVs)

			# CN changes on different chromosome
			gain_num = 1
			loss_num = 1
			CNVs = {}
			CNVs[cons.GAIN] = {}
			CNVs[cons.GAIN][cons.A] = {}
			CNVs[cons.GAIN][cons.A][1] = cnv1
			CNVs[cons.LOSS] = {}
			CNVs[cons.LOSS][cons.B] = {}
			CNVs[cons.LOSS][cons.B][2] = cnv2
			with self.assertRaises(eo.NotProperLOH):
				submarine.is_it_LOH(gain_num, loss_num, CNVs)


	def test_add_CN_change_to_hash(self):
			# two CN changes, gain and loss, on two different segments
			cnv1 = cnv.CNV(3, 0, 1, 1, 1)	
			cnv2 = cnv.CNV(-1, 2, 1, 1, 1)	
			cnv_list = [cnv1, cnv2]
			# other CN change on other alleles
			cnv3 = cnv.CNV(-1, 0, 1, 1, 1)
			cnv_list_b = [cnv3]
			# lineages
			lin0 = lineage.Lineage([1, 2, 3], 0, None, None, None, None, None, None, None, None)
			lin1 = lineage.Lineage([], 0, cnv_list, None, None, None, None, None, None, None)
			lin2 = lineage.Lineage([], 0, None, cnv_list_b, None, None, None, None, None, None)
			lin_list = [lin0, lin1, lin2]

			# test insertion of cnv, phase A, cnv is present
			CNVs = {}
			gain_num = 0
			loss_num = 0
			phase = cons.A
			lin_index = 1
			seg_index = 0
			cnv_index_list = [0, 0, 0]
			gain_num, loss_num = submarine.add_CN_change_to_hash(lin_list, lin_index, seg_index, 
				CNVs, gain_num, loss_num, phase, cnv_index_list)
			self.assertEqual(gain_num, 1)
			self.assertEqual(loss_num, 0)
			self.assertEqual(cnv_index_list, [0, 1, 0])
			self.assertEqual(CNVs[cons.GAIN][cons.A][1], cnv1)

			# test insertion of cnv, phase B, cnv is present
			CNVs = {}
			gain_num = 0
			loss_num = 0
			phase = cons.B
			lin_index = 2
			seg_index = 0
			cnv_index_list = [0, 0, 0]
			gain_num, loss_num = submarine.add_CN_change_to_hash(lin_list, lin_index, seg_index, 
				CNVs, gain_num, loss_num, phase, cnv_index_list)
			self.assertEqual(gain_num, 0)
			self.assertEqual(loss_num, 1)
			self.assertEqual(cnv_index_list, [0, 0, 1])
			self.assertEqual(CNVs[cons.LOSS][cons.B][2], cnv3)

			# test insertion of cnv, phase A, no cnv present
			CNVs = {}
			gain_num = 0
			loss_num = 0
			phase = cons.A
			lin_index = 1
			seg_index = 1
			cnv_index_list = [0, 1, 0]
			gain_num, loss_num = submarine.add_CN_change_to_hash(lin_list, lin_index, seg_index, 
				CNVs, gain_num, loss_num, phase, cnv_index_list)
			self.assertEqual(gain_num, 0)
			self.assertEqual(loss_num, 0)
			self.assertEqual(cnv_index_list, [0, 1, 0])
			self.assertEqual(len(CNVs.keys()), 0)

			# test insertion of cnv, phase A, cnv present at position
			CNVs = {}
			gain_num = 0
			loss_num = 0
			phase = cons.A
			lin_index = 1
			seg_index = 2
			cnv_index_list = [0, 1, 0]
			gain_num, loss_num = submarine.add_CN_change_to_hash(lin_list, lin_index, seg_index, 
				CNVs, gain_num, loss_num, phase, cnv_index_list)
			self.assertEqual(gain_num, 0)
			self.assertEqual(loss_num, 1)
			self.assertEqual(cnv_index_list, [0, 2, 0])
			self.assertEqual(CNVs[cons.LOSS][cons.A][1], cnv2)

			# test insertion of cnv, phase A, no cnv anymore
			CNVs = {}
			gain_num = 0
			loss_num = 0
			phase = cons.A
			lin_index = 1
			seg_index = 5
			cnv_index_list = [0, 2, 0]
			gain_num, loss_num = submarine.add_CN_change_to_hash(lin_list, lin_index, seg_index, 
				CNVs, gain_num, loss_num, phase, cnv_index_list)
			self.assertEqual(gain_num, 0)
			self.assertEqual(loss_num, 0)
			self.assertEqual(cnv_index_list, [0, 2, 0])
			self.assertEqual(len(CNVs.keys()), 0)

	def test_get_Z_matrix(self):
			lin0 = lineage.Lineage([1, 2, 3], 0, None, None, None, None, None, None, None, None)
			lin1 = lineage.Lineage([2, 3], 0, None, None, None, None, None, None, None, None)
			lin2 = lineage.Lineage([], 0, None, None, None, None, None, None, None, None)
			lin3 = lineage.Lineage([], 0, None, None, None, None, None, None, None, None)
			my_lineages = [lin0, lin1, lin2, lin3]

			my_z = [
				[-1, 1, 1, 1],
				[-1, -1, 1, 1],
				[-1, -1, -1, 0],
				[-1, -1, -1, -1]
				]

			z_matrix, zero_count = submarine.get_Z_matrix(my_lineages)
			self.assertEqual(my_z, z_matrix)
			self.assertEqual(1, zero_count)

	def test_sort_segments(self):
			# create segments
			end = 0
			count = 0
			hm = 0
			s1 = segment.Segment(2, 2, end, count, hm)
			s2 = segment.Segment(2, 1, end, count, hm)
			s3 = segment.Segment(1, 1, end, count, hm)
			s4 = segment.Segment(1, 2, end, count, hm)
			list = [s1, s2, s3, s4]

			# test
			list = submarine.sort_segments(list)

			self.assertListEqual(list, [s3, s4, s2, s1])

	def test_sort_snps_ssms(self):
			# create mutations
			s1 = snp_ssm.SNP()
			s2 = snp_ssm.SNP()
			s3 = snp_ssm.SNP()
			s4 = snp_ssm.SNP()
			s1.chr = 2
			s1.pos = 2
			s2.chr = 2
			s2.pos = 1
			s3.chr = 1
			s3.pos = 1
			s4.chr = 1
			s4.pos = 2
			list = [s2, s3, s1, s4]

			# test
			list = submarine.sort_snps_ssms(list)

			self.assertListEqual(list, [s3, s4, s2, s1])

	def test_snp_ssm_equality(self):
			snp1 = snp_ssm.SNP()
			snp1.chr = 1
			snp1.pos = 4
			snp2 = snp_ssm.SNP()
			snp2.chr = 1
			snp2.pos = 4
			# equality
			self.assertEqual(snp1, snp2)
			self.assertEqual(snp2, snp1)

			# difference in one attribute
			snp2.chr = 2
			self.assertNotEqual(snp1, snp2)
			self.assertNotEqual(snp2, snp1)
			snp2.chr = 1
			snp2.pos = 14
			self.assertNotEqual(snp1, snp2)
			self.assertNotEqual(snp2, snp1)

			# difference in both attributes
			snp2.chr = 12
			self.assertNotEqual(snp1, snp2)
			self.assertNotEqual(snp2, snp1)

	def test_snp_ssm_lt(self):
			snp1 = snp_ssm.SNP()
			snp1.chr = 1
			snp1.pos = 4
			snp2 = snp_ssm.SNP()
			snp2.chr = 1
			snp2.pos = 4
			# equality
			self.assertFalse(snp1 < snp2)
			self.assertFalse(snp2 < snp1)

			# difference in first attribute
			snp2.chr = 2
			self.assertTrue(snp1 < snp2)
			self.assertFalse(snp2 < snp1)

			# difference in second attribute
			snp2.chr = 1
			snp2.pos = 2
			self.assertFalse(snp1 < snp2)
			self.assertTrue(snp2 < snp1)

			# differences in both attributes
			# both attributes less than
			snp2.chr = 3
			snp2.pos = 823
			self.assertTrue(snp1 < snp2)
			self.assertFalse(snp2 < snp1)
			# one atribute less than, the other greater
			self.assertTrue(snp1 < snp2)
			self.assertFalse(snp2 < snp1)

			# chromosome is 0
			snp1.chr = 0
			snp1.pos = 10
			snp2.chr = 1
			snp2.pos = 0
			self.assertTrue(snp1 < snp2)
			self.assertFalse(snp2 < snp1)

def suite():
	 return unittest.TestLoader().loadTestsFromTestCase(ModelTest)

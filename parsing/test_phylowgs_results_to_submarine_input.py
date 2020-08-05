import unittest
import phylowgs_results_to_submarine_input as parsing
from numpy import log
from scipy.stats import binom
import numpy as np

class TestPhyloWGSParsing(unittest.TestCase):

	def test_get_all_CNAs(self):

		cna_file = "test_CNAs.txt"
		male = True

		cnas, seg_index, cna_num = parsing.get_all_CNAs(cna_file, male)
	
		self.assertEqual(cnas["c5"]["physical_cnas"][1]["phase"], "A")

	def test_assign_seg_index_to_ssms_get_impact_matrix(self):
		# one SSM not overlapping any CNA
		# one SSM in middle part of physical CNAs
		# one SSM in first part of physical CNA and also in second because of same start positions
		ssms = {"s0": {}, "s1": {"chromosome": "1", "position": 101, "a": [12, 13], "d": [26, 25], "subclone": 1, "index": 1},
			"s2": {"chromosome": "2", "position": 1, "a": [26, 24], "d": [41, 35], "subclone": 3, "index": 2}}
		
		cnas = {"c0": {"overlapping_ssms": ["s1"], "subclone": 2, "physical_cnas": 
			[{"CNA_index": 0, "seg_index": 0, "phase": "B", "change": -1, "start": 1, "end": 100, "chromosome": "1"},
			{"CNA_index": 1, "seg_index": 1, "phase": "A", "change": 1, "start": 100, "end": 200, "chromosome": "1"},
			{"CNA_index": 2, "seg_index": 2, "phase": "B", "change": -1, "start": 200, "end": 300, "chromosome": "1"}]},
			"c1": {"overlapping_ssms": ["s2"], "subclone": 4, "physical_cnas":
			[{"CNA_index": 3, "seg_index": 3, "phase": "A", "change": 2, "start": 1, "end": 100, "chromosome": "2"},
			{"CNA_index": 4, "seg_index": 3, "phase": "B", "change": 1, "start": 1, "end": 100, "chromosome": "2"}]}}
		z_matrix = [[-1, 1, 1, 1, 1], [-1, -1, 1, 1, 1], [-1, -1, -1, 1, 1], [-1, -1, -1, -1, 1], [-1, -1, -1, -1, -1]]
		normal_seg_index = 4
		cna_num = 5
		freqs = {"0": [1.0, 1.0], "1": [0.8, 0.7], "2": [0.6, 0.5], "3": [0.8, 0.7], "4": [0.6, 0.5]}
		male = True
		sequencing_error = 0.001

		true_impact_matrix = np.zeros(3*5).reshape(3, 5)
		true_impact_matrix[1][1] = 1
		true_impact_matrix[2][4] = 1

		impact_matrix = parsing.assign_seg_index_to_ssms_get_impact_matrix(ssms, cnas, z_matrix, normal_seg_index, cna_num, freqs,
			male, sequencing_error)

		self.assertEqual(ssms["s0"]["seg_index"], 4)
		self.assertEqual(ssms["s1"]["seg_index"], 1)
		self.assertEqual(ssms["s2"]["seg_index"], 3)
		self.assertTrue((impact_matrix == true_impact_matrix).all())

	def test_set_impact_entry_if_necessary(self):

		# 1) CNA is not in descendant subclone of SSM
		ssm = {"subclone": 1, "index": 0}
		cna = {"subclone": 2}
		z_matrix = [[-1, 1, 1], [-1, -1, -1], [-1, -1, -1]]
		impact_matrix = np.asarray([[0]])
		impact_matrix_true = np.asarray([[0]])
		physical_cna_index = 0
		next_CNA_same_seg_index = False
		freqs = {}
		male = True
		sequencing_error = 0.001

		parsing.set_impact_entry_if_necessary(ssm, cna, physical_cna_index, next_CNA_same_seg_index, z_matrix, freqs, impact_matrix, male, sequencing_error)
		
		self.assertTrue((impact_matrix == impact_matrix_true).all())

		# 2) two CN changes, SSM influenced by B
		ssm = {"subclone": 1, "index": 3, "a": [26, 24], "d": [41, 35], "chromosome": "10"}
		cna = {"subclone": 2, "physical_cnas": [{"CNA_index": 1, "phase": "A", "change": 2, "start": 100}, {"CNA_index": 2, "phase": "B", "change": 1, "start": 100}]}
		z_matrix = [[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]]
		impact_matrix = np.zeros(12).reshape(4, 3)
		physical_cna_index = 0
		next_CNA_same_seg_index = True
		freqs = {"0": [1.0, 1.0], "1": [0.8, 0.7], "2": [0.6, 0.5]}
		male = True
		sequencing_error = 0.001

		impact_matrix_true = np.zeros(12).reshape(4, 3)
		impact_matrix_true[3][2] = 1

		parsing.set_impact_entry_if_necessary(ssm, cna, physical_cna_index, next_CNA_same_seg_index, z_matrix, freqs, impact_matrix, male, sequencing_error)
		
		self.assertTrue((impact_matrix == impact_matrix_true).all())

		# 3) two CN changes, SSM influenced by A
		ssm = {"subclone": 1, "index": 3, "a": [15, 11], "d": [41, 35], "chromosome": "10"}
		cna = {"subclone": 2, "physical_cnas": [{"CNA_index": 1, "phase": "A", "change": 2, "start": 100}, {"CNA_index": 2, "phase": "B", "change": 1, "start": 100}]}
		z_matrix = [[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]]
		impact_matrix = np.zeros(12).reshape(4, 3)
		physical_cna_index = 0
		next_CNA_same_seg_index = True
		freqs = {"0": [1.0, 1.0], "1": [0.8, 0.7], "2": [0.6, 0.5]}
		male = True
		sequencing_error = 0.001

		impact_matrix_true = np.zeros(12).reshape(4, 3)
		impact_matrix_true[3][1] = 1

		parsing.set_impact_entry_if_necessary(ssm, cna, physical_cna_index, next_CNA_same_seg_index, z_matrix, freqs, impact_matrix, male, sequencing_error)
		
		self.assertTrue((impact_matrix == impact_matrix_true).all())

		# 4) one CN change, SSM influenced by it
		ssm = {"subclone": 1, "index": 3, "a": [12, 13], "d": [26, 25], "chromosome": "10"}
		cna = {"subclone": 2, "physical_cnas": [{"CNA_index": 1, "phase": "A", "change": 2, "start": 100}, {"CNA_index": 2, "phase": "B", "change": 1, "start": 200}]}
		z_matrix = [[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]]
		impact_matrix = np.zeros(12).reshape(4, 3)
		physical_cna_index = 1
		next_CNA_same_seg_index = False
		freqs = {"0": [1.0, 1.0], "1": [0.8, 0.7], "2": [0.6, 0.5]}
		male = True
		sequencing_error = 0.001

		impact_matrix_true = np.zeros(12).reshape(4, 3)
		impact_matrix_true[3][2] = 1

		parsing.set_impact_entry_if_necessary(ssm, cna, physical_cna_index, next_CNA_same_seg_index, z_matrix, freqs, impact_matrix, male, sequencing_error)
		
		self.assertTrue((impact_matrix == impact_matrix_true).all())

		# 5) one CN change, SSM not influenced by it
		ssm = {"subclone": 1, "index": 3, "a": [18, 18], "d": [26, 25], "chromosome": "10"}
		cna = {"subclone": 2, "physical_cnas": [{"CNA_index": 1, "phase": "A", "change": 2, "start": 100}, {"CNA_index": 2, "phase": "B", "change": 1, "start": 200}]}
		z_matrix = [[-1, 1, 1], [-1, -1, 1], [-1, -1, -1]]
		impact_matrix = np.zeros(12).reshape(4, 3)
		physical_cna_index = 1
		next_CNA_same_seg_index = False
		freqs = {"0": [1.0, 1.0], "1": [0.8, 0.7], "2": [0.6, 0.5]}
		male = True
		sequencing_error = 0.001

		impact_matrix_true = np.zeros(12).reshape(4, 3)

		parsing.set_impact_entry_if_necessary(ssm, cna, physical_cna_index, next_CNA_same_seg_index, z_matrix, freqs, impact_matrix, male, sequencing_error)
		
		self.assertTrue((impact_matrix == impact_matrix_true).all())

	def test_get_llh_for_ssm(self):
		ssm = {"chromosome": "3", "d": [10, 10], "a": [5, 2]}
		freqs = {"0": [1.0, 1.0], "1": [0.9, 1.0], "2": [0.8, 0.8], "3": [0.7, 0.6]}
		sample_size = 2
		k = 1
		kp = 3
		influence = True
		change_cna = 1
		male = False
		change_major = 1
		change_minor = 0
		sequencing_error = 0.001

		cn_variant_0 = 0.9 + 0.7
		cn_variant_1 = 1.0 + 0.6
		cn_ref_0 = 1.1
		cn_ref_1 = 1.0
		cn_total_0 = 2.7
		cn_total_1 = 2.6
		p_0 = ((cn_ref_0 * ( 1 - sequencing_error)) + (cn_variant_0 * sequencing_error)) / cn_total_0
		p_1 = ((cn_ref_1 * ( 1 - sequencing_error)) + (cn_variant_1 * sequencing_error)) / cn_total_1

		llh = parsing.compute_log_llh_binomial(ssm["d"][0], ssm["a"][0], p_0) + parsing.compute_log_llh_binomial(ssm["d"][1], ssm["a"][1], p_1)

		self.assertEqual(llh, parsing.get_llh_for_ssm(ssm, freqs, sample_size, k, kp, influence, change_cna, male, change_major, change_minor, sequencing_error))


	def test_compute_log_llh_binomial(self):
		n = 10
		k = 5
		p = 0.5
		value = log(binom.pmf(k, n, p))

		self.assertAlmostEqual(value, parsing.compute_log_llh_binomial(n, k, p))

		n = 10
		k = 3
		p = 0.5
		value = log(binom.pmf(k, n, p))

		self.assertAlmostEqual(value, parsing.compute_log_llh_binomial(n, k, p))
		

	def test_compute_cn_variant(self):
		freqs = {"0": [1.0, 1.0, 1.0], "1": [0.8, 0.7, 0.9], "2": [0.6, 0.3, 0.5],
			"3": [0.2, 0.2, 0.1]}
		sample = 1
		k = 2
		kp = 3
		influence = True
		change = -1

		self.assertAlmostEqual(parsing.compute_cn_variant(freqs, sample, k, kp, influence, change), 0.1)

		influence = False 

		self.assertEqual(parsing.compute_cn_variant(freqs, sample, k, kp, influence, change), 0.3)

	def test_compute_cn_reference(self):
		# autosom + male 
		chromosome = "1"
		male = True
		freqs = {"0": [1.0, 1.0, 1.0], "1": [0.8, 0.7, 0.9], "2": [0.6, 0.3, 0.5]}
		sample = 1
		kp = 2
		change_major = 2
		change_minor = 1
		cn_variant = 1.0

		self.assertEqual(parsing.compute_cn_reference(chromosome, male, freqs, sample, kp, change_major, change_minor, cn_variant), 1.9)

	def test_get_relationship_from_structure(self):
		structure = {'0': [1], '1': [2, 9, 11], '2': [3, 6, 8], '3': [4], '4': [5], '6': [7], '9': [10]}

		z_matrix = parsing.get_relationship_from_structure(12, structure)

		self.assertEqual(z_matrix[0], [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
		self.assertEqual(z_matrix[1], [-1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
		self.assertEqual(z_matrix[2], [-1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1])
		self.assertEqual(z_matrix[3][4], 1)
		self.assertEqual(z_matrix[3][5], 1)
		self.assertEqual(z_matrix[4][5], 1)
		self.assertEqual(z_matrix[6][7], 1)
		self.assertEqual(z_matrix[9][10], 1)

def suite():
	return unittest.TestLoader().loadTestsFromTestCase(TestPhyloWGSParsing)

if __name__ == "__main__":
	suite_test_parsing_for_phylowgs = suite()
	unittest.TextTestRunner(verbosity=2).run(suite_test_parsing_for_phylowgs)

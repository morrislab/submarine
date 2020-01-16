class Lineage(object):

	def __init__(self, sublineages, frequency, cnvs_a, cnvs_b, snps, 
		snps_a, snps_b, ssms, ssms_a, ssms_b):
		self.sublins = sublineages
		self.freq = frequency
		self.cnvs_a = cnvs_a
		self.cnvs_b = cnvs_b
		self.snps = snps
		self.snps_a = snps_a
		self.snps_b = snps_b
		self.ssms = ssms
		self.ssms_a = ssms_a
		self.ssms_b = ssms_b

	def __eq__(self, other):
		lin_equal = ((self.sublins, self.cnvs_a, self.cnvs_b, self.snps,
			self.snps_a, self.snps_b, self.ssms, self.ssms_a, self.ssms_b) ==
			(other.sublins, other.cnvs_a, other.cnvs_b, other.snps,
			other.snps_a, other.snps_b, other.ssms, other.ssms_a, other.ssms_b))
		if isinstance(self.freq, list) is False:
		    freq_equal = abs(self.freq - other.freq) < 0.00000001
		else:
                    freq_equal = True
                    for i in range(len(self.freq)):
                        if abs(self.freq[0] - other.freq[0]) >= 0.00000001:
                            freq_equal = False
                            break
		return (lin_equal and freq_equal)

	def same_lineage_expect_sublins(self, other):
		return ((self.freq, self.cnvs_a, self.cnvs_b, self.snps,
			self.snps_a, self.snps_b, self.ssms, self.ssms_a, self.ssms_b) ==
			(other.freq, other.cnvs_a, other.cnvs_b, other.snps,
			other.snps_a, other.snps_b, other.ssms, other.ssms_a, other.ssms_b))

	def create_dict(self):
		my_dict = {}
		# add sublineages
		my_dict["sublins"] = self.sublins
		# add frequency
		my_dict["freq"] = self.freq
		# add cnvs
		my_dict["cnvs_a"] = [vars(a) for a in self.cnvs_a]
		my_dict["cnvs_b"] = [vars(b) for b in self.cnvs_b]
		# add ssms
		my_dict["ssms"] = [vars(i) for i in self.ssms]
		my_dict["ssms_a"] = [vars(i) for i in self.ssms_a]
		my_dict["ssms_b"] = [vars(i) for i in self.ssms_b]
		
		return my_dict

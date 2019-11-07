class Segment(object):
	
	# TODO call 'count' -> 'depth'
	def __init__(self, chr, start, end, count, haploid_mass):
		self.chr = chr
		self.start = start
		self.end = end
		self.count = count
		self.hm = haploid_mass
		self.cn = -1
		self.index = -1

	def print_segment(self):
		print self.chr
		print self.start
		print self.end
		print self.count
		print self.hm
		print self.cn

class Segment_allele_specific(object):
	
	def __init__(self, chr, start, end, given_cn_A, standard_error_A,
		given_cn_B, standard_error_B):
		self.chr = chr
		self.start = start
		self.end = end
		self.given_cn_A = given_cn_A
		self.standard_error_A = standard_error_A
		self.given_cn_B = given_cn_B
		self.standard_error_B = standard_error_B
		self.inferred_cn_A = -1
		self.inferred_cn_B = -1
		self.index = -1

	def __eq__(self, other):
		return ((self.chr, self.start, self.end, self.given_cn_A, self.standard_error_A, self.given_cn_B,
			self.standard_error_B, self.inferred_cn_A, self.inferred_cn_B, self.index) == (other.chr,
			other.start, other.end, other.given_cn_A, other.standard_error_A, other.given_cn_B, 
			other.standard_error_B, other.inferred_cn_A, other.inferred_cn_B, other.index))

class Segment_simple(object):

	def __init__(self, chr, start, status):
		self.chr = chr
		self.start = start
		self.status  = status

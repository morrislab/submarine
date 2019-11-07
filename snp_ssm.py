class SNP_SSM(object):

	def __init__(self):
		self.chr = -1
		self.pos = -1
		self.variant_count = -1
		self.ref_count = -1
		self.seg_index = -1
		self.infl_cnv_same_lin = False
		self.phase = None
		self.lineage = None

	#def __init__(self, chr, pos):
	#	self.chr = chr
	#	self.pos = pos
	#	self.variant_count = -1
	#	self.ref_count = -1
	#	self.seg_index = -1

	def __eq__(self, other):
		return ((self.chr, self.pos) == (other.chr, other.pos))

	def __lt__(self, other):
		return ((self.chr, self.pos) < (other.chr, other.pos))
	
	def set_all_but_seg_index(self, chr, pos, variant_count, ref_count):
		self.chr = chr
		self.pos = pos
		self.variant_count = variant_count
		self.ref_count = ref_count

	def set_all(self, chr, pos, variant_count, ref_count, seg_index):
		self.set_all_but_seg_index(chr, pos, variant_count, ref_count)
		self.seg_index = seg_index

	def add_variant_count(self, variant_count):
		self.variant_count += variant_count

	def add_ref_count(self, ref_count):
		self.ref_count += ref_count

	def print_snp_ssm(self):
		print self.chr
		print self.pos
		print self.variant_count
		print self.ref_count
		print self.seg_index

class SNP(SNP_SSM):
	pass

class SSM(SNP_SSM):
	pass

class SNP_SSM_simple(object):
	
	def __init__(self, chr, pos):
		self.chr = chr
		self.pos = pos

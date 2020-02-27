class CNV(object):

	def __init__(self, change, seg_index, chr, start, end):
		self.change = change
		self.seg_index = seg_index
		self.chr = chr
		self.start = start
		self.end = end
		self.phase = -1
		self.lineage = -1
		self.index = -1

	def __eq__(self, other):
		return ((self.change, self.chr, self.start, self.end) == (other.change, other.chr, other.start, 
			other.end))

	def __lt__(self, other):
		return ((self.chr, self.start, self.end, self.change) < (other.chr, other.start, other.end, 
			other.change))

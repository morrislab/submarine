class  MyException(Exception):
	def __init__(self, value):
		self.value = value

	def __str__(self):
		return repr(self.value)

class NoParentsLeftNoise(Exception):
	def __init__(self, message, k, avFreqs_from_initial_pps):
		self.message = message
		self.k = k
		self.avFreqs_from_initial_pps = avFreqs_from_initial_pps

	def __str__(self):
		return repr(self.message)

class NoiseBufferTooLarge(Exception):
	def __init__(self, message):
		self.message = message

class ZInconsistenceInfo(Exception):

	def __init__(self, x, y, s, v_x, v_y, v_s):
		self.x = x
		self.y = y
		self.s = s
		# both values for x and y can be present (1) or absent (internally -1 but externally 0)
		# if value means "absent" chose externally used representation
		self.v_x = max(0, v_x)
		self.v_y = max(0, v_y)
		self.v_s = v_s
		self.message = "Partial tree rule conflict.\nThe following three relationships are not allowed together: Z({0}, {1}) = {2}, Z({0}, {3}) = {4}, Z({1}, {3}) = {5}.".format(
			x, y, v_x, s, v_y, v_s)

class NoParentsLeft(MyException):
	pass
	
class AddingException(MyException):
	pass

class ConcavityException(MyException):
	pass

class FileExistsException(MyException):
	pass

class FileDoesNotExistException(MyException):
	pass

class SegmentException(MyException):
	pass

class SegmentAssignmentException(MyException):
	pass

class SSMAssignmentException(MyException):
	pass

class BAFComputationException(MyException):
	pass

class UnallowedNameException(MyException):
	pass

class NoGradientException(MyException):
	pass

class NoSolutionWithLineSearchException(MyException):
	pass

class CiLineSearchEpsilonPlateau(MyException):
	pass

class ReadCountsUnavailableError(MyException):
	pass

class SSMNotFoundException(MyException):
	pass

class ZMatrixPhisInfeasibleException(MyException):
	pass
class NoRootException(MyException):
	pass
class TooMuchChangeException(MyException):
	pass
class NotProperLOH(MyException):
	pass
class ZInconsistence(MyException):
	pass
class no_CNVs(MyException):
	pass
class ADRelationNotPossible(MyException):
	pass
class ZUpdateNotPossible(MyException):
	pass
class ParameterException(MyException):
	pass
class DifferentDimensionFormat(MyException):
	pass
class NoReconstructionWithGivenLineageNumber(MyException):
	pass
class FixPhiIncompatibleException(MyException):
	pass
class LineageWith0FreqMutations(MyException):
	pass
class ZMatrixNotNone(MyException):
	pass
class SumRuleRelationshipForbidsUpdate(MyException):
        pass
class RelationshipAlreadySet(MyException):
	pass
class ReconstructionInvalid(MyException):
	pass
class SmallerNegAvFreq(MyException):
	pass

import unittest
import test_io_file
import test_submarine

if __name__ == '__main__':
	suite_io_file = test_io_file.suite()
	suite_submarine = test_submarine.suite()

	print("")

	#unittest.TextTestRunner(verbosity=2).run(suite_submarine) 
	unittest.TextTestRunner(verbosity=2).run(suite_io_file)

	print("")

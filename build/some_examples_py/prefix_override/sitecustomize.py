import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/maryam-mahmood/sdaadof_ws/install/some_examples_py'

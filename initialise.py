import argparse

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument("-nn", "--run-number", required=False, default=None, type=int)
parser.add_argument("-mm", "--model-version", required=False, default=None, type=int)
parser.add_argument("-yy", "--memory-version", required=False, default=None, type=int)
parser.add_argument("--run-folder", required=False, default='./run/', type=str)
parser.add_argument("--run-archive-folder", required=False, default='./run_archive/', type=str)

args, _ = parser.parse_known_args()

INITIAL_RUN_NUMBER = args.run_number if args.run_number is not None else args.memory_version
INITIAL_MODEL_VERSION = args.model_version
INITIAL_MEMORY_VERSION = args.memory_version

run_folder = args.run_folder
run_archive_folder = args.run_archive_folder

run_folder = 'E:/C4/run/'
run_archive_folder = 'E:/C4/run_archive/'

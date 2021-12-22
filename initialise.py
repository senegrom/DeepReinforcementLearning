import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--run-number", required=False, default=None, type=int)
parser.add_argument("-m", "--model-version", required=False, default=None, type=int)
parser.add_argument("-y", "--memory-version", required=False, default=None, type=int)

args, _ = parser.parse_known_args()

INITIAL_RUN_NUMBER = args.run_number if args.run_number is not None else args.memory_version
INITIAL_MODEL_VERSION = args.model_version
INITIAL_MEMORY_VERSION = args.memory_version

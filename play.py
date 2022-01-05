import argparse
import logging
import sys

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument("-m1", "--model1", required=False, default=-1, type=int)
parser.add_argument("-m2", "--model2", required=True, type=int)
parser.add_argument("-n", "--n_games", required=False, default=1, type=int)
parser.add_argument("-s", "--start", required=False, default=0, type=int)

args, _ = parser.parse_known_args()

from funcs import play_matches_between_versions
from game import Game


def main():
    formatter = logging.Formatter('%(message)s')
    logger = logging.getLogger("PLAY")
    logger.setLevel(1)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)

    env = Game()
    scores, _, points, sp_scores = play_matches_between_versions(env, args.model1, args.model2, args.n_games,
                                                                 logger, turns_until_tau0=0,
                                                                 goes_first=args.start)
    print('\nSCORES')
    print(scores)
    print('\nSTARTING PLAYER / NON-STARTING PLAYER SCORES')
    print(sp_scores)


if __name__ == "__main__":
    main()

import argparse

import loggers as lg
from funcs import play_matches_between_versions
from game import Game


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("-m1", "--model1", required=False, default=-1, type=int)
    parser.add_argument("-m2", "--model2", required=True, type=int)
    parser.add_argument("-n", "--n_games", required=False, default=1, type=int)
    parser.add_argument("-s", "--start", required=False, default=0, type=int)

    args, _ = parser.parse_known_args()

    env = Game()
    scores, _, points, sp_scores = play_matches_between_versions(env, args.model1, args.model2, args.n_games,
                                                                 lg.logger_tourney, args.start)
    print('\nSCORES')
    print(scores)
    print('\nSTARTING PLAYER / NON-STARTING PLAYER SCORES')
    print(sp_scores)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from argparse import ArgumentParser

import pandas as pd


def parse_arguments():
    parser = ArgumentParser(
        description="Shuffle a dataset in a .csv file and save the new version to a new file"
    )
    parser.add_argument("-i", "--input", type=str, help="Source .csv file to be shuffled")
    parser.add_argument(
        "-o", "--output", type=str, help="Destination .csv file to save the new versio to"
    )
    parser.add_argument("-s", "--seed", type=int, default=42, help="Seed for shuffling the dataset")

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    print(f"Reading csv at {args.input}")
    df = pd.read_csv(args.input)

    if args.seed is not None:
        print(f"Reshuffling data using {args.seed} as seed")
    else:
        print("Reshuffling data randomly without seed")
    df_shuffled = df.sample(frac=1.0, random_state=args.seed)

    print(f"Writing data to {args.output}")
    df_shuffled.to_csv(args.output)


if __name__ == "__main__":
    main()

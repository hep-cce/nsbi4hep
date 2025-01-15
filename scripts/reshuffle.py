import pandas as pd

from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser(description='Python script for training (deep) neural networks in a SM vs BSM classification scenario.')
    parser.add_argument('source', type=str, help='Source .csv file to be shuffled')
    parser.add_argument('destination', type=str, help='Destination .csv file to save the new versio to')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Seed for shuffling the dataset')

    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    
    print(f'Reading csv at {args.source}')
    sample = pd.read_csv(args.source)

    if args.seed is not None:
        print(f'Reshuffling data using {args.seed} as seed')
    else:
        print('Reshuffling data randomly without seed')
    reshuffled = sample.sample(frac=1.0, random_state=args.seed)

    print(f'Writing data to {args.destination}')
    reshuffled.to_csv(args.destination)
        

if __name__ == '__main__':
    main()
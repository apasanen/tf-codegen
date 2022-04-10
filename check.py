import argparse
import pandas as pd
import numpy as np


def readFile(fileName):
    x = pd.read_csv(fileName).to_numpy()[:,1:]
    return x, np.argmax(x, axis = 1) 

def main(args):
    r, rDec = readFile(args.ref)
    n = len(r)
    for filename in args.test:
        x, xDec = readFile(filename)
        print(filename, sum(rDec == xDec)/n * 100, np.max(np.abs(r-x)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate data')
    parser.add_argument('ref')
    parser.add_argument('test', nargs= '+')
    exit(main(parser.parse_args()))

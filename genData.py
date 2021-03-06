import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def splitData(df):
    return df[['x', 'y']].to_numpy().astype(np.float32), df['label'].to_numpy().astype(np.int)

def genData(N, fig):
    dims = 2
    pds = [{'loc': 0.0, 'scale' : 2.0}, {'loc': 10.0, 'scale' : 2.0}]
    data = np.empty([0, dims + 1])
    for idx, prm in enumerate(pds):
        d = np.concatenate((np.ones([N, 1], dtype=np.uint8) * idx, tfp.distributions.Normal(**prm).sample([N, dims])), axis=1)
        data = np.concatenate((data, d), axis=0)
    df = pd.DataFrame(data = data, columns = ['label', 'x', 'y'])
    df=(df-df.min())/(df.max()-df.min())
    df['label'] = df['label'].astype(np.uint8)
    config = {'x' : 'x', 'y' : 'y', 'kind' : 'scatter', 'c' : 'label', 'cmap' : "viridis"}
    if fig is not None:
        df.plot(**config).get_figure().savefig(fig)
    return df

def main(args):
    train, test = train_test_split(genData(args.N, args.fig), test_size=0.2)
    train.to_csv(args.train)
    test.to_csv(args.test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate data')
    parser.add_argument('--N', default = 10000)
    parser.add_argument('--fig')
    parser.add_argument('train')
    parser.add_argument('test')
    exit(main(parser.parse_args()))

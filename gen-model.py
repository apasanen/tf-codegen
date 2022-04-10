import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import numpy as np
import pandas as pd
from genData import splitData 
    
def main(args):
    train = pd.read_csv(args.data)
    model = tf.keras.Sequential([
        keras.layers.InputLayer(input_shape=(2,)),
        keras.layers.Dense(2, activation='relu'),
    ])
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = tf.optimizers.Adam(),
              metrics = ['accuracy'])
    history = model.fit(*splitData(train), epochs=3, verbose=True)
    pd.DataFrame(history.history).plot(figsize=(8,5)).get_figure().savefig('model-train.pdf')
    model.save(args.model)
    model.summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate data')
    parser.add_argument('data')
    parser.add_argument('model')
    exit(main(parser.parse_args()))

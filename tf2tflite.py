import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import argparse
import tensorflow as tf
from genData import splitData
import pandas as pd

class Singleton:
    dataFile = None

def main(args):
    converter = tf.lite.TFLiteConverter.from_saved_model(args.input)
    if args.quant:
        Singleton.dataFile = args.quant
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if args.ioint:
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            converter.inference_type=tf.int8
        #converter.target_spec.supported_ops = [tf.float16]
        #converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        converter.representative_dataset = representative_data_gen

    tfmodel = converter.convert()
    with open(args.output, "wb") as fout:
        fout.write(tfmodel)

def representative_data_gen():
    data = pd.read_csv(Singleton.dataFile)
    for input_value in splitData(data)[0]:
        yield [input_value]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert')
    parser.add_argument('--quant')
    parser.add_argument('--ioint', '-i', action='store_true', default=False)
    parser.add_argument('input')
    parser.add_argument('output')
    exit(main(parser.parse_args()))

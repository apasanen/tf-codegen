import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
from genData import splitData
import tflite
import sys

#logging.basicConfig(level=logging.DEBUG)

activation = {
    'relu' : lambda x: np.maximum(x, 0),
    'linear' : lambda x: x,
    'none' : None
}

TFACTIVATION = {
    tflite.ActivationFunctionType.NONE : 'linear',
    tflite.ActivationFunctionType.RELU : 'relu'
}

def myInferenceSample(model, y):
    for idx, i in enumerate(model.layers):
        config = i.get_config()
        y = np.matmul(y, i.weights[0])
        logging.debug(f'np.matmul({y}, {i.weights[0]})')
        if config['use_bias']:
            y = np.add(y, i.weights[1])
            logging.debug(f'np.add({y}, {i.weights[1]})')
        y = activation[config['activation']](y)
    return y

def myInference(model, x):
    return np.vstack([myInferenceSample(model, i) for i in x])    

def tfApply0(test, args):
    model = tf.keras.models.load_model(args.model)
    refTest = splitData(test)[0]
    return model(refTest).numpy()

def tfApply1(test, args):
    model = tf.keras.models.load_model(args.model)
    return myInference(model, splitData(test)[0])

def tfliteApply0(test, args):
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    out = []
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    for x in splitData(test)[0]:
        if input_details['dtype'] != np.float32:
            x = np.round(x / input_scale + input_zero_point)
            print('intput', 'scale', input_scale, 'zero', input_zero_point)
        
        print('x', x)
        interpreter.set_tensor(input_details['index'], [x.astype(input_details["dtype"])])
        interpreter.invoke()
        y = interpreter.get_tensor(output_details['index'])[0]
        print('y',y)

        if output_details['dtype'] != np.float32:
            y = ( y + output_zero_point ) * output_scale
            print('output', 'scale', output_scale, 'zero', output_zero_point)
        out.append(y.astype(np.float32))
    return np.vstack(out)

INTERPRET = {
    tflite.TensorType.INT8: np.int8,
    tflite.TensorType.INT32: np.int32,
    tflite.TensorType.FLOAT32 : np.float32
}

def tensorAsNumpy(model, tensor):
    nparray = model.Buffers(tensor.Buffer()).DataAsNumpy()
    if np.isscalar(nparray):
        return nparray
    else:
        return np.frombuffer(nparray.tobytes(), dtype=INTERPRET[tensor.Type()]).reshape(tensor.ShapeAsNumpy()).transpose()

class FullyConnected:
    def __init__(self, model, graph, op):
        op_opt = op.BuiltinOptions()
        self.opt = tflite.FullyConnectedOptions()
        self.opt.Init(op_opt.Bytes, op_opt.Pos)
        self.config = {'use_bias' : True, 
            'activation' : TFACTIVATION[self.opt.FusedActivationFunction()]}

        self.weights = {}
        for k in range(1, 3):
            self.weights[k-1] = tensorAsNumpy(model, graph.Tensors(op.Inputs(k)))

    def get_config(self):
        return self.config


class Quantize:
    def __init__(self, model, graph, op):
        self.config = {}
        op_opt = op.BuiltinOptions()
        print(op.OutputsLength())
        for i in range(op.InputsLength()):
            t = graph.Tensors(op.Inputs(i))
            print(dir(t))
            print(t.ShapeAsNumpy())
            print(t.Name())
            q = t.Quantization()
            print('ScaleLength', q.ScaleLength())
            self.p = tensorAsNumpy(model, graph.Tensors(op.Inputs(i)))
        for i in range(op.OutputsLength()):
            t = graph.Tensors(op.Outputs(i))
            print(dir(t))
            print(t.ShapeAsNumpy())
            print(t.Name())
            b = model.Buffers(t.Buffer())
            print(b.DataAsNumpy())
            q = t.Quantization()
            print('ScaleLength', q.ScaleLength())
            print(q.Details())
            print(q.ScaleAsNumpy())
            self.p = tensorAsNumpy(model, graph.Tensors(op.Outputs(i)))

    def get_config(self):
        return self.config

class DeQuantize:
    def __init__(self, model, graph, op):
        op_opt = op.BuiltinOptions()
        self.config = {}
        for i in range(op.InputsLength()):
            t = graph.Tensors(op.Inputs(i))
            self.p = tensorAsNumpy(model, graph.Tensors(op.Inputs(i)))

    def get_config(self):
        return self.config

FACTORY = {
    'FULLY_CONNECTED' : FullyConnected,
    'QUANTIZE' :  Quantize,
    'DEQUANTIZE' :  DeQuantize
}

class myModel():
    def __init__(self, filename):
        with open(filename, 'rb') as f:
            buf = f.read()
            self.model = tflite.Model.GetRootAsModel(buf, 0)
        self.layers = []
        for s in range(self.model.SubgraphsLength()):
            graph = self.model.Subgraphs(s)
            for j in range(graph.OperatorsLength()):
                op = graph.Operators(j)
                opcode = self.model.OperatorCodes(op.OpcodeIndex())
                self.layers.append(FACTORY[tflite.opcode2name(opcode.BuiltinCode())](self.model, graph, op))

def tfliteApply1(test, args):
    model = myModel(args.model)
    return myInference(model, splitData(test)[0])

config = {
    '' : {
        False: tfApply0, 
        True: tfApply1
    },
    '.tflite' : {
        False: tfliteApply0,
        True: tfliteApply1,
    }
}

def main(args):
    test = pd.read_csv(args.data)
    out = config[os.path.splitext(args.model)[1]][args.customInference](test, args)
    pd.DataFrame(out).to_csv(args.output)

if __name__ == "__main__":
    #sys.argv = ['evaluate.py', '-c', '3', 'test.csv', 'model-quant.tflite', 'out-3q.txt']
    parser = argparse.ArgumentParser(description='Generate data')
    parser.add_argument('--customInference', '-c', action='store_true', default=False)
    parser.add_argument('data')
    parser.add_argument('model')
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    exit(main(parser.parse_args()))

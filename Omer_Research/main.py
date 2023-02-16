from network import Network
import numpy as np
import json
import random
import os

def function(x):
    out = 0.15085804227*x + 0.5
    if out < 0: out = 0
    if out > 1: out = 1
    return out

def func2(x):
    return 1/(1 + np.exp(-x))

func = function

nn = Network([1,3,3,1])
inputs = [[x] for x in range(-100,100)]

#nn.load_params("test.json")

def train():
    for i in range(20000):
        os.system("cls")
        print("Epoch : %s" % i)
        random.shuffle(inputs)
        outputs = [[func(inp[0])] for inp in inputs]
        nn.backward(inputs,outputs)
        
    nn.save_params("test.json")
        
train()

test = [[2],[0],[-3]]
print(nn.forward(test))

outputs = []
for inp in test:
    outputs.append([func(inp[0])])
print(outputs)


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from two_layer_model import Model


def generator(batch):
    data = pd.read_csv('data/iris.csv')

    data.drop(columns=['Id'], inplace=True)
    data = data.sample(frac=1).reset_index(drop=True)
    print(data.head())
    data['setosa'] = data['Species'].apply(lambda x: 1 if x == 'Iris-setosa' else 0)
    data['versicolor'] = data['Species'].apply(lambda x: 1 if x == 'Iris-versicolor' else 0)
    data['virginica'] = data['Species'].apply(lambda x: 1 if x == 'Iris-virginica' else 0)

    input = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    truth = data[['setosa', "versicolor", "virginica"]]
    idx = 0
    while idx < len(input):
        end = idx + batch
        x = input.iloc[idx:end, :].to_numpy()
        y = truth.iloc[idx:end, :].to_numpy()
        yield x, y
        idx += batch


model = Model(4, 15, 3)
batch = 30
learning_rate = 0.01
grad_clip = 1
loss_result = []
epoch = 50
for _ in range(epoch):
    for x, y in generator(batch=batch):
        grad_1, grad_w2, loss = model.forward(x, y)
        model.update_weights(grad_1, grad_w2, learning_rate, grad_clip)
        loss_result.append(loss)
        # print(f"x is {x}, {x.shape}")
        # print(f"y is {y}, {y.shape}")
        # print(f"gradient 1 is {grad_1} of shape {grad_1.shape}")
        # print(f"gradient 2 is {grad_w2} of shape {grad_w2.shape}")
        print(f"loss is {loss}")
    # input("Press Enter to continue...")

plt.plot(np.arange(len(loss_result)), loss_result)
plt.show()

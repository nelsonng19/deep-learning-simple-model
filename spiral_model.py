import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from two_layer_model import Model


def generator(batch):
    data = pd.read_csv('data/spiral.csv')

    data = data.sample(frac=1).reset_index(drop=True)

    input = data[["x", "y"]]
    truth = data[['a', "b"]]
    idx = 0
    while idx < len(input):
        end = idx + batch
        x = input.iloc[idx:end, :].to_numpy()
        y = truth.iloc[idx:end, :].to_numpy()
        yield x, y
        idx += batch


def generate_heat_map(spacing, model):
    x = np.linspace(0, 1, spacing)

    # numpy.linspace creates an array of
    # 9 linearly placed elements between
    # -4 and 4, both inclusive
    y = np.linspace(0, 1, spacing)

    # The meshgrid function returns
    # two 2-dimensional arrays
    x_1, y_1 = np.meshgrid(x, y)
    input_pairs = np.stack((x_1, y_1), axis=-1)
    input_list = input_pairs.reshape(-1, 2)
    print(input_list.shape)
    model_pred = []
    for x_a, y_a in input_list:
        # Feed [x_a, y_a] to your model here
        model_pred.append(model.inference(np.array([x_a, y_a]).reshape(1, 2))[0][0])

    print(model_pred)
    model_pred = np.array(model_pred).reshape(spacing, spacing)
    plt.pcolormesh(x_1, y_1, model_pred[:-1, :-1], shading='flat')
    plt.colorbar()  # Add a color bar for reference
    plt.title("Heatmap using pcolormesh")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.show()


model = Model(2, 40, 2)
batch = 10
learning_rate = 0.000001
grad_clip = 1
loss_result = []
epoch = 500


for _ in range(epoch):
    for x, y in generator(batch=batch):
        grad_1, grad_w2, loss = model.forward(x, y)
        model.update_weights(grad_1, grad_w2, learning_rate, grad_clip)
        loss_result.append(loss)
        generate_heat_map(30, model)
        print(f"x is {x}, {x.shape}")
        print(f"y is {y}, {y.shape}")
        print(f"gradient 1 is {grad_1} of shape {grad_1.shape}")
        print(f"gradient 2 is {grad_w2} of shape {grad_w2.shape}")
        print(f"loss is {loss}")
    # input("Press Enter to continue...")

plt.plot(np.arange(len(loss_result)), loss_result)
plt.show()

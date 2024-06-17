import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import linalg


class model():
    def __init__(self, Ni, Nd, Nc):
        self.w1 = np.random.rand(Ni, Nd)
        self.w2 = np.random.rand(Nd, Nc)
        self.Ni = Ni
        self.Nd = Nd
        self.Nc = Nc

    def layer_1(self, input):
        out = input @ self.w1
        # Relu
        out[out <= 0] = 0.0
        return out

    def layer_2(self, input):
        out = input @ self.w2
        # Softmax
        e = np.exp(out)
        e_sum = np.sum(e, 1)
        out = e / e_sum[:, None]
        return out

    def forward(self, x, y):
        # x:(b, Ni), y:(b, Nc)
        # h:(b, Nd)
        h = self.layer_1(x)
        # y_hat:(b,Nc)
        y_hat = self.layer_2(h)

        # gradient at w1
        batch = np.shape(x)[0]
        # YY:(b, Nc)
        YY = (-y * (np.ones((batch, self.Nc)) - y_hat)) / batch
        #dl_dh:(b, Nd)
        dl_dh = YY @ np.transpose(self.w2)
        # indicator = np.zeros((1, self.Nd))
        #
        # indicator[h > 0] = 1
        #
        # dl_dh = dl_dh * indicator
        dl_dh[h <= 0] = 0.0
        #grad_1:(Ni, Nd)
        grad_1 = np.transpose(x) @ dl_dh

        # gradient at w2
        # x_w1:(Nd, b)
        x_w1 = np.transpose(h)
        #grad_w2:(Nd, Nc)
        grad_w2 = x_w1 @ YY

        loss = np.sum(-y * np.log(y_hat), 1)
        loss = np.mean(loss)
        return grad_1, grad_w2, loss

    def update_weights(self, grad_1, grad_w2, learning_rate, grad_clip=None):
        mag_1 = linalg.norm(grad_1)
        mag_2 = linalg.norm(grad_w2)

        if grad_clip and mag_1 > grad_clip:
            grad_1 = grad_1 / mag_1 * grad_clip

        self.w1 -= learning_rate * grad_1

        if grad_clip and mag_2 > grad_clip:
            grad_w2 = grad_w2 / mag_2 * grad_clip

        self.w2 -= learning_rate * grad_w2

        if np.isnan(self.w1).any() or np.isnan(self.w2).any():
            input("stop")



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


model = model(4, 15, 3)
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
        print(f"x is {x}, {x.shape}")
        print(f"y is {y}, {y.shape}")
        print(f"gradient 1 is {grad_1} of shape {grad_1.shape}")
        print(f"gradient 2 is {grad_w2} of shape {grad_w2.shape}")
        print(f"loss is {loss}")
    # input("Press Enter to continue...")

plt.plot(np.arange(len(loss_result)),loss_result)
plt.show()
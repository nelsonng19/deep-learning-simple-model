import numpy as np
from numpy import linalg


class Model:
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

    def inference(self, x):
        # x:(b, Ni), y:(b, Nc)
        # h:(b, Nd)
        h = self.layer_1(x)
        # y_hat:(b,Nc)
        return self.layer_2(h)

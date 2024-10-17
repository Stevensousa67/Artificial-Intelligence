# This program will calculate the dot product when given two vectors and a bias

from math import exp

class Neuron:
    def __init__(self, x, w, b):
        self.x = x
        self.w = w
        self.b = b

    def dot_product(self):
        for i in range(len(self.x)):
            self.x[i] *= self.w[i]
        return sum(self.x) + self.b

    def sigmoid(self):
        return 1 / (1 + exp(-self.dot_product()))
    
    def feed_forward(self):
        return self.sigmoid()
    
    def MSE(y, y_hat):
        return (y - y_hat) ** 2
    
if __name__ == '__main__':
    x = [2, 3]  # Vector (x1 = 2, x2 = 3)
    w = [0, 1]  # Weights (w1 = 0, w2 = 1)
    b = 4       # Bias

    neuron = Neuron(x, w, b)
    print(neuron.feed_forward())
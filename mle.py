import math
import statistics

import numpy as np
import matplotlib.pyplot as plt


class NewtonRaphson:
    def __init__(self, a, b):
        self.alphaInit = a
        self.betaInit = b

    def getHessian(self, a, b, x):
        secAlpha = 0
        n = len(x)
        for i in range(n):
            secAlpha = secAlpha + (math.exp(-(x[i] - a) / b))

        secAlpha = -(1 / b ** 2) * secAlpha

        firstTermAB = 0
        secondTermAB = 0
        for i in range(n):
            firstTermAB = firstTermAB + (math.exp(-(x[i] - a) / b))
            secondTermAB = secondTermAB + ((x[i] - a) * (math.exp(-(x[i] - a) / b)))

        secAlphaBeta = -(n / b ** 2) + ((1 / b ** 2) * firstTermAB) - ((1 / b ** 3) * secondTermAB)

        firstTermB = 0
        secondTermB = 0
        thirdTermB = 0
        for i in range(n):
            firstTermB = firstTermB + (x[i] - a)
            secondTermB = secondTermB + ((x[i] - a) * (math.exp(-(x[i] - a) / b)))
            thirdTermB = thirdTermB + (((x[i] - a) ** 2) * (math.exp(-(x[i] - a) / b)))

        secBeta = (n / b ** 2) - ((2 / b ** 3) * firstTermB) + ((2 / b ** 3) * secondTermB) - (
                (1 / b ** 4) * thirdTermB)

        hessian = np.array([[secAlpha, secAlphaBeta], [secAlphaBeta, secBeta]])
        return hessian

    def gradient(self, a, b, x):
        n = len(x)
        firstAlpha = 0
        for i in range(n):
            firstAlpha = firstAlpha + (math.exp(-(x[i] - a) / b))

        firstAlpha = (n / b) - ((1 / b) * firstAlpha)

        firstTermB = 0
        secondTermB = 0
        for i in range(n):
            firstTermB = firstTermB + (x[i] - a)
            secondTermB = secondTermB + ((x[i] - a) * (math.exp(-(x[i] - a) / b)))

        firstBeta = -(n / b) + ((1 / b ** 2) * firstTermB) - ((1 / b ** 2) * secondTermB)
        gradient = np.array([[firstAlpha], [firstBeta]])
        return gradient


n = [100, 1000, 10000]
alpha, beta = 400, 150
param = {
    100: [],
    1000: [],
    10000: []
}
for i in n:
    for j in range(1, 20):
        s = np.random.gumbel(alpha, beta, i)

        alphaOld = 300
        betaOld = 100
        dif = 9999
        MaxItr = 100
        itr = 0
        NR = NewtonRaphson(alphaOld, betaOld)
        while dif > 0.001 or itr > MaxItr :
            h = NR.getHessian(alphaOld, betaOld, s)
            g = NR.gradient(alphaOld, betaOld, s)
            updatedParam = np.array([[alphaOld], [betaOld]]) - np.matmul(np.linalg.inv(h), g)
            # print(updatedParam)
            dif = updatedParam - np.array([[alphaOld], [betaOld]])
            dif = np.matmul(dif.transpose(), dif)
            alphaOld = updatedParam[0][0]
            betaOld = updatedParam[1][0]
            itr += 1
        param[i].append((alphaOld, betaOld))

alphaMeanSD = {
    100: [],
    1000: [],
    10000: []
}
betaMeanSD = {
    100: [],
    1000: [],
    10000: []
}

for i in n:
    alphaMeanSD[i] = (statistics.mean(al[0] for al in param[i]), statistics.stdev(al[0] for al in param[i]))
    betaMeanSD[i] = (statistics.mean(be[1] for be in param[i]), statistics.stdev(be[1] for be in param[i]))

print("Original parameters:")
print("alpha: " + str(alpha) + " beta: " + str(beta))
print(" ")

for i in n:
    print("For n=" + str(i) + ":")
    print("Alpha:")
    print("Mean: " + str(round(alphaMeanSD[i][0], 2)) + " Standard Deviation:" + str(round(alphaMeanSD[i][1], 2)))
    print("Beta:")
    print("Mean: " + str(round(betaMeanSD[i][0], 2)) + " Standard Deviation:" + str(round(betaMeanSD[i][1], 2)))
    print(" ")

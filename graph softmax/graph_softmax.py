from torch import enable_grad
from torch.nn import functional as F
import torch
import numpy
import pickle
import heapq
import math

class graph_softmax:
    @staticmethod
    @enable_grad()

    def formula(z_row = None, lastword_index = None, fdist_new=None, dic=None):
        m = F.softmax(z_row, dim=-1).numpy()
        z_test = z_row.tolist()[-1]
        z_new, pos = [], []

        for i in range(len(z_test)):
            if z_test[i] != float("-inf"):
                z_new.append(z_test[i])
                pos.append(i)

        x = F.softmax(torch.tensor(z_new), dim=-1)
        x = torch.tensor(x, requires_grad=True, dtype=torch.float64)
        m1 = torch.argmax(x)
        z = torch.tensor(z_new, dtype=torch.float64)
        A = torch.tensor([0] * len(z_new), dtype=torch.float64)

        while True:
            if dic[lastword_index].strip() in fdist_new:
                qh = fdist_new[dic[lastword_index].strip()]['qh']
            if dic[lastword_index].strip() in fdist_new:
                for word, v in fdist_new[dic[lastword_index].strip()].items():
                    if word != 'qh' and dic.index(word) in pos:
                        s1 += 1
                        A[pos.index(dic.index(word))] = v / qh
                    if ' ' + word in dic and dic.index(' ' + word) in pos:
                        s2 += 1
                        A[pos.index(dic.index(' ' + word))] = v / qh
            ss = s1 + s2
            break

        learing_rate = 3e-03
        L, nn, fn = 2, -1, 0

        while True:
            f = -sum(x * z) + sum(x * torch.log(x)) + sum((x - A) ** 2) * L
            f.backward()
            nn += 1
            if nn == 0:
                first = abs(f.data - fn)

            if abs(f.data - fn) < 1e-05:
                break
            if nn == 500:
                break

            fn = f.data
            a = x.data - learing_rate * x.grad.data
            for i in range(len(a)):
                a[i] = 0 if math.isnan(a[i]) or a[i] == float("-inf") or a[i] == float("inf") else a[i]
            b = sorted(a.numpy())
            k, sigma = 0, 0
            for j, bj in enumerate(b):
                sigma += bj
                s = bj + (1 - sigma) / (j + 1)
                if s > 0:
                    k = j
            Lambda = (1 - sum(b[:(k + 1)])) / (k + 1)
            Lambda = 0 if math.isnan(Lambda) else Lambda
            a_np = a.numpy()
            x = []
            for i in range(len(a_np)):
                x.append(max(a_np[i] + Lambda, 0))
            index = 0
            x2, z2, A2, pos2 = [], [], [], []
            for i in range(len(x)):
                if x[i] != 0:
                    x2.append(x[i])
                    z2.append(z[i])
                    A2.append(A[i])
                    pos2.append(pos[i])
            z = torch.tensor(z2, dtype=torch.float64)
            A = torch.tensor(A2, dtype=torch.float64)
            pos = pos2
            x = torch.tensor(x2, requires_grad=True, dtype=torch.float64)
        x2 = [0] * 50257
        for i in range(len(pos)):
            x2[pos[i]] = x[i]
        x = torch.tensor(x2, requires_grad=True, dtype=torch.float64)

        return x

from matplotlib.pyplot import get
import numpy as np
from scipy import linalg

def get_matrix(negp_price, negq_givenp_price, q_price, pandq_price):
    a1 = negp_price
    a2 = negq_givenp_price
    a3 = q_price
    a4 = pandq_price

    m = np.array([
        [-a1, -a2, -(a3 - 1), -(a4 - 1)],
        [-a1, -(a2 - 1), -(a3 - 1), -a4],
        [-(a1 - 1), 0, -(a3 - 1), -a4],
        [-(a1 - 1), 0, -a3, -a4]
        ])

    return m

def get_sol(matrix):
    r = linalg.null_space(matrix)
    return r


if __name__ == '__main__':
    m = get_matrix(0.48, 0.48, 0.52, 0.27)
    r = get_sol(m)
    print(r)
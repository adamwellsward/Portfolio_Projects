# p_euler_20.py
"""
Project Euler Problem 20 Solution
Adam Ward
"""

import numpy as np
from math import factorial

def factorial_sum():
    num = factorial(100)
    digits = np.array(list(str(num))).astype(int)
    return np.sum(digits)

if __name__ == "__main__":
    print(factorial_sum())
    pass
# p_euler_16.py
"""
Project Euler Problem 16 Solution
Adam Ward
"""

import numpy as np

def power_sum():
    num = 2**1000
    digits = np.array(list(str(num))).astype(int)
    return np.sum(digits)

if __name__ == "__main__":
    print(power_sum())
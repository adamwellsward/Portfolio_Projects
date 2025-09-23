# p_euler_6.py
"""
Project Euler Problem 6 Solution
Adam Ward
"""

import numpy as np

def sum_square_diff(N):
    nums = np.arange(1,N+1)
    return (np.sum(nums))**2 - np.sum(nums**2)

if __name__ == "__main__":
    print(sum_square_diff(100))
    pass
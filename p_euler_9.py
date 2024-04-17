# p_euler_9.py
"""
Project Euler Problem 9 Solution
Adam Ward
"""

import numpy as np

def pyth_triplet():
    for a in range(1000):
        for b in range(a,1000):
            c = 1000 - b - a
            if c <= b:
                break
            if a**2 + b**2 == c**2:
                return a*b*c
    return -1

if __name__ == "__main__":
    # print(pyth_triplet())
    pass
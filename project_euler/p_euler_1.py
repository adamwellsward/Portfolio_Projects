# p_euler_1.py
"""
Project Euler Problem 1 Solution
Adam Ward
"""

import numpy as np

def mult_3_5(n):
    full = np.arange(n)[1:]         # construct full array of numbers up to n, excluding 0
    three = full[full % 3 == 0]     # use mask to obtain multiples of 3        
    five  = full[full % 5 == 0]     # use mask to obtain multiples of 5
    five  = five[five % 3 != 0]     # remove multiples of 3 from multiples of 5
    return sum(three) + sum(five)   # return the sum of both arrays

if __name__ == "__main__":
    print(mult_3_5(1000))
    pass
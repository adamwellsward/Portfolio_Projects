# p_euler_18.py
"""
Project Euler Problem 18 Solution
Adam Ward
"""

import numpy as np

def small_tri_sum(file="small_tri.txt"):
    # read in the triangle
    with open(file) as f:
        tri = [[int(num) for num in line.strip().split(' ')] for line in f.readlines()]
    
    # iterate from the bottom up, replacing each row with the greatest sum of itself and the two numbers beneath it
    num_rows = len(tri)
    for i in reversed(range(num_rows-1)):
        for j in range(len(tri[i])):
            tri[i][j] += max(tri[i+1][j], tri[i+1][j+1])

    return tri[0][0]

if __name__ == "__main__":
    print(small_tri_sum())
    pass
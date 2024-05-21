# p_euler_15.py
"""
Project Euler Problem 15 Solution
Adam Ward
"""

from math import comb

def num_paths(n):
    """Find the number of paths from corner to corner in an n x n grid. """
    paths = comb(2*n,n)
    return paths

if __name__ == "__main__":
    print(num_paths(20))
    pass
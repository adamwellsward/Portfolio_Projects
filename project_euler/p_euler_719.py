# p_euler_719.py
"""
Project Euler Problem 719 Solution
Adam Ward
"""

import numpy as np
from itertools import combinations as comb
# idea: use combinations to get all the combinations of the digits to take the sum of them

if __name__ == "__main__":
    print(list(comb(str(8281), 3)))
    pass
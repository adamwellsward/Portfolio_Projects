# p_euler_12.py
"""
Project Euler Problem 12 Solution
Adam Ward
"""

import numpy as np

def tri_nums():
    # Generator for triangle numbers
    current = 1
    add = 1
    yield current
    while True:
        add += 1
        current += add 
        yield current

def num_divisors(N):
    """
    Returns the first triangle number that has more than N factors
    """
    # iterate through the generator of triangle numbers
    for num in tri_nums():
        # initialize the factors list
        factors = [1]

        # search for factors up to the square root of the number
        for i in range(2, int(num**.5) + 1):
            if num % i == 0:
                factors.append(i)

        # get the rest of the factors by dividing the number by each of its factors,
        # then union the sets of factors to get rid of any duplicates
        factors = set(factors).union(set([num//j for j in factors]))

        # check how many factors there are, and stop once we are over N
        if len(factors) > N:
            break

    return num

if __name__ == "__main__":
    # print(list(tri_nums()))
    print(num_divisors(500))
    pass
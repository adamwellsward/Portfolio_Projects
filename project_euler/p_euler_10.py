# p_euler_10.py
"""
Project Euler Problem 10 Solution
Adam Ward
"""

import numpy as np

def primes_up_to(N):
    """Compute the primes up to N."""
    primes_list = [2]               # start the list with 2
    current = 3                     # start the search at three
    while primes_list[-1] < N:     
        isprime = True
        sqrt_n = int(np.sqrt(current)) + 1   # compute this outside to save time, make it an int to speed comparison
        for p in primes_list:       # Check the list of primes instead of all numbers up to current
            if p > sqrt_n:          # break if the number gets bigger than the square root of the current
                break
            if current % p == 0:    # check if it's a divisor
                isprime = False
                break
        if isprime:
            primes_list.append(current)
        current += 2                # only check odd numbers

    return primes_list[:-1]

if __name__ == "__main__":
    primes = primes_up_to(2000000)
    print(np.sum(primes))
    pass
# p_euler_7.py
"""
Project Euler Problem 7 Solution
Adam Ward
"""
import numpy as np

# This function was implemented as part of a coding lab for Math 347 at BYU, Winter 2024

def primes_fast(N):
    """Compute the first N primes."""
    primes_list = [2]               # start the list with 2
    current = 3                     # start the search at three
    while len(primes_list) < N:     
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
    return primes_list

if __name__ == "__main__":
    print(primes_fast(10001)[-1])
    pass
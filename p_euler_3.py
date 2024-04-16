# p_euler_3.py
"""
Project Euler Problem 3 Solution
Adam Ward
"""
import numpy as np

# use this function from the Volume 1 Profiling Lab to quickly compute the list of primes
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

def largest_prime_factor(n):
    """ My second attempt that worked. 
    This approach isn't the most versatile and scalable, but it works for this problem
    """
    factors = []
    all_primes = primes_fast(100001)    # compute the list of primes
    for p in all_primes:
        if n % p == 0:                  # get the factors of n
            factors.append(p)
    return factors, factors[-1]         # return the largest one


if __name__ == "__main__":
    # print(largest_prime_factor(600851475143))
    pass

""" My failed first attempt due to trying to allocate too much memory """
    # full = np.arange(2,n)
    # factors = []
    # while len(full) > 0:
    #     first = full[0]
    #     if n % first == 0:
    #         factors.append(first)
    #     full = full[full % full[0] != 0]
    # print(factors)
    # return factors[-1]

""" My failed third attempt to find factors the hard way """
    # factors = [n]
    # for i in reversed(range(int(n/2))):
    #     if n % i == 0:
    #         factors.append(i)
    #         print(factors)

    # return factors[-1]
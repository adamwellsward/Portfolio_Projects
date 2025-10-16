# important_functions.py
"""
Functions I have written that come up frequently in Project Euler problems.

Adam Ward
"""
import numpy as np
from itertools import permutations

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

def is_prime(x):
    """Checks if any number is prime."""
    if (x % 2 == 0 and x != 2) or x <= 1:
        return False        # even numbers (except 2), 1, 0, and negative numbers are not prime
    elif x == 2:
        return True
    else:
        sqrt_x = int(np.sqrt(x)) + 1
        for i in range(3, sqrt_x, 2):
            if x % i == 0:
                return False
        return True
    
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
    This approach has been scaled to work for any number
    """
    factors = []
    all_primes = primes_fast(int(np.sqrt(n)+1))    # compute the list of primes
    for p in all_primes:
        if n % p == 0:                  # get the factors of n
            factors.append(p)
    return factors, factors[-1] if len(factors) > 0 else None        # return the largest one

def n_pandigitals(n):
    """Return an array of the n-digit pandigital numbers."""
    numbers = "".join(str(d) for d in range(1, n+1))
    return np.array([int("".join(num)) for num in permutations(numbers)])

def n_pandigit_w_zero(n):
    """Return an array of the n-digit pandigital numbers (INCLUDING 0)."""
    numbers = "".join(str(d) for d in range(n+1))
    perms = np.array([int("".join(num)) for num in permutations(numbers)])
    return perms[np.array([True if len(str(num)) == n+1 else False for num in perms])]

def import_data(filename):
    """
    Given a filename for a .txt file downloaded from Project Euler, 
    read in the file and return an array with all of the elements of the file.
    """
    with open(filename) as f:
        full_string = f.readline()
    data = np.array(full_string.replace('"', '').rsplit(sep=','), dtype=str)
    return data

def is_triangular(x):
    """If x is triangular, return True. Else return False."""
    if (-1 + (1 + 8*x)**.5) % 2 == 0:
        return True
    else:
        return False
    
def is_pentagonal(x):
    """If x is pentagonal, return True. Else return False."""
    if (1 + (1 + 24*x)**.5) % 6 == 0:
        return True
    else:
        return False
    
def is_hexagonal(x):
    """If x is hexagonal, return True. Else return False."""
    if (1 + (1 + 8*x)**.5) % 4 == 0:
        return True
    else:
        return False
    
    

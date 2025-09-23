# p_euler_23.py
"""
Project Euler Problem 23 Solution
Adam Ward
"""

import numpy as np
from itertools import combinations

def get_proper_divisors(x):
    '''Return a NumPy array with all of the factors.'''
    # 0 has no divisors so don't do it
    if x == 0: return np.array([])    
    
    # intialize divisors with 1 and the number itself
    divisors = [1, x]

    # obtain the first divisors by checking numbers up to the square root
    for i in range(2, int(np.sqrt(x)) + 1):
        if x % i == 0:
            divisors.append(i)

    # obtain the rest of the divisors by dividing the number by the lowest divisors
    new_divisors = [x//j for j in divisors]

    # remove duplicates and return as an array
    return np.array(sorted(list(set(divisors).union(set(new_divisors)))))[:-1]

def is_abundant(y): return True if np.sum(get_proper_divisors(y)) > y else False

def get_abundant_numbers(n):
    '''obtain an array of all abundant numbers up to n, inclusive.'''
    abund_nums = [12]
    for i in range(13, n+1):
        if is_abundant(i):
            abund_nums.append(i)
    return np.array(abund_nums)

def get_final_sum(n):
    abund_nums = get_abundant_numbers(n)

    # start the sum with the sum of all numbers up to 24, the smallest that can be written as 
    # the sum of two abundant numbers
    total_sum = np.sum(np.arange(24))

    # now loop through the rest of the numbers
    for i in range(24, n+1):
        abundant = False
        for a_num in abund_nums[abund_nums < (i - 10)]:
            if (i - a_num) in abund_nums:
                abundant = True
                break

        if not abundant:
            total_sum += i

    return total_sum

if __name__ == "__main__":
    print(get_final_sum(28123))
    pass
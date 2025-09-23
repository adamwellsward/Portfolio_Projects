# p_euler_25.py
"""
Project Euler Problem 25 Solution
Adam Ward
"""

import numpy as numpy

# This solution completed as a coding lab problem in 3/24
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    f_1 = 1
    yield f_1       # yield the first two digits independently
    f_2 = 1
    yield f_2
    while True:     # now iteratively compute and yield the next term in the Fibonacci sequence
        f_n = f_1 + f_2
        f_1 = f_2
        f_2 = f_n
        yield f_n

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    for i,x in enumerate(fibonacci()):  # iterate through the infinite generator 
        if len(str(x)) == N:            # return the index when the number of digits equals N
            return i+1

if __name__ == "__main__":
    print(fibonacci_digits())
    pass
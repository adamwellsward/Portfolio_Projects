# p_euler_2.py
"""
Project Euler Problem 2 Solution
Adam Ward
"""
import numpy as np

def fibonacci(N=4e6):
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    f_1 = 1
    yield f_1       # yield the first two digits independently
    f_2 = 1
    yield f_2
    f_n = f_1 + f_2
    while f_n < N:      # now iteratively compute and yield the next term in the Fibonacci sequence
        yield f_n       # until the sequence surpasses the cap of N (defaulting to 4 million)
        f_1 = f_2
        f_2 = f_n
        f_n = f_1 + f_2

def even_fib_sum(N=4e6):
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    generated = fibonacci(N)                    # generate the fibonacci sequence
    values = np.array(list(generated))[2::3]    # slice out the even numbers
    return sum(values)                          # return the sum

if __name__ == "__main__":
    print(even_fib_sum())
    pass
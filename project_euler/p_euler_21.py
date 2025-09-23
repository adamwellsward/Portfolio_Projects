# p_euler_21.py
"""
Project Euler Problem 21 Solution
Adam Ward
"""
import numpy as np

def d(n):
    # obtain proper divisors and add them
    sum = 0
    for i in range(1, n//2 + 1):
        if n % i == 0:
            sum += i
    return sum

def get_full_sum(N=10_000):
    """Evaluate the sum of all the amicable numbers under N."""
    # iterate through numbers to find amicable numbers, not checking those that have already been checked...
    ami_nums = []
    checked = set()
    for i in range(1, N):
        if i in checked:
            continue
        else:
            out = d(i)
            new = d(out)
            if ((out != i) and (new == i)):
                ami_nums.append(i)
                ami_nums.append(out)
                checked.add(i)
                checked.add(out)

    return np.sum(ami_nums)

if __name__ == "__main__":
    print(get_full_sum())
    pass
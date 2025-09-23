# p_euler_5.py
"""
Project Euler Problem 5 Solution
Adam Ward
"""

import numpy as np

def smallest_mult():
    num = 2522                  # we can start at the smallest multiple up to 10
    found = False
    while not found:
        for i in range(1,21):
            if num % i != 0:    # run through the 20 numbers, breaking if one is not a factor
                num += 2        # we only need to consider even numbers
                break
            if i == 20:         # if 20 was a divisor, we found the number!
                found = True
    return num

if __name__ == "__main__":
    print(smallest_mult())
    pass
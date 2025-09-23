# p_euler_4.py
"""
Project Euler Problem 4 Solution
Adam Ward
"""
import numpy as np

def palindrone_prod():
    largest_palin = 0                   # initialize
    for i in range(100,1000):           # run through all possible products of 3 digit numbers
        for j in range(100,1000):
            num = i * j
            prod = str(num)
            if prod == prod[::-1]:      # check if it is a palindrome
                if num > largest_palin: # if it is larger than the current largest, set it to be the largest
                    largest_palin = num
    return largest_palin

if __name__ == "__main__":
    print(palindrone_prod())
    pass
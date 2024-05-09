# p_euler_14.py
"""
Project Euler Problem 14 Solution
Adam Ward
"""

import numpy as np

# Collatz sequence generator given a starting value
def collatz(N):
    if N == 0:
        raise ValueError("Zero not defined for Collatz sequence")
    sequence = [N]
    curr = N
    while curr != 1:                # append new terms in the sequence until we reach 1 using the definition
        if curr % 2 == 0:
            curr = curr // 2
            sequence.append(curr)
        else:
            curr = 3 * curr + 1
            sequence.append(curr)
    return sequence

def longest_chain(N=int(1e6)):
    longest_length = 0
    number = 0
    for num in range(1, N+1):
        length = len(collatz(num))  # get the length of the collatz sequence
        if length > longest_length: # if its the new longest, save it
            longest_length = length
            number = num
            print("New longest length:\t", longest_length, "\nNumber:\t\t\t", number)
    return number

if __name__ == "__main__":
    print(longest_chain())    
    pass
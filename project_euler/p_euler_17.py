# p_euler_17.py
"""
Project Euler Problem 17 Solution
Adam Ward
"""

def letters_used(N):
    """ Returns the number of letters used to write out every number up 
        to and including N. """
    # dictionary of written out numbers
    words = {'1':"one", '2':"two", '3':"three",
             '4':"four", '5':"five", '6':"six",
             '7':"seven", '8':"eight", '9':"nine", '0':""}

    # function to calculate length of 1-digit number
    def single(dig):
        return len(words[dig])
    
    # function to calculate length of a general 2-digit number
    def double(two_dig):
        first, last = two_dig[0], two_dig[1]
        if first == '0':
            return len(words[last])
        elif first == '1':
            if last == '0':
                return len("ten")
            elif last == '1':
                return len("eleven")
            elif last == '2':
                return len("twelve")
            elif last == '3':
                return len("thirteen")
            elif last == '5':
                return len("fifteen")
            elif last == '8':
                return len("eighteen")
            else:
                return len(words[last] + "teen")
        elif first == '2':
            return len("twenty" + words[last])
        elif first == '3':
            return len("thirty" + words[last])
        elif first == '4':
            return len("forty" + words[last])
        elif first == '5':
            return len("fifty" + words[last])
        elif first == '8':
            return len("eighty" + words[last])
        else:
            return len(words[num[0]] + "ty" + words[last])

    letters = 0
    # run through all of the numbers and add up the letters
    for num in range(1,N+1):
        num = str(num)
        if len(num) == 1:
            letters += single(num)
            continue 
        
        if len(num) == 2:
            letters += double(num)
            continue
        
        if len(num) == 3:
            letters += len(words[num[0]] + "hundred")
            num = num[1:]       # make the number 2 digits to run the double function
            if num == '00':     # don't add anything if it is an even hundred
                continue
            letters += (double(num) + 3) # extra 3 accounts for the 'and' in the number
            continue

        if len(num) == 4:
            letters += len("onethousand")
    
    return letters

if __name__ == "__main__":
    print(letters_used(1000))
    pass
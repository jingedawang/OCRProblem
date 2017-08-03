# Caculate four fundamental rules expression

import io
from collections import deque
import time


def delSpace(s):
    """ Delete blank space in String 
    """
    s1 = ''
    for a in s:
        # if a is blank space,
        if a != ' ':
            s1 += a
    return s1


def toDigitl(sl):
    """ 
    INPUT:['1','+','2.3','+','4'] OUTPT:[1,'+',2.3,'+',4] 
    """
    d = []
    sign = ['+', '-', '*', '/', '(', ')']
    for s in sl:
        if (s in sign):
            d.append(s)
        elif s.isnumeric():
            d.append(int(s))
        else:
            d.append(float(s))
    return d


def tokenList(s):
    """ 
    e.g. s = '1+2+4' , return ['1','+','2','+','4'] 
    """
    t1 = []
    sp = ''
    sign = ['+', '-', '*', '/', '(', ')']
    j, k = 1, 0
    for i in range(len(s)):
        if s[i] in sign:
            if s[i - 1] in sign:
                sp = s[k:k + j]
                j += 1
            else:
                t1.append(sp)
            t1.append(s[i])
            j = 1
            k = i + 1
        else:
            sp = s[k:k + j]
            j += 1
            # process last char
        if i == len(s) - 1:
            t1.append(sp)
            # pop right bracket
        if i == len(s) - 1 and s[i] in sign:
            t1.pop()
    return t1


def fourRules(a, b, s):
    if s == '+': return a + b
    if s == '-': return a - b
    if s == '*': return a * b
    if s == '/': return a / b


def xcacu(ns, sg):
    b1 = ns.pop()
    a1 = ns.pop()
    r = fourRules(a1, b1, sg)
    ns.append(r)


def caculate(tokenlist):
    numStack = deque()
    opStack = deque()
    ops = ['+', '-', '*', '/', '(', ')']
    opps = {'+': 0, '-': 0, '*': 1, '/': 1, '(': 2, ')': 2}
    opCurrent = ''

    for i in range(len(tokenlist)):
        if tokenlist[i] in ops:
            opStack.append(tokenlist[i])
        else:
            numStack.append(tokenlist[i])
            if i < len(tokenlist) - 1 and len(numStack) >= 2 and len(opStack) >= 1:
                opCurrent = opStack.pop()
                if opps[tokenlist[i + 1]] <= opps[opCurrent]:
                    xcacu(numStack, opCurrent)
                else:
                    opStack.append(opCurrent)
                if i == len(tokenlist) - 1:
                    xcacu(numStack, opCurrent)
                    # print(i,numStack,opStack)
    while (len(opStack) > 0):
        xcacu(numStack, opStack.pop())
    return numStack.pop()


def main():
    s = input()
    print s
    while s != 'q':
        starttime = time.time()
        print s
        tt = toDigitl(tokenList(delSpace(s)))
        print('%s is %.2f.' % (s, caculate(tt)))

        endtime = time.time() - starttime
        print('RUN TIME:%.10f ms' % (endtime * 1000))

        s = input()


if __name__ == '__main__':
    main()
import random as r
import numpy as np
I = 0

def f(i):
    print(1+1)
    I = i
    asd = g()
    print(I)
    return asd
def g():
    print("I",I)
    return 2

min = 0
max = 150
numero = np.random.randint(min, max)
np.random.seed(0)
lista = np.random.randint(min, max, 10)
print(lista)
print(f(1))

print(I)
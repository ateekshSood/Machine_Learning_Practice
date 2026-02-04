
import math

a , b , c = 2 , 3 , 4
s = ( a + b + c )/2
print(s)
print((s * (s - a) * (s - b ) * (s-c)) ** 1/2)
print(math.sqrt((s * (s - a) * (s - b ) * (s-c))))
def firstn(n):
    num = 0
    while num < n:
        yield num
        num += 1

sum_of_first_n=0
for i in firstn(1000):
    sum_of_first_n += i

print(sum_of_first_n)

x = [1, 0, 1, 5, 4, 0]

y1 = [1/e if e else 0 for e in x]
y2 = [1/e if e != 0 else 0 for e in x]

print(y1)
print(y2)

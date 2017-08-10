import itertools

d = {"cell1" : [11,11,11], "x0" : [1,1,1], "x1" : [2,2,2], "x3" : [3,3,3], "x5" : [4,4,4], "x4" : [6,6,6], "y0" : [10,10,10]}
testfile = open("testfile.txt", "w")

table = []

for row in zip(*([key] + value for key, value in sorted(d.items()))):
    table.append(row)

print table

testfile.write( "\t".join(table[0]))

for i in range(1, len(table), 1):
    testfile.write("\n")
    for x in range(len(table[i])):
        testfile.write(str(table[i][x]) + "\t")


testfile.close()
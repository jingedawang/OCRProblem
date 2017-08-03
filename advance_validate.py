


f = open("accuracy_validation_2nd.txt", "r")
lines = f.readlines()
for j in range(lines.__len__()):
    line = lines[j]
    index_left = -1
    index_right = -1
    for i in range(line.__len__()):
        if line[i] == '(':
            index_left = i
        elif line[i] == ')':
            index_right = i
    if index_left == -1 and index_right == -1:
        continue
    elif index_right < index_left:
        print "Wrong at index " + str(j) + ": " + line
    elif index_right - index_left != 4:
        print "Wrong at index " + str(j) + ": " + line
    # else:
    #     print "Right at index " + str(j)
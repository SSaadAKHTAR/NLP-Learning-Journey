# Task 1
# Implement the MIN-EDIT-DISTANCE function from the pseudocode. Use the Levenshtein Distance
def min_edit_distance(source, target):
    n = len(source)
    m = len(target)

    D = [[0 for j in range(m + 1)] for i in range(n + 1)]


    for i in range(1, n + 1):
        D[i][0] = D[i-1][0] + 1 
    for j in range(1, m + 1):
        D[0][j] = D[0][j-1] + 1  

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if source[i - 1] == target[j - 1]:
                cost = 0
            else:
                cost = 2

            D[i][j] = min(D[i - 1][j] + 1,      
                           D[i][j - 1] + 1,     
                           D[i - 1][j - 1] + cost)  
        # print("/n", D)

    return D[n][m]


D = min_edit_distance("intention", "execution")
print(D)



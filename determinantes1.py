import numpy as np

n = 3
A = np.zeros((6 * n - 2, n ** 2), int)
for j in range(0, n ** 2):
    for i in range(0, n):
        A[i, j] = int(i == j % n)
    for i in range(0, n):
        A[i + n, j] = int(i == j // n)
    for i in range(0, 2 * n - 1):
        A[i + 2 * n, j] = int(i == n - 1 + j // n - j % n)
    for i in range(0, 2 * n - 1):
        A[i + 4 * n - 1, j] = int(i == j // n + j % n)


print(A)
print(np.linalg.det(np.matmul(np.transpose(A), A) + 0 * np.identity(n ** 2, int)))
print(np.linalg.det(np.matmul(np.transpose(A), A) + 10e-3 * np.identity(n ** 2, int)))
print(np.linalg.det(np.matmul(np.transpose(A), A) + 10e-2 * np.identity(n ** 2, int)))
print(np.linalg.det(np.matmul(np.transpose(A), A) + 10e-1 * np.identity(n ** 2, int)))

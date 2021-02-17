import numpy as np


def getA(size):
    # A1 = np.zeros((6 * n1 - 2, n1 ** 2), int)
    A = np.zeros((2 * size, size ** 2), int)
    for j in range(0, size ** 2):
        for i in range(0, size):
            A[i, j] = int(i == j % size)
        for i in range(0, size):
            A[i + size, j] = int(i == j // size)
        # for i in range(0, 2 * n1 - 1):
        #     A[i + 2 * n1, j] = int(i == n1 - 1 + j // n1 - j % n1)
        # for i in range(0, 2 * n1 - 1):
        #     A[i + 4 * n1 - 1, j] = int(i == j // n1 + j % n1)
    return A


def printDeterminants(A, n, deltas=[0, 1e-3, 1e-2, 1e-1], label=''):
    print(label, f"(n = {n})" + ': ')
    # print(A)
    AtA = np.matmul(np.transpose(A), A)
    I = np.identity(n ** 2, int)
    for delta in deltas:
        print(f"d ={delta: .1E}: {np.linalg.det(AtA + delta * I)}")
    print('')


def main(path='EP1_dados/'):
    p1 = np.load(path + 'im1/p1.npy')
    n1 = int(p1.size / 2)
    print('p1:', p1, f"n1 = {n1}", '\n')
    A1 = getA(n1)

    p2 = np.load(path + 'im2/p1.npy')
    n2 = int(p2.size / 2)
    print('p2:', p2, f"n2 = {n2}", '\n')
    A2 = getA(n2)

    p3 = np.load(path + 'im3/p1.npy')
    n3 = int(p3.size / 2)
    print('p3:', p3, f"n3 = {n3}", '\n')
    A3 = getA(n3)

    printDeterminants(A1, n1, label='im1')
    printDeterminants(A2, n2, label='im2')
    printDeterminants(A3, n3, label='im3')


main()

import numpy as np


def calculateA(size, A=np.zeros(0)):
    if A.size == 0:
        A = np.zeros((2 * size, size ** 2), int)
    for j in range(0, size ** 2):
        for i in range(0, size):
            A[i, j] = int(i == j % size)
        for i in range(0, size):
            A[i + size, j] = int(i == j // size)
    return A


def calculateDiagonalA(size, A=np.zeros(0)):
    if A.size == 0:
        A = np.zeros((6 * size - 2, size ** 2), int)
    for j in range(0, size ** 2):
        for i in range(0, size):
            A[i, j] = int(i == j % size)
        for i in range(0, size):
            A[i + size, j] = int(i == j // size)
        for i in range(0, 2 * size - 1):
            A[i + 2 * size, j] = int(i == size - 1 + j // size - j % size)
        for i in range(0, 2 * size - 1):
            A[i + 4 * size - 1, j] = int(i == j // size + j % size)
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
    print('p1:\n')

    im1_p1 = np.load(path + 'im1/p1.npy')
    im1_n1 = int(im1_p1.size / 2)
    print('im1_p1:', im1_p1, f"im1_n1 = {im1_n1}\n")
    im1_A1 = calculateA(im1_n1)

    im2_p1 = np.load(path + 'im2/p1.npy')
    im2_n1 = int(im2_p1.size / 2)
    print('im2_p1:', im2_p1, f"im2_n1 = {im2_n1}\n")
    im2_A1 = calculateA(im2_n1)

    im3_p1 = np.load(path + 'im3/p1.npy')
    im3_n1 = int(im3_p1.size / 2)
    print('im3_p1:', im3_p1, f"im3_n1 = {im3_n1}\n")
    im3_A1 = calculateA(im3_n1)

    printDeterminants(im1_A1, im1_n1, label='im1_p1')
    printDeterminants(im2_A1, im2_n1, label='im2_p1')
    printDeterminants(im3_A1, im3_n1, label='im3_p1')

    print('p2:\n')

    im1_p2 = np.load(path + 'im1/p2.npy')
    im1_n2 = int((im1_p2.size + 2) / 6)
    print('im1_p2:', im1_p2, f"im1_n2 = {im1_n2}\n")
    im1_A2 = calculateDiagonalA(im1_n2)

    im2_p2 = np.load(path + 'im2/p2.npy')
    im2_n2 = int((im2_p2.size + 2)/ 6)
    print('im2_p2:', im2_p1, f"im2_n2 = {im2_n2}\n")
    im2_A2 = calculateDiagonalA(im2_n1)

    im3_p2 = np.load(path + 'im3/p2.npy')
    im3_n2 = int((im3_p2.size + 2)/ 6)
    print('im3_p2:', im3_p1, f"im3_n2 = {im3_n2}\n")
    im3_A2 = calculateDiagonalA(im3_n1)

    printDeterminants(im1_A2, im1_n2, label='im1_p2')
    printDeterminants(im2_A2, im2_n2, label='im2_p2')
    printDeterminants(im3_A2, im3_n2, label='im3_p2')


main()

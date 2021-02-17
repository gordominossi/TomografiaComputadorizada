import numpy as np
import matplotlib.pyplot as plt


'''
' Monta a matriz A para o caso das projeções verticais e horizontais
' Recebe o número n (size) usado para discretizar a imagem e uma matriz A a ser preenchida (opcional)
'''


def calculateA(size, A=np.zeros(0)):
    if A.size == 0:
        A = np.zeros((2 * size, size ** 2), int)
    for j in range(0, size ** 2):
        for i in range(0, size):
            A[i, j] = int(i == j % size)
        for i in range(0, size):
            A[i + size, j] = int(i == j // size)
    return A


'''
' Monta a matriz A para o caso das projeções verticais, horizontais e diagonais
' Recebe o número n (size) usado para discretizar a imagem e uma matriz A a ser preenchida (opcional)
'''


def calculateDiagonalA(size, A=np.zeros(0)):
    if A.size == 0:
        A = np.zeros((6 * size - 2, size ** 2), int)
    A = calculateA(size, A)
    for j in range(0, size ** 2):
        for i in range(0, 2 * size - 1):
            A[i + 2 * size, j] = int(i == size - 1 + j // size - j % size)
        for i in range(0, 2 * size - 1):
            A[i + 4 * size - 1, j] = int(i == j // size + j % size)
    return A


'''
' Calcula os determinantes da multiplicação A_transposta * A mais um termo de regularização delta * I
' a fim de minimizar a norma euclidiana da solução
'''


def calculateDeterminants(A, n, deltas=[0, 1e-3, 1e-2, 1e-1]):
    dets = []
    AtA = np.matmul(np.transpose(A), A)
    I = np.identity(n ** 2, int)
    for delta in deltas:
        dets.append(np.linalg.det(AtA + delta * I))
    return dets


'''
' Função principal do programa
' Calcula os determinantes para as figuras dadas e exibe uma tabela com os resultados
'''


def main(path='EP1_dados/', deltas=[0, 1e-3, 1e-2, 1e-1]):

    # Carrega informações de p1
    im1_p1 = np.load(path + 'im1/p1.npy')
    im2_p1 = np.load(path + 'im2/p1.npy')
    im3_p1 = np.load(path + 'im3/p1.npy')

    im1_n1 = int(im1_p1.size / 2)
    im1_A1 = calculateA(im1_n1)

    im2_n1 = int(im2_p1.size / 2)
    im2_A1 = calculateA(im2_n1)

    im3_n1 = int(im3_p1.size / 2)
    im3_A1 = calculateA(im3_n1)

    # print('\n Calculando determinantes de p1:\n_____\n')

    dets1_1 = calculateDeterminants(im1_A1, im1_n1, deltas)
    dets1_2 = calculateDeterminants(im2_A1, im2_n1, deltas)
    dets1_3 = calculateDeterminants(im3_A1, im3_n1, deltas)

    # Carrega informações de p2
    im1_p2 = np.load(path + 'im1/p2.npy')
    im2_p2 = np.load(path + 'im2/p2.npy')
    im3_p2 = np.load(path + 'im3/p2.npy')

    im1_n2 = int((im1_p2.size + 2) / 6)
    im1_A2 = calculateDiagonalA(im1_n2)

    im2_n2 = int((im2_p2.size + 2) / 6)
    im2_A2 = calculateDiagonalA(im2_n1)

    im3_n2 = int((im3_p2.size + 2) / 6)
    im3_A2 = calculateDiagonalA(im3_n1)

    # print('\n Calculando determinantes de p2:\n_____\n')

    dets2_1 = calculateDeterminants(im1_A2, im1_n2, deltas)
    dets2_2 = calculateDeterminants(im2_A2, im2_n2, deltas)
    dets2_3 = calculateDeterminants(im3_A2, im3_n2, deltas)

    # Configura a tabela referente a p1 para exibição
    fig1, ax1 = plt.subplots()
    fig1.patch.set_visible(False)
    fig1.suptitle("Projeções horizontais e verticais")
    fig1.tight_layout()
    ax1.axis('off')
    ax1.axis('tight')

    table1 = ax1.table(colLabels=list(map(lambda delta: f"delta = {delta:.0e}", deltas)), rowLabels=[
        ' im1 ', ' im2 ',  ' im3 '], cellText=[dets1_1, dets1_2, dets1_3], loc='top')

    table1.auto_set_font_size(False)
    table1.set_fontsize(6)
    fig1.suptitle("Projeções horizontais e verticais")
    fig1.tight_layout()

    # Configura a tabela referente a p2 para exibição
    fig2, ax2 = plt.subplots()
    fig2.patch.set_visible(False)
    ax2.axis('off')
    ax2.axis('tight')

    table2 = ax2.table(colLabels=list(map(lambda delta: f"delta = {delta:.0e}", deltas)), rowLabels=[
        ' im1 ', ' im2 ',  ' im3 '], cellText=[dets2_1, dets2_2, dets2_3], loc='top')

    table2.auto_set_font_size(False)
    table2.set_fontsize(6)
    fig2.suptitle("Projeções horizontais, verticais e diagonais")
    fig2.tight_layout()

    # Exibe as tabelas
    plt.show()


main()

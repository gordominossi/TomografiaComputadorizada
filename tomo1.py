import determinantes
import numpy as np
import matplotlib.pyplot as plt


def main(path='EP1_dados/', deltas=[0, 1e-3, 1e-2, 1e-1]):
 # Carrega informações de p1
    im1_p1 = np.load(path + 'im1/p1.npy')
    im2_p1 = np.load(path + 'im2/p1.npy')
    im3_p1 = np.load(path + 'im3/p1.npy')

    im1_n1 = int(im1_p1.size / 2)
    im1_A1 = determinantes.calculateA(im1_n1)

    im2_n1 = int(im2_p1.size / 2)
    im2_A1 = determinantes.calculateA(im2_n1)

    im3_n1 = int(im3_p1.size / 2)
    im3_A1 = determinantes.calculateA(im3_n1)

    # print('\n Calculando determinantes de p1:\n_____\n')

    dets1_1 = determinantes.calculateDeterminants(im1_A1, im1_n1, deltas)
    dets2_1 = determinantes.calculateDeterminants(im2_A1, im2_n1, deltas)
    dets3_1 = determinantes.calculateDeterminants(im3_A1, im3_n1, deltas)

    # Configura as imagens referentes a f1 para exibição
    fig1, axs1 = plt.subplots(1, len(dets1_1) + 1)
    fig1.patch.set_visible(False)
    for i in range(len(dets1_1)):
        axs1[i].axis('off')
        if dets1_1[i] != 0:
            f1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(
                im1_A1.T, im1_A1) + deltas[i] * np.identity(im1_n1 ** 2, int)), im1_A1.T), im1_p1)
            axs1[i].imshow(f1.reshape(im1_n1, im1_n1).T)
    axs1[-1].axis('off')
    axs1[-1].imshow(plt.imread(path + 'im1/im1.png'))

    indexDet2_1 = np.argmin(np.ma.MaskedArray(dets2_1, mask=dets2_1 != 0))
    f2 = np.matmul(np.matmul(np.linalg.inv(np.matmul(im2_A1.T, im2_A1) + np.flipud(
        deltas)[indexDet2_1] * np.identity(im2_n1 ** 2, int)), im2_A1.T), im2_p1)

    fig2, ax2 = plt.subplots()
    fig2.patch.set_visible(False)
    ax2.axis('off')
    ax2.imshow(f2.reshape(im2_n1, im2_n1).T)

    indexDet3_1 = np.argmin(np.ma.MaskedArray(dets3_1, mask=dets3_1 != 0))
    f3 = np.matmul(np.matmul(np.linalg.inv(np.matmul(im3_A1.T, im3_A1) + np.flipud(
        deltas)[indexDet3_1] * np.identity(im3_n1 ** 2, int)), im3_A1.T), im3_p1)

    fig3, ax3 = plt.subplots()
    fig3.patch.set_visible(False)
    ax3.axis('off')
    ax3.imshow(f3.reshape(im3_n1, im3_n1).T)

    # Exibe as imagens
    plt.show()


main()

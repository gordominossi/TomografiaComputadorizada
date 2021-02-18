import determinantes
import numpy as np
import matplotlib.pyplot as plt


'''
' Encontra a solução f para (At*A + d * I_n) * f = At * p
'''


def solveF(A, delta, n, p):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A) + delta * np.identity(n ** 2, int)), A.T), p)


'''
' Função principal do programa
' Calcula as soluções f para as figuras dadas e exibe uma tabela com os resultados
'''


def main(path='EP1_dados/', deltas=[0, 1e-3, 1e-2, 1e-1]):
    # Carrega informações de p1
    im1_p2 = np.load(path + 'im1/p2.npy')
    im2_p2 = np.load(path + 'im2/p2.npy')
    im3_p2 = np.load(path + 'im3/p2.npy')
    ims_p2 = [im1_p2, im2_p2, im3_p2]

    ims_n2 = []
    ims_A2 = []
    dets2 = []
    print('\n Calculando determinantes de A2...\n')
    for j in range(len(ims_p2)):
        ims_n2.append(int((ims_p2[j].size + 2) / 6))
        ims_A2.append(determinantes.calculateDiagonalA(ims_n2[j]))
        dets2.append(determinantes.calculateDeterminants(
            ims_A2[j], ims_n2[j], deltas))

    # Configura as imagens referentes a f para exibição
    for i in range(len(ims_p2)):
        fig, axs = plt.subplots(1, len(dets2[i]) + 1)
        fig.patch.set_visible(False)
        for j in range(len(dets2[i])):
            axs[j].axis('off')
            try:
                # Encontra a solução f para as projeções p dadas
                f = solveF(ims_A2[i], deltas[j], ims_n2[i], ims_p2[i])
                axs[j].imshow(f.reshape(ims_n2[i], ims_n2[i]).T)
            except:
                pass
            axs[j].set_title(f'delta = {deltas[j]}')
            axs[j].title.set_fontsize(6)
        # Configura a imagem de f*
        axs[-1].axis('off')
        axs[-1].imshow(plt.imread(path + f'im{i + 1}/im{i + 1}.png'))
        axs[-1].set_title('f*')
        axs[-1].title.set_fontsize(6)
        fig.suptitle(f'Plotagem de f para im{i + 1}')

    # Exibe as imagens
    plt.show()


main()

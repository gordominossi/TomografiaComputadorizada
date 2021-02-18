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
    im1_p1 = np.load(path + 'im1/p1.npy')
    im2_p1 = np.load(path + 'im2/p1.npy')
    im3_p1 = np.load(path + 'im3/p1.npy')
    ims_p1 = [im1_p1, im2_p1, im3_p1]

    ims_n1 = []
    ims_A1 = []
    dets1 = []
    print('\n Calculando determinantes de p1...\n')
    for j in range(len(ims_p1)):
        ims_n1.append(int(ims_p1[j].size / 2))
        ims_A1.append(determinantes.calculateA(ims_n1[j]))
        dets1.append(determinantes.calculateDeterminants(
            ims_A1[j], ims_n1[j], deltas))

    # Configura as imagens referentes a f1 para exibição
    for i in range(len(ims_p1)):
        fig, axs = plt.subplots(1, len(dets1[i]) + 1)
        fig.patch.set_visible(False)
        for j in range(len(dets1[i])):
            axs[j].axis('off')
            try:
                # Encontra a solução f para as projeções p dadas
                f = solveF(ims_A1[i], deltas[j], ims_n1[i], ims_p1[i])
                axs[j].imshow(f.reshape(ims_n1[i], ims_n1[i]).T)
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

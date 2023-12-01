from keras.layers import Input
from keras.models import Model
import argparse
from utils.util_module import *
from utils.util_metric import *
from utils.util_norm import *
from utils.util_pnp import *
import math
import time


def block_solver(VT, encode_dim):
    V11V11T = np.matmul(np.transpose(VT[0:encode_dim, 0:encode_dim], (1, 0)), VT[0:encode_dim, 0:encode_dim])
    V12V12T = np.matmul(np.transpose(VT[encode_dim:2048, 0:encode_dim], (1, 0)),
                        VT[encode_dim:2048, 0:encode_dim])
    V11V21T = np.matmul(np.transpose(VT[0:encode_dim, 0:encode_dim], (1, 0)),
                        VT[0:encode_dim, encode_dim:2048])
    V12V22T = np.matmul(np.transpose(VT[encode_dim:2048, 0:encode_dim], (1, 0)),
                        VT[encode_dim:2048, encode_dim:2048])
    V21V21T = np.matmul(np.transpose(VT[0:encode_dim, encode_dim:2048], (1, 0)),
                        VT[0:encode_dim, encode_dim:2048])
    V22V22T = np.matmul(np.transpose(VT[encode_dim:2048, encode_dim:2048], (1, 0)),
                        VT[encode_dim:16384, encode_dim:2048])
    return np.repeat(V11V11T[np.newaxis, :, :], 100, axis=0), np.repeat(V12V12T[np.newaxis, :, :], 100,
                                                                        axis=0), np.repeat(V11V21T[np.newaxis, :, :],
                                                                                           100, axis=0), np.repeat(
        V12V22T[np.newaxis, :, :], 100, axis=0), np.repeat(V21V21T[np.newaxis, :, :], 100, axis=0), np.repeat(
        V22V22T[np.newaxis, :, :], 100, axis=0)


def svd_solver(V11V11T, V12V12T, V11V21T, V12V22T, V21V21T, V22V22T, encode_dim, rho):
    P_inv = np.zeros((100, 2048, 2048))
    P_inv[:, 0:encode_dim, 0:encode_dim] = np.dot(1 / (2 + rho), V11V11T) + np.dot(1 / rho, V12V12T)
    P_inv[:, 0:encode_dim, encode_dim:2048] = np.dot(1 / (2 + rho), V11V21T) + np.dot(1 / rho, V12V22T)
    P_inv[:, encode_dim:2048, 0:encode_dim] = np.transpose(P_inv[:, 0:encode_dim, encode_dim:2048], (0, 2, 1))
    P_inv[:, encode_dim:2048, encode_dim:2048] = np.dot(1 / (2 + rho), V21V21T) + np.dot(1 / rho, V22V22T)
    return P_inv


def quan_value(x, mu, B):
    sign_x = x >= 0
    sign_x = 2 * (sign_x.astype(int) - 0.5)
    x = sign_x * np.log(1 + mu * sign_x * x) / np.log(1 + mu)

    x = (x + 1) / 2
    level = 2 ** B
    q_value = np.round(x * level - 0.5)
    return q_value


def dequan_value(x, mu, B):
    level = 2 ** B
    x = (x + 0.5) / level
    x = x * 2 - 1
    sign_x = x >= 0
    sign_x = 2 * (sign_x.astype(int) - 0.5)
    dq_value = sign_x * ((1 + mu) ** (sign_x * x) - 1) / mu
    return dq_value


def main(args):
    # data loading
    data = np.load('data_npz/pppcf_data.npz')
    x_FS_origin_r = data['x_test']
    x_FS_origin_c = R2C(x_FS_origin_r)
    x_DA_trun_c = HS_to_DA(x_FS_origin_c)
    x_DA_trun_r = C2R(x_DA_trun_c)

    del data, x_FS_origin_r, x_DA_trun_c

    # model loading
    channel_input = Input(shape=(32, 32, 2))
    channel_noise = Input(shape=(1,))
    channel_output = fddnet(channel_input, channel_noise, block_num=args.block_num, norm=args.norm,
                            dropout=args.dropout,
                            weight=args.weight, dim=48)
    denoising_model = Model(inputs=[channel_input, channel_noise], outputs=channel_output, name='denoising')
    denoising_model.load_weights(args.model_name_path)

    # parameters init
    encode_dim = args.encode_dim
    if encode_dim == 512:
        rhos_init = 1e-10
        sigma_init = 0.1
        set_num = 64
    elif encode_dim == 256:
        rhos_init = 1e-12
        sigma_init = 0.1
        set_num = 32
    elif encode_dim == 128:
        rhos_init = 1e-12
        sigma_init = 1
        set_num = 32
    elif encode_dim == 64:
        rhos_init = 1e-12
        sigma_init = 1
        set_num = 5
    elif encode_dim == 32:
        rhos_init = 1e-13
        sigma_init = 1
        set_num = 2

    itr = 20
    rhos = [rhos_init * math.pow(1.5, i) for i in range(itr)]
    sigmas = [sigma_init * math.pow(0.8, i) for i in range(itr)]

    # results
    B = args.quan_bit
    mu = 1300

    '''
    (1) generate random matrix A
        U,_,_ = SVD(A)
        D = U[0:encode_dim]
    '''
    D_ = np.load('data_npz/random_matrix.npz')
    D_ = D_['U']
    D = D_[0:encode_dim, :]
    del D_

    _, _, VT = np.linalg.svd(D)
    V11V11T, V12V12T, V11V21T, V12V22T, V21V21T, V22V22T = block_solver(VT, encode_dim)
    del VT

    D = np.repeat(D[np.newaxis, :, :], 100, axis=0)

    # SVD+分块加速 #
    t1 = time.time()
    '''
    (2) compression
    '''
    H_real = np.squeeze(x_DA_trun_r).reshape((100, 2048, 1))
    Y = np.matmul(D, H_real)

    '''
    (3) quan
    '''
    Y_quan = np.zeros((100, encode_dim, 1))
    for quan_index in range(100):
        Y_quan[quan_index, :, :] = quan_value(Y[quan_index, :, :], mu=mu, B=B)

    '''
    (4) dequan
    '''
    Y_dequan = np.zeros((100, encode_dim, 1))
    for dequan_index in range(100):
        Y_dequan[dequan_index, :, :] = dequan_value(Y_quan[dequan_index, :, :], mu=mu, B=B)

    '''
    (7) pnp solver
    '''
    # init
    max_set = np.abs(np.matmul(np.transpose(D, (0, 2, 1)), Y_dequan))
    index_max = list(np.argsort(max_set, axis=1)[:, ::-1][:, 0:set_num])
    H_ = np.zeros((100, 2048, 1))
    for j in range(100):
        index_max_ = list(index_max[j][:, 0])
        H_[j, index_max_, 0] = np.dot(np.linalg.pinv(D[j, :, index_max_].T), Y[j, :, 0])
    error = np.sum((H_real - H_) ** 2)
    # pnp
    for k in range(itr):
        rho = rhos[k]
        sigma = np.tile(np.expand_dims(np.array(sigmas[k], ), axis=0), 100)
        P_inv = svd_solver(V11V11T, V12V12T, V11V21T, V12V22T, V21V21T, V22V22T, encode_dim, rho)
        Q = 2 * np.matmul(np.transpose(D, (0, 2, 1)), Y) + rho * H_
        Z = np.matmul(P_inv, Q)
        Z_ = np.reshape(Z, (100, 32, 32, 2))
        H = denoising_model.predict([Z_, sigma])
        H_ = np.reshape(H, (100, 2048, 1))
        H_error = np.sum((H_real - H_) ** 2)
        if H_error < error:
            error = H_error
        elif H_error >= error and k >= 3:
            break
        x_FS_pre_c = DA_to_HS(R2C(Z_))
        _, nmse = cal_nmse_c(x_FS_origin_c, x_FS_pre_c)
        _, gcs = cal_cosine_similarity_tensor(x_FS_origin_c, x_FS_pre_c)
        print("itr{} error {} ".format(k + 1, nmse))
    t2 = time.time()
    print("Execute time of one block of 100 samples:{}".format(t2 - t1))
    print("Eocode_dim {} NMSE {}dB GCS {}".format(encode_dim, nmse, gcs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ''' System parameter'''
    parser.add_argument("-ed", "--encode_dim", default=256, type=int)
    parser.add_argument("-B", "--quan_bit", default=6, type=int)
    parser.add_argument("-b", "--block_num", default=8, type=int)
    parser.add_argument("-n", "--norm", default='none', type=str)
    parser.add_argument("-w", "--weight", default=1e-6, type=float)
    parser.add_argument("-d", "--dropout", default=0, type=float)
    parser.add_argument("-m", "--model_name", default='fddnet', type=str)
    parser.add_argument("-mp", "--model_name_path", default='./model_zoo/ffdnet.h5', type=str)
    args = parser.parse_args()
    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")
    main(args)

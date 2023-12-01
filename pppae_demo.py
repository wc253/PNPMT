from keras.layers import Input
from keras.models import Model
import argparse
from utils.util_module import *
from utils.util_norm import *
from utils.util_pnp import *
import math


def main(args):
    # data loading
    data = np.load('data_npz/pppae_data.npz')
    H_true = data['x_test']
    del data

    Number_of_antenna = args.Number_of_antenna
    antenna_mode = args.mode
    antenna_location, r, c = get_dh_ae(Number_of_antenna, antenna_mode)
    antenna_location_inverse = (~(antenna_location.astype(np.bool))).astype(np.float)

    # parameters init
    itr = 10
    rhos_init = 1e-1
    sigma_init = 0.1
    rhos = [rhos_init * math.pow(1.5, i) for i in range(itr)]
    sigmas = [sigma_init * math.pow(0.8, i) for i in range(itr)]

    # model loading
    if args.model_name == 'fddnet':
        channel_input = Input(shape=(32, 32, 2))
        channel_noise = Input(shape=(1,))
        channel_output = fddnet(channel_input, channel_noise, block_num=args.block_num, norm=args.norm,
                                dropout=args.dropout,
                                weight=args.weight, dim=48)
        denoising_model = Model(inputs=[channel_input, channel_noise], outputs=channel_output, name='denoising')
        denoising_model.load_weights(args.model_name_path)

    snr = 20
    noise_power = 1 / np.power(10, snr / 10)
    noise_ = np.zeros((1, 256, 32)) + 1j * np.zeros((1, 256, 32))
    for i in range(1):
        noise_1 = np.random.randn(256, 32) + 1j * np.random.randn(256, 32)
        noise_[i, :, :] = np.sqrt(noise_power) * noise_1 / np.sqrt(np.sum(np.abs(noise_1) ** 2))
    noise = C2R(noise_)
    del noise_
    Y_all_R = H_true + noise
    del noise

    if Number_of_antenna == 8:
        Y_pilot = np.zeros((1, 256, 8, 2))
    elif Number_of_antenna == 16:
        Y_pilot = np.zeros((1, 256, 16, 2))

    for i in range(1):
        for j in range(Y_pilot.shape[2]):
            for k in range(Y_pilot.shape[1]):
                h = r[j * Y_pilot.shape[1] + k]
                w = c[j * Y_pilot.shape[1] + k]
                Y_pilot[i, k, j, :] = Y_all_R[i, h, w, :]

    # results
    PPPAE_NMSE = []
    '''
    (1) generate data Y_LS
    '''
    H_true_r = H_true[0, :, :, 0]
    H_true_i = H_true[0, :, :, 1]
    H_true_ = H_true_r + 1j * H_true_i

    H_RBF = interp(Y_pilot[0, :, :, :], r, c, interp='rbf')

    H_ = H_RBF
    error = cal_nmse_single_c(H_true_, H_RBF)
    '''
    (7) pnp solver
    '''
    # pnp
    for k in range(itr):
        rho = rhos[k]
        sigma = np.tile(np.expand_dims(np.array(sigmas[k], ), axis=0), 1)

        Z = ((antenna_location * H_RBF + antenna_location_inverse * H_) + rho * H_) / (1 + rho)
        Z_error = cal_nmse_single_c(H_true_, Z)

        # SF to AD
        Z_DS = np.fft.ifft(Z, axis=0)
        Z_AD = np.fft.fft(Z_DS, axis=1)
        Z_AD_trun = Z_AD[0:32, :]
        Z_AD_trun_r = np.real(Z_AD_trun)
        Z_AD_trun_r = Z_AD_trun_r[:, :, np.newaxis]
        Z_AD_trun_i = np.imag(Z_AD_trun)
        Z_AD_trun_i = Z_AD_trun_i[:, :, np.newaxis]
        Z_AD_trun_ = np.concatenate([Z_AD_trun_r, Z_AD_trun_i], axis=-1)
        Z_AD_trun_ = Z_AD_trun_[np.newaxis, :, :, :]

        H_AD_trun = denoising_model.predict([Z_AD_trun_, sigma])

        # AD to SF
        H_AD_trun_r = H_AD_trun[:, :, :, 0]
        H_AD_trun_i = H_AD_trun[:, :, :, 1]
        H_AD_trun_ = H_AD_trun_r + 1j * H_AD_trun_i
        H_AD_trun_ = np.squeeze(H_AD_trun_)
        H_AD_c = np.concatenate([H_AD_trun_, np.zeros(shape=[224, 32])])
        H_DS = np.fft.ifft(H_AD_c, axis=1)
        H_SF = np.fft.fft(H_DS, axis=0)
        H_ = H_SF

        H_error = cal_nmse_single_c(H_true_, H_)
        print("itr{} Z_error {} H_ERROR{}".format(k + 1, Z_error, H_error))
    PPPAE_NMSE.append(cal_nmse_single_c(H_true_, H_))
    print("snr {}dB PNPAE_NMSE {}dB".format(snr, PPPAE_NMSE[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ''' System parameter'''
    parser.add_argument("-na", "--Number_of_antenna", default=16, type=int)
    parser.add_argument("-mode", "--mode", default=1, type=int)
    parser.add_argument("-b", "--block_num", default=8, type=int)
    parser.add_argument("-n", "--norm", default='none', type=str)
    parser.add_argument("-w", "--weight", default=1e-6, type=float)
    parser.add_argument("-d", "--dropout", default=0, type=float)
    parser.add_argument("-m", "--model_name", default='fddnet', type=str)
    parser.add_argument("-mp", "--model_name_path",
                        default='./model_zoo/ffdnet.h5', type=str)
    args = parser.parse_args()
    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")
    main(args)

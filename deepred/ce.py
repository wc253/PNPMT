"""Real-Hilbert-space RED equilibrium solver (not an asserted energy gradient).

Latent arrays store complex AD coefficients; FFDNet sees real/imag channels.
S is the original zero-padding/IFFT/FFT synthesis. S* = 8 T, S*S = 8 I.
A = M S / sqrt(8), b = y / sqrt(8), so ||A|| <= 1. This scaling
keeps the original FFDNet input scale and absorbs data weighting into lambda.
"""
import time
import numpy as np


def analysis(h):
    return np.fft.fft(np.fft.ifft(h, axis=0), axis=1)[:32]


def synthesis(x):
    return np.fft.fft(np.fft.ifft(np.pad(x, ((0, 224), (0, 0))), axis=1), axis=0)


class Operator:
    def __init__(self, mask):
        self.mask = np.asarray(mask)
        if self.mask.shape != (256, 32) or not np.all(np.isin(self.mask, [0, 1])):
            raise ValueError('Expected binary (256,32) mask')

    def forward(self, x):
        return self.mask * synthesis(x) / np.sqrt(8)

    def adjoint(self, y):
        return np.sqrt(8) * analysis(self.mask * y)


def nmse(h, truth):
    return float(10 * np.log10(max(np.linalg.norm(h-truth)**2, 1e-300) /
                              np.linalg.norm(truth)**2))


class FFDNet:
    def __init__(self, weights):
        from keras.layers import Input
        from keras.models import Model
        from utils.util_module import fddnet
        a, s = Input((32, 32, 2)), Input((1,))
        self.model = Model([a, s], fddnet(a, s, 8, 'none', 0, 1e-6, 48))
        self.model.load_weights(str(weights))
        self.calls = 0

    def __call__(self, x, sigma):
        self.calls += 1
        packed = np.stack([x.real, x.imag], axis=-1)[None].astype('float32')
        out = self.model([packed, np.array([[sigma]], dtype='float32')],
                         training=False).numpy()[0]
        return out[..., 0] + 1j*out[..., 1]


def solve_red(op, y, x0, denoise, truth, lam=0.1, step=0.5, sigma=0.1,
              iterations=100, tolerance=1e-5, growth_limit=1e6):
    if (not np.isfinite([lam, step, sigma, tolerance, growth_limit]).all()
            or lam <= 0 or step <= 0 or sigma < 0 or iterations < 1
            or tolerance <= 0 or growth_limit <= 0):
        raise ValueError('Invalid RED parameters')
    x = np.array(x0, copy=True)
    b = y / np.sqrt(8)
    history, status = [], 'max_iterations'
    start = time.perf_counter()
    # Cache D(x) across iterations: K updates need K+1 calls, including final diagnostics.
    d = denoise(x, sigma)
    calls = 1
    for k in range(iterations + 1):
        data = op.forward(x)-b
        g = op.adjoint(data) + lam*(x-d)
        residual = float(np.linalg.norm(g))
        if not np.isfinite(x).all() or not np.isfinite(residual):
            status = 'unstable'
            history.append(dict(iteration=k, reason='nonfinite state or equilibrium residual',
                                elapsed_seconds=time.perf_counter()-start, denoiser_calls=calls))
            break
        record = dict(iteration=k, nmse_db=nmse(synthesis(x), truth),
                      data_residual=float(np.linalg.norm(data)),
                      data_relative=float(np.linalg.norm(data)/max(np.linalg.norm(b), 1e-30)),
                      equilibrium_residual=residual,
                      equilibrium_relative=float(residual/max(np.linalg.norm(x), 1e-30)),
                      elapsed_seconds=time.perf_counter()-start, denoiser_calls=calls)
        history.append(record)
        if np.linalg.norm(x) > growth_limit*max(np.linalg.norm(x0), 1):
            status = 'unstable'
            break
        if record['equilibrium_relative'] <= tolerance:
            status = 'converged'
            break
        if k == iterations:
            break
        new = x-step*g
        record['relative_update'] = float(np.linalg.norm(new-x)/max(np.linalg.norm(x), 1e-30))
        x = new
        d = denoise(x, sigma)
        calls += 1
    return dict(status=status, history=history, seconds=time.perf_counter()-start,
                denoiser_calls=calls), x


def solve_pnp(mask, y, h0, denoise, truth, iterations=10):
    h = np.array(h0, copy=True)
    history = []
    start = time.perf_counter()
    for k in range(iterations):
        rho, sigma = 1e-10*1.5**k, 0.1*0.8**k
        z = (mask*y+rho*h)/(mask+rho)
        h = synthesis(denoise(analysis(z), sigma))
        history.append(dict(iteration=k+1, nmse_db=nmse(h, truth),
                            pre_denoising_nmse_db=nmse(z, truth),
                            data_residual=float(np.linalg.norm(mask*h-y)/np.sqrt(8)),
                            elapsed_seconds=time.perf_counter()-start,
                            denoiser_calls=k+1))
    return dict(history=history, seconds=time.perf_counter()-start,
                denoiser_calls=iterations), h

"""Run from checkout: python -m deepred.experiment. No training or GPU needed."""
import os
import re
for key in ('TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(key, '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from deepred.ce import FFDNet, Operator, analysis, synthesis, solve_pnp, solve_red
from utils.util_pnp import get_dh_ce, interp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--output', default='results/deepred')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    seed = 2026
    # Exactly reproduce original noise generation; original solver actually uses mode=2.
    np.random.seed(seed)
    raw = np.load('data_npz/pppce_data.npz')['x_test']
    truth = raw[0,...,0]+1j*raw[0,...,1]
    noise = np.random.randn(256,32)+1j*np.random.randn(256,32)
    noise = np.sqrt(.01)*noise/np.linalg.norm(noise)
    mask, r, c = get_dh_ce(256, 2)
    noisy = truth+noise
    pilot = np.zeros((8,32,2))
    for j in range(32):
        for k in range(8):
            value = noisy[r[j*8+k], c[j*8+k]]
            pilot[k,j] = [value.real, value.imag]
    initial = interp(pilot, r, c, interp='rbf')
    y = mask*noisy
    op = Operator(mask)
    x0 = analysis(initial)
    common_initial = synthesis(x0)
    rng = np.random.default_rng(1)
    u = rng.normal(size=(32,32))+1j*rng.normal(size=(32,32))
    v = rng.normal(size=(256,32))+1j*rng.normal(size=(256,32))
    a, b = np.vdot(op.forward(u),v), np.vdot(u,op.adjoint(v))
    adjoint_error = float(abs(a-b)/max(abs(a),abs(b),1e-30))
    # Original program is executed unchanged, with seed injected before runpy.
    command = [sys.executable, '-c', "import numpy as np, runpy; np.random.seed(2026); runpy.run_path('pppce_demo.py', run_name='__main__')"]
    start = time.perf_counter()
    original = subprocess.run(command, capture_output=True, text=True)
    (out/'original_pnp.log').write_text(original.stdout+original.stderr)
    if original.returncode:
        raise RuntimeError('Original PnP failed; see original_pnp.log')
    original_seconds = time.perf_counter()-start
    denoise = FFDNet(root/'model_zoo/ffdnet.h5')
    denoise(x0,.1)  # warm-up excluded from algorithm budgets
    pnp, pnp_h = solve_pnp(mask,y,common_initial,denoise,truth)
    # Unprojected initial reproduces the original PnP, recording both output definitions.
    reproduction, _ = solve_pnp(mask,y,initial,denoise,truth)
    original_nmse = float(re.search(r'PPPCE_NMSE ([-+0-9.eE]+)dB', original.stdout).group(1))
    np.testing.assert_allclose(original_nmse,
        reproduction['history'][-1]['pre_denoising_nmse_db'], atol=1e-6, rtol=0)
    # Additional inference checks: direct call must match the original predict API.
    packed = np.stack([x0.real,x0.imag],axis=-1)[None].astype('float32')
    pred = denoise.model.predict([packed,np.array([[.1]],dtype='float32')],verbose=0)[0]
    direct = denoise(x0,.1)
    np.testing.assert_allclose(direct, pred[...,0]+1j*pred[...,1], atol=1e-6, rtol=0)
    np.testing.assert_array_equal(direct, denoise(x0,.1))
    runs = []
    arrays = dict(truth=truth, y=y, mask=mask, noise=noise, initial=common_initial, pnp=pnp_h)
    for lam in (.01,.1,1.):
        for step in (.25,.75):
            result, x = solve_red(op,y,x0,denoise,truth,lam=lam,step=step,
                                  sigma=.1,iterations=args.iterations)
            result.update(lam=lam, step=step, sigma=.1)
            runs.append(result)
            arrays[f'red_lambda_{lam}_step_{step}'] = synthesis(x)
            print(lam,step,result['status'],result['history'][-1],flush=True)
    extension, _ = solve_red(op,y,x0,denoise,truth,lam=.1,step=.75,
                              sigma=.1,iterations=1000)
    extension.update(lam=.1,step=.75,sigma=.1,
        rationale='Descriptive extension of best 100-iteration sweep case; not held-out selection')
    result = dict(extended_red=extension, original_nmse_db=original_nmse,
                  inference_checks='predict agreement and deterministic inference passed', seed=seed, iterations=args.iterations, pilot_mode=2, pilots=256,
                  nominal_snr_db=20, noise_model='fixed-energy normalized complex Gaussian draw',
                  data_term='0.5 * ||M S x - y||^2 / 8; lambda absorbs noise weighting',
                  dataset_sha256=hashlib.sha256((root/'data_npz/pppce_data.npz').read_bytes()).hexdigest(),
                  weights_sha256=hashlib.sha256((root/'model_zoo/ffdnet.h5').read_bytes()).hexdigest(),
                  adjoint_relative_error=adjoint_error, original_exit_code=original.returncode,
                  original_process_seconds=original_seconds, pnp=pnp,
                  original_reproduction=reproduction, red=runs,
                  warmup_calls=1, selection='No held-out set; sweep is descriptive, not test-set tuning evidence')
    (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'arrays.npz',**arrays)


if __name__ == '__main__':
    main()

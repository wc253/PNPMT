# Channel Estimation RED proof of concept

Original demos, utilities, weights and datasets remain unchanged. Run from the
repository root using the existing Python 3.10 / TensorFlow CPU environment:

```sh
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONDONTWRITEBYTECODE=1
/workspace/.venvs/pnpmt/bin/python -m unittest discover -s tests -v
/workspace/.venvs/pnpmt/bin/python -m deepred.experiment
```

The experiment overwrites only its output directory (default `results/deepred`).
Use `--output /tmp/pnpmt-red-reproduction` to retain committed measurements.
`--iterations` controls the six-case sweep (default 100); the descriptive
extension always permits 1000 updates. No training or GPU is used.

## Operator and solver convention

`analysis=T` is the original IFFT over frequency, FFT over antennas, then first
32 delay bins. `synthesis=S` zero-pads and applies the original inverse sequence.
For NumPy's default FFT normalization, S* = 8T and T S = I.
We use A=M S/sqrt(8), b=y/sqrt(8). Thus ||A|| <= 1 and

    g(x) = A*(A x - b) + lambda (x - D_sigma(x))
    x_next = x - step * g(x)

All optimization derivatives use the real inner product Re(vdot(u,v)); complex
arrays are a compact storage convention. FFDNet receives exactly the original
real/imaginary coefficients, without extra signal normalization, with
`training=False`. Sigma and lambda are fixed within each RED run. The data
term is ||M S x-y||^2/16, not a noise-whitened likelihood; lambda is tied to
this declared scaling and must not be copied unchanged to other conventions.

We solve an equilibrium equation. Without verified Jacobian symmetry and local
homogeneity, g is not asserted to be the gradient of the classical RED energy.
Small steps are a practical precaution, not a general convergence guarantee.
Stop when ||g||/max(||x||,1e-30) <= 1e-5; otherwise report the iteration cap or
numerical instability. No truth-dependent early stopping is used.

The initial D(x) call is cached: K updates plus final equilibrium diagnostics
cost K+1 calls. History at iteration 9 has ten calls, suitable for comparison
with ten-step PnP. Model loading, warmup and inference validation are outside
solver timings and budgets. Timing includes per-iteration diagnostics.

## Comparison boundaries

Seed 2026 reproduces the original fixed-energy noise generation at nominal
20 dB. The effective original mask is mode 2, regardless of CLI default 1;
this behavior is documented, not corrected in the original code.

The unchanged demo runs in a seeded subprocess and its log is retained. A
separate reproduction checks its final pre-denoising NMSE within 1e-6 dB.
Controlled PnP and RED share y, mask, noise and projected RBF initial S T h0;
both report full-channel denoiser-domain synthesis S x. PnP preserves its
original rho/sigma schedules; RED uses fixed sigma .1, deliberately a different
prior schedule. This is not an isolated solver-only ablation. PnP is not proven
to optimize the RED equilibrium. Only one bundled CE channel exists: results
are descriptive, with no held-out hyperparameter validation or superiority claim.

`metrics.json` contains every iteration's NMSE/residual/time/call count, dataset
and weight hashes, original reconstruction and controlled run. `arrays.npz`
retains observations, truth, common initial and estimates. Original subprocess
time includes startup/model loading and should not be compared to warm timings.
The extended RED run is a descriptive continuation study of the lowest-NMSE
100-update setting, not evidence from an independently selected configuration.

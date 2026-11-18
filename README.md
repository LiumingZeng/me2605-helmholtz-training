# ME2605 Final Project — Spectral Bias or Operator Conditioning in High-Frequency Helmholtz PINNs

**Course:** ME2605 Training Neural Networks, Fall 2026  
**Author:** Liuming Zeng

## Research question

High-wavenumber Helmholtz problems are a standard stress test for physics-informed neural
networks (PINNs). Two mechanisms are usually blamed: the spectral bias of smooth activations,
and the ill-conditioning of the indefinite Helmholtz operator. This project separates them with
two controlled regimes and a frequency/activation grid, then tests whether the diagnosis yields
a training recipe that cuts time-to-target by at least 10%.

## Baseline

The baseline is the official DeepXDE 2D Helmholtz Dirichlet example:

<https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Helmholtz_Dirichlet_2d.py>

- Domain `[0,1]^2`, Dirichlet `u=0`, hard boundary factor `x(1-x)y(1-y)`.
- Exact solution `u = sin(k0 x) sin(k0 y)`, `k0 = 2*pi*n`; source `f = k0^2 sin(k0 x) sin(k0 y)`.
- Network `[2,150,150,150,1]`, 10 collocation points per wavelength, 5000 Adam steps, lr `1e-3`.
- Official activation: `sin`; the standard PINN baseline replaces it with `tanh`.

A minimal PyTorch reproduction is in `code/scripts/baseline.py`.

## Pilot results (single seed; pilot hardware)

These are single-seed pilot numbers; final runs will be on the Hetao cluster and will be
re-measured with three seeds.

| Setting | Final rel. L2 | Time | Time to 1e-2 |
|---|---|---|---|
| n=2, tanh | 1.71 | 75.2 s | never |
| n=2, sin (official) | 5.60e-3 | 52.2 s | 35.9 s |
| n=4, tanh | 1.00 | 96.7 s | never |
| n=4, sin | 1.00 | 100.4 s | never |

The `tanh` PINN fails even at `n=2`; the official `sin` activation reaches `1e-2` in 35.9 s and
`5.6e-3` in 52.2 s. At `n=4` both baselines collapse to the zero output (the loss equals the
source-only loss).

## Planned experiments

- **Regimes:** frequency-scaled (`k0=2*pi*n`, `u=sin(k0 x) sin(k0 y)`) vs. fixed-solution
  (`u=sin(pi x) sin(3 pi y)`, vary `k0` independently).
- **Grid:** `n in {2,3,4,6,8}`; activation in `{tanh, sin, SIREN, tanh+Fourier}`;
  width `{150,256}`, depth `{3,4,6}`; 3 seeds.
- **Diagnostics:** empirical NTK spectrum/condition number, Hessian/Gauss-Newton eigenvalues,
  FFT error spectrum, per-layer gradient norms.
- **Interventions:** SIREN/Fourier features, curriculum on `n`, source normalization, RAD/RAR
  adaptive sampling, Helmholtz-symbol preconditioning, AdamW + warmup/cosine, L-BFGS fine-tune.
- **Targets:** at `n=2`, reach `1e-2` in <= 32.3 s (>=10% faster than 35.9 s); at `n=4`, reach
  `1e-2` where both baselines fail.

## Quick start

```powershell
# from code/
D:\venvs\me2605-pinn\Scripts\python.exe scripts\baseline.py --n 2 --activation sin
D:\venvs\me2605-pinn\Scripts\python.exe scripts\baseline.py --n 2 --activation tanh
D:\venvs\me2605-pinn\Scripts\python.exe scripts\baseline.py --n 4 --activation sin
```

The script prints per-100-step loss/relative L2 and a final JSON summary with
`time_to_1e-2_s` and `time_to_1e-3_s`.

## Repository structure

```
proposal/    Proposal (LaTeX + PDF)
code/        Baseline reproduction and experiments
```

## Status

Proposal + baseline reproduction. The frequency/activation grid and diagnostics will be added
during the project.

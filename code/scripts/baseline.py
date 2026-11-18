#!/usr/bin/env python
"""Minimal PyTorch reproduction of the DeepXDE 2D Helmholtz Dirichlet demo.

DeepXDE reference:
https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Helmholtz_Dirichlet_2d.py

Example:
    python scripts/baseline.py --n 2 --activation sin --steps 5000
    python scripts/baseline.py --n 4 --activation tanh --steps 5000
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn


class SinAct(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def make_net(width: int, depth: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = 2
    for _ in range(depth):
        layers.append(nn.Linear(prev, width))
        layers.append(SinAct() if activation == "sin" else nn.Tanh())
        prev = width
    layers.append(nn.Linear(prev, 1))
    net = nn.Sequential(*layers)
    for module in net:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)  # Glorot uniform
            nn.init.zeros_(module.bias)
    return net


class HardModel(nn.Module):
    """u = x(1-x)y(1-y) * N(x,y), so u = 0 on the boundary."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        bc = x[:, 0:1] * (1.0 - x[:, 0:1]) * x[:, 1:2] * (1.0 - x[:, 1:2])
        return bc * raw


def grad_scalar(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def pde_residual(u: torch.Tensor, x: torch.Tensor, k0: float) -> torch.Tensor:
    g = grad_scalar(u, x)
    u_x, u_y = g[:, 0:1], g[:, 1:2]
    u_xx = grad_scalar(u_x, x)[:, 0:1]
    u_yy = grad_scalar(u_y, x)[:, 1:2]
    f = k0**2 * torch.sin(k0 * x[:, 0:1]) * torch.sin(k0 * x[:, 1:2])
    return -(u_xx + u_yy) - k0**2 * u - f


def grid_points(n: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, n, device=device)
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def exact_solution(x: torch.Tensor, k0: float) -> torch.Tensor:
    return torch.sin(k0 * x[:, 0:1]) * torch.sin(k0 * x[:, 1:2])


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> float:
    return (
        torch.linalg.vector_norm(pred - true) / torch.linalg.vector_norm(true)
    ).item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2, help="Number of wavelengths.")
    parser.add_argument("--activation", choices=["sin", "tanh"], default="sin")
    parser.add_argument("--width", type=int, default=150)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
        torch.cuda.manual_seed_all(args.seed)
    else:
        device = torch.device("cpu")

    k0 = 2.0 * math.pi * args.n
    # DeepXDE: precision_train=10 and precision_test=30 points per wavelength.
    n_train = 10 * args.n
    n_test = 30 * args.n

    train_points = grid_points(n_train, device).requires_grad_(True)
    test_points = grid_points(n_test, device)
    test_exact = exact_solution(test_points, k0)

    net = make_net(args.width, args.depth, args.activation).to(device)
    model = HardModel(net).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict] = []
    time_to_1e2 = None
    time_to_1e3 = None
    start = time.perf_counter()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        u = model(train_points)
        residual = pde_residual(u, train_points, k0)
        loss = torch.mean(residual**2)
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            with torch.no_grad():
                u_pred = model(test_points)
                rel = relative_l2(u_pred, test_exact)
            elapsed = time.perf_counter() - start
            if time_to_1e2 is None and rel <= 1e-2:
                time_to_1e2 = elapsed
            if time_to_1e3 is None and rel <= 1e-3:
                time_to_1e3 = elapsed
            history.append(
                {"step": step, "loss": loss.item(), "rel_l2": rel, "time_s": elapsed}
            )
            print(
                f"step {step:5d}  loss {loss.item():.3e}  relL2 {rel:.3e}  "
                f"time {elapsed:.1f}s",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    total = time.perf_counter() - start
    result = {
        "n": args.n,
        "activation": args.activation,
        "width": args.width,
        "depth": args.depth,
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "final_rel_l2": history[-1]["rel_l2"],
        "total_time_s": total,
        "time_to_1e-2_s": time_to_1e2,
        "time_to_1e-3_s": time_to_1e3,
        "history": history,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The digits reference: a model snnTorch trains, and snnTorch's own spikes,
for the NIR bridge's parity test.

The one reference whose weights nobody chose.
`crates/neuralos-nir2json/tests/parity.rs` converts the model as a
stranger's `--sim-units` does, at snnTorch's 0.1 ms, holds each test digit's
pixels as a current, `p` as `p/255` in thousandths, and compares each
layer's spike trains and the class to snnTorch's own run of the trained
model on the same 8-bit pixels.

The digits are drawn here, from strokes: ten templates, two shapes for 1
(with or without a flag), 4 (closed or open) and 7 (without or with a bar),
anti-aliased, a random stroke width and a small wobble; each fitted to a
20 × 20 box keeping its shape and centered in 28 × 28 by its center of mass,
as MNIST's were; then deformed as Cireşan, Meier, Gambardella and
Schmidhuber deform MNIST's (arXiv 1003.0358, § 4): a rotation and a
horizontal shear, each by an angle from [−β, β], β 15° (7.5° for 1 and 7),
horizontal and vertical scaling from [1 − γ/100, 1 + γ/100] with γ from
[15, 20], and Simard, Steinkraus and Platt's elastic distortion
(ICDAR 2003, § 2) with σ from [5, 6] pixels and α from [36, 38]; then
quantized to 8 bits. Each epoch draws fresh digits. No dataset is downloaded
or committed: the reference rebuilds offline from this script.

The model: 784 → 32 → 10, two biased Linears (snnTorch exports each as an
`Affine`) and a `Leaky` after each, β and the threshold per neuron (the
exporter takes both per neuron or neither), β learned and clamped to
[0, 0.999] after each step, since the exporter reads the raw parameter
(`tau = dt/(1 − β)`), and the zero reset, what NIR's LIF means (the exporter
writes `v_reset = 0`). Adam trains it on each output neuron's spike count
over 25 steps. Then snnTorch's own run of the trained model: ten test
digits, one a class, each held for STEPS steps, every neuron's spike steps.

Byte for byte: one thread (with two, an operation of Adam's update split
between them gave its second half other last bits in some runs, and the
trained weights drifted from there), torch's deterministic algorithms, fixed
seeds, and `PYTHONHASHSEED=0`, which the exporter's edge order needs: the
script runs itself under it. A second run on the same machine writes both
files byte for byte, pinned by the fixture directory's SHA256SUMS.

Stack (of record): the repo's .nirenv, snntorch 1.0.0 · nir
1.0.9.dev1+g7883c3c85 · torch 2.13.0+cpu · numpy 2.5.2 (its random streams
are part of the reference). Run:

    .nirenv/bin/python3 tools/gen_snnTorch_digits.py <out dir>

Out, as committed: crates/neuralos-nir2json/tests/fixtures/parity/,
`digits.nir` and `digits.json`.
"""
import json
import os
import sys

import nir
import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from snntorch import utils
from snntorch.export_nir import export_to_nir

STEPS = 1500
HIDDEN = 32
TRAIN_STEPS = 25
EPOCHS = 6
PER_EPOCH = 6000
TEST = 1000  # our own test digits, for the accuracy digits.json gives
REFERENCE = 10  # the parity test's digits, one a class


def arc(cx, cy, rx, ry, a0, a1, n=18):
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def line(*pts):
    return np.array(pts, dtype=float)


def cat(*parts):
    return np.vstack(parts)


# each digit's shapes, each a list of strokes in a unit box, y down
SHAPES = {
    0: [[arc(0.5, 0.5, 0.26, 0.38, 0, 360, 28)]],
    1: [[line((0.52, 0.10), (0.52, 0.90))],
        [line((0.36, 0.26), (0.52, 0.10), (0.52, 0.90))]],
    2: [[cat(arc(0.5, 0.32, 0.24, 0.20, 180, 380), line((0.26, 0.90), (0.78, 0.90)))]],
    3: [[cat(arc(0.48, 0.30, 0.22, 0.18, 200, 450), arc(0.48, 0.68, 0.25, 0.20, -90, 160))]],
    4: [[line((0.62, 0.10), (0.20, 0.64), (0.82, 0.64)), line((0.64, 0.36), (0.64, 0.92))],
        [line((0.30, 0.10), (0.24, 0.60), (0.82, 0.60)), line((0.64, 0.12), (0.64, 0.92))]],
    5: [[cat(line((0.74, 0.12), (0.32, 0.12)), arc(0.47, 0.66, 0.25, 0.22, 230, 510))]],
    6: [[arc(0.66, 0.60, 0.38, 0.48, 270, 150), arc(0.50, 0.69, 0.20, 0.19, 0, 360, 24)]],
    7: [[line((0.22, 0.12), (0.78, 0.12), (0.42, 0.90))],
        [line((0.22, 0.12), (0.78, 0.12), (0.42, 0.90)), line((0.45, 0.52), (0.74, 0.52))]],
    8: [[arc(0.5, 0.29, 0.17, 0.16, 0, 360, 22), arc(0.5, 0.68, 0.22, 0.21, 0, 360, 24)]],
    9: [[arc(0.47, 0.33, 0.21, 0.20, 0, 360, 24), line((0.68, 0.33), (0.64, 0.90))]],
}
# each pixel's center, x then y, row by row
PIX = np.stack(
    np.meshgrid(np.arange(28) + 0.5, np.arange(28) + 0.5, indexing="xy"), -1
).reshape(-1, 2)


def ink(segs, width):
    """Strokes on the 28 x 28 grid, 0 to 1: each pixel covered by its
    distance to the nearest segment, a half pixel of anti-aliasing."""
    ax, ay = segs[:, 0, 0], segs[:, 0, 1]
    bx, by = segs[:, 1, 0] - ax, segs[:, 1, 1] - ay
    px, py = PIX[:, :1] - ax, PIX[:, 1:] - ay
    t = np.clip((px * bx + py * by) / np.maximum(bx * bx + by * by, 1e-9), 0, 1)
    dx, dy = px - t * bx, py - t * by
    d = np.sqrt((dx * dx + dy * dy).min(1))
    return np.clip(width / 2 + 0.5 - d, 0, 1).reshape(28, 28)


def draw(digit, rng):
    """One digit as MNIST's were made: one of its shapes, wobbled, fitted
    to a 20 x 20 box keeping its shape, then moved so its center of mass
    is the image's center; 0 to 1."""
    shapes = SHAPES[digit]
    strokes = [s + rng.normal(0, 0.008, s.shape) for s in shapes[rng.integers(len(shapes))]]
    width = rng.uniform(1.6, 3.2)
    pts = np.vstack(strokes)
    lo, hi = pts.min(0), pts.max(0)
    segs = np.concatenate([np.stack([p[:-1], p[1:]], 1) for p in strokes])
    segs = (segs - (lo + hi) / 2) * ((20 - width) / (hi - lo).max()) + 14
    img = ink(segs, width).reshape(-1)
    com = (PIX * img[:, None]).sum(0) / img.sum()
    return ink(segs + (14 - com), width)


def blur(sigma):
    """A Gaussian of `sigma` pixels as a matrix, zero outside the image."""
    i = np.arange(28)
    g = np.exp(-0.5 * ((i[:, None] - i[None]) / sigma) ** 2)
    return g / np.exp(-0.5 * (np.arange(-40, 41) / sigma) ** 2).sum()


def deform(img, digit, rng):
    """Ciresan et al.'s deformation of one image: a rotation and a
    horizontal shear by angles from [-beta, beta], each axis scaled from
    [1 - gamma/100, 1 + gamma/100], and Simard's elastic field, each output
    pixel read from the image there, bilinear, 0 outside; 8 bits."""
    beta = np.radians(7.5 if digit in (1, 7) else 15.0)
    rot, shear = rng.uniform(-beta, beta, 2)
    gamma = rng.uniform(15, 20) / 100
    sx, sy = rng.uniform(1 - gamma, 1 + gamma, 2)
    sigma, alpha = rng.uniform(5, 6), rng.uniform(36, 38)
    g = blur(sigma)
    dx = alpha * (g @ rng.uniform(-1, 1, (28, 28)) @ g.T)
    dy = alpha * (g @ rng.uniform(-1, 1, (28, 28)) @ g.T)
    m = (np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
         @ np.array([[1.0, np.tan(shear)], [0.0, 1.0]]) @ np.diag([1 / sx, 1 / sy]))
    yy, xx = np.mgrid[0:28, 0:28].astype(float)
    x = m[0, 0] * (xx - 13.5) + m[0, 1] * (yy - 13.5) + 13.5 + dx
    y = m[1, 0] * (xx - 13.5) + m[1, 1] * (yy - 13.5) + 13.5 + dy
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0

    def at(r, c):
        inside = (c >= 0) & (c < 28) & (r >= 0) & (r < 28)
        return np.where(inside, img[np.clip(r, 0, 27), np.clip(c, 0, 27)], 0.0)

    v = ((1 - fx) * (1 - fy) * at(y0, x0) + fx * (1 - fy) * at(y0, x0 + 1)
         + (1 - fx) * fy * at(y0 + 1, x0) + fx * fy * at(y0 + 1, x0 + 1))
    return np.round(np.clip(v, 0, 1) * 255).astype(np.int64)


def digits(n, rng):
    """`n` digits, the labels 0 to 9 in turn: 8-bit pixels, row by row."""
    labels = np.arange(n) % 10
    return np.stack([deform(draw(d, rng), d, rng).reshape(-1) for d in labels]), labels


def as_input(pixels):
    """8-bit pixels as the model reads them, `p/255`."""
    return torch.tensor(pixels / 255, dtype=torch.float32)


def leaky(n, output=False):
    return snn.Leaky(beta=torch.full((n,), 0.9), threshold=torch.full((n,), 1.0),
                     learn_beta=True, init_hidden=True, output=output, reset_mechanism="zero")


def counts(net, x, steps):
    """Each output neuron's spike count over `steps` steps of `x`."""
    utils.reset(net)
    total = 0
    for _ in range(steps):
        spk, _ = net(x)
        total = total + spk
    return total


def train():
    """The trained model, and how often the clamp moved a beta."""
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    net = nn.Sequential(nn.Linear(784, HIDDEN), leaky(HIDDEN), nn.Linear(HIDDEN, 10),
                        leaky(10, output=True))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    clamped = 0
    for epoch in range(EPOCHS):
        pixels, labels = digits(PER_EPOCH, np.random.default_rng([1, epoch]))
        x, y = as_input(pixels), torch.tensor(labels)
        perm = torch.randperm(len(x))
        for b in range(0, len(x), 100):
            idx = perm[b:b + 100]
            loss = F.cross_entropy(counts(net, x[idx], TRAIN_STEPS), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for mod in net:
                    if isinstance(mod, snn.Leaky):
                        clamped += int(((mod.beta < 0) | (mod.beta > 0.999)).sum())
                        mod.beta.clamp_(0.0, 0.999)
    return net, clamped


def spikes(net, x, steps):
    """snnTorch's own run: each digit of `x` held for `steps` steps; every
    neuron's spike steps, the first layer's then the second's, a list a
    digit."""
    utils.reset(net)
    fired = []
    with torch.no_grad():
        for _ in range(steps):
            k1 = net[1](net[0](x))
            k2, _ = net[3](net[2](k1))
            fired.append(torch.cat([k1, k2], 1).numpy().astype(bool))
    k = np.stack(fired, 1)  # digit, step, neuron
    return [[np.nonzero(k[i, :, j])[0].tolist() for j in range(k.shape[2])]
            for i in range(len(x))]


def main():
    # the exporter's edge list follows the hash seed: 0 makes it one order
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, *sys.argv],
                  {**os.environ, "PYTHONHASHSEED": "0"})
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    net, clamped = train()
    test, labels = digits(TEST, np.random.default_rng(2))
    with torch.no_grad():
        right = counts(net, as_input(test), TRAIN_STEPS).argmax(1).numpy() == labels
    utils.reset(net)
    nir.write(f"{out}/digits.nir", export_to_nir(net, torch.zeros(784)))
    trains = spikes(net, as_input(test[:REFERENCE]), STEPS)
    w1, b1 = net[0].weight.detach(), net[0].bias.detach()
    w2, b2 = net[2].weight.detach(), net[2].bias.detach()
    model = {"layers": [784, HIDDEN, 10], "train_steps": TRAIN_STEPS, "epochs": EPOCHS,
             "per_epoch": PER_EPOCH, "test_digits": TEST,
             "test_accuracy": float(right.mean()), "beta_clamped": clamped,
             "beta": [[float(b.min()), float(b.max())]
                      for b in (net[1].beta.detach(), net[3].beta.detach())],
             "max_drive": float((as_input(test) @ w1.T + b1).abs().max()),
             "max_w": [float(w1.abs().max()), float(w2.abs().max())],
             "max_b": [float(b1.abs().max()), float(b2.abs().max())]}
    configs = [{"name": f"digit_{i}", "nir": "digits.nir", "features": 784,
                "layers": [HIDDEN, 10], "label": int(labels[i]),
                "input": {"pixels": test[i].tolist()}, "spikes": trains[i]}
               for i in range(REFERENCE)]
    reference = {"generator": "tools/gen_snnTorch_digits.py", "snntorch": snn.__version__,
                 "torch": torch.__version__, "nir": nir.__version__, "numpy": np.__version__,
                 "dt_s": 1e-4, "steps": STEPS, "reset": "zero", "model": model,
                 "input": "per config: {\"pixels\": [...]}, one 8-bit pixel a feature, read "
                          "as p/255 and held at every step",
                 "configs": configs}
    # one digit a line: the pixels and the spike lists are data
    with open(f"{out}/digits.json", "w") as f:
        f.write("{\n")
        for key, value in reference.items():
            if key != "configs":
                f.write(f" {json.dumps(key)}: {json.dumps(value)},\n")
        f.write(' "configs": [\n')
        f.write(",\n".join(f"  {json.dumps(c)}" for c in configs))
        f.write("\n ]\n}\n")
    print(json.dumps(model))


if __name__ == "__main__":
    main()

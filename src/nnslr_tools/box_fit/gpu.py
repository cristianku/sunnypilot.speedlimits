"""Training of Fit box's rankers on the GPU (PyTorch, CUDA): the colour model's boosted trees and the edges' MLPs.

The same algorithms as :func:`nnslr_tools.box_fit.gbdt.train` and :func:`nnslr_tools.box_fit.edges.train_ranker`, in
float64, with the same random draws (numpy, the same seeds), so they give the same models up to the order of the sums;
what they return is what those return, read by the runtime with numpy alone (``colour.npz``, ``edges.npz``). Only the
training needs a V100: a machine without CUDA stops with :class:`NoCard`, never trains on the CPU.

- The trees grow a level at a time: the gradient sums of every (node, feature, bin) of a level in one scatter, the
  gain of every split of every node at once.
- The three MLPs of the edges train together, the cases of a mini-batch padded to one tensor.
"""
from __future__ import annotations

import numpy as np

from nnslr_tools.box_fit import gbdt


class NoCard(RuntimeError):
    """No CUDA card: the training of Fit box runs only on the GPU."""


def device(name: str = 'cuda'):
    import torch
    if not name.startswith('cuda'):
        raise NoCard(f'Fit box trains on the GPU only, not on {name!r}')
    if not torch.cuda.is_available():
        raise NoCard('no CUDA card is visible (CUDA_VISIBLE_DEVICES, nvidia-smi): Fit box trains on the GPU only')
    return torch.device(name)


def train_trees(X, y, *, n_trees=300, depth=6, lr=0.05, bins=32, min_leaf=20, lam=1.0, subsample=0.8, seed=0,
                dev='cuda') -> dict:
    """:func:`gbdt.train` on the GPU: the same arguments, the same random rows, the same trees (``gbdt.pack`` reads
    them)."""
    import torch
    dev = device(dev) if isinstance(dev, str) else dev
    rng = np.random.default_rng(seed)
    edges = gbdt._bin_edges(X, bins)
    nb = bins + 1
    N, nf = X.shape
    B = torch.from_numpy(gbdt._binned(X, edges).astype(np.int64)).to(dev)
    offsets = (torch.arange(nf, device=dev) * nb)[None, :]
    yt = torch.from_numpy(np.asarray(y, np.float64)).to(dev)
    base = float(y.mean())
    pred = torch.full((N,), base, dtype=torch.float64, device=dev)
    trees = []
    for _ in range(n_trees):
        sub = torch.from_numpy(rng.random(N) < subsample).to(dev)
        g = torch.where(sub, yt - pred, torch.zeros((), dtype=torch.float64, device=dev))
        w = sub.to(torch.float64)
        node = torch.zeros(N, dtype=torch.int64, device=dev)   # the node of each row, of the tree being grown
        nodes = []                                               # [feature, bin, left, right, value]; leaf: feature -1
        level = [0]
        nodes.append(None)
        for d in range(depth + 1):
            K = len(level)
            # The index of each row's node in this level (rows already in a leaf: -1).
            where = torch.full((len(nodes),), -1, dtype=torch.int64, device=dev)
            where[torch.tensor(level, device=dev)] = torch.arange(K, device=dev)
            k = where[node]
            live = k >= 0
            kl = k.clamp(min=0)
            G = torch.zeros(K, dtype=torch.float64, device=dev).index_add_(0, kl, g * live)
            n = torch.zeros(K, dtype=torch.float64, device=dev).index_add_(0, kl, w * live)
            values = (G / (n + lam)).tolist()
            ns = n.tolist()
            if d == depth:
                for i, me in enumerate(level):
                    nodes[me] = [-1, 0, -1, -1, values[i]]
                break
            # Gradient sums and counts per (node, feature, bin), cumulated over the bins: every split's gain at once.
            rows = torch.nonzero(live & sub).squeeze(1)
            flat = (kl[rows, None] * (nf * nb) + offsets + B[rows]).reshape(-1)
            cg = torch.zeros(K * nf * nb, dtype=torch.float64, device=dev).index_add_(
                0, flat, g[rows].repeat_interleave(nf)).view(K, nf, nb).cumsum(2)
            cn = torch.zeros(K * nf * nb, dtype=torch.float64, device=dev).index_add_(
                0, flat, torch.ones(len(flat), dtype=torch.float64, device=dev)).view(K, nf, nb).cumsum(2)
            Gk, nk = G[:, None, None], n[:, None, None]
            ok = (cn >= min_leaf) & (nk - cn >= min_leaf)
            gain = cg ** 2 / (cn + lam) + (Gk - cg) ** 2 / (nk - cn + lam) - Gk ** 2 / (nk + lam)
            gain = torch.where(ok, gain, torch.full_like(gain, -1.0)).view(K, nf * nb)
            best, at = gain.max(1)
            best, at = best.tolist(), at.tolist()
            nxt = []
            split_feature = torch.full((K,), -1, dtype=torch.int64, device=dev)
            split_bin = torch.zeros(K, dtype=torch.int64, device=dev)
            left_id = torch.zeros(K, dtype=torch.int64, device=dev)
            for i, me in enumerate(level):
                if ns[i] < 2 * min_leaf or not best[i] > 0:
                    nodes[me] = [-1, 0, -1, -1, values[i]]
                    continue
                j, t = divmod(at[i], nb)
                left, right = len(nodes), len(nodes) + 1
                nodes += [None, None]
                nodes[me] = [j, t, left, right, values[i]]
                nxt += [left, right]
                split_feature[i], split_bin[i], left_id[i] = j, t, left
            if not nxt:
                break
            # Every row of a split node goes to its left (bin <= t) or right child.
            f = split_feature[kl]
            moving = live & (f >= 0)
            go_left = B.gather(1, f.clamp(min=0)[:, None]).squeeze(1) <= split_bin[kl]
            child = left_id[kl] + (~go_left).to(torch.int64)
            node = torch.where(moving, child, node)
            level = nxt
        leaf_value = torch.tensor([v[4] for v in nodes], dtype=torch.float64, device=dev)
        pred += lr * leaf_value[node]
        # A split at bin t sends x < edges[t] to the left (gbdt.train's thresholds).
        trees.append(np.array([(j, (edges[j][t] if t < len(edges[j]) else np.inf) if j >= 0 else 0, a, b, v)
                               for j, t, a, b, v in nodes], float))
    return {'base': base, 'lr': lr, 'trees': trees}


def train_rankers(cases, seeds, *, hidden=32, epochs=30, lr=2e-3, wd=1e-4, tau=0.05, ncand=200, top=30, batch=16,
                  dev='cuda') -> list:
    """:func:`edges.train_ranker` for each of ``seeds``, all trained together on the GPU: the same initial weights,
    the same order of the cases and the same candidates drawn; [(mu, sd, [W1, b1, W2, b2])] in numpy."""
    import torch
    dev = device(dev) if isinstance(dev, str) else dev
    E = len(seeds)
    rngs = [np.random.default_rng(seed) for seed in seeds]
    X = np.vstack([F for F, _ in cases]).astype(np.float64)
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    mu[0], sd[0] = 0, 1
    d = X.shape[1]
    init = [[r.normal(0, 1 / np.sqrt(d), (d, hidden)), np.zeros(hidden), r.normal(0, 1 / np.sqrt(hidden), hidden),
             np.zeros(1)] for r in rngs]
    ps = [torch.tensor(np.stack([p[i] for p in init]), dtype=torch.float64, device=dev) for i in range(4)]
    m = [torch.zeros_like(p) for p in ps]
    v = [torch.zeros_like(p) for p in ps]
    # Every case once on the card, padded to the most candidates; a case cut to ``ncand`` takes the rows drawn.
    sizes = [len(y) for _, y in cases]
    C = max(sizes)
    Z = torch.zeros((len(cases), C, d), dtype=torch.float64)
    Y = torch.full((len(cases), C), -np.inf, dtype=torch.float64)
    for i, (F, y) in enumerate(cases):
        Z[i, :len(y)] = torch.from_numpy((F.astype(np.float64) - mu) / sd)
        Y[i, :len(y)] = torch.from_numpy(y.astype(np.float64))
    Z, Y = Z.to(dev), Y.to(dev)
    order_by = [np.argsort(-y)[:top] for _, y in cases]
    width = max(min(s, ncand + top) for s in sizes)
    t = 0
    for _ in range(epochs):
        orders = [r.permutation(len(cases)) for r in rngs]
        for start in range(0, len(cases), batch):
            # The rows of each case of the batch, drawn with each MLP's own generator in train_ranker's order.
            picks = np.zeros((E, batch, width), np.int64)
            valid = np.zeros((E, batch, width), bool)
            ids = np.zeros((E, batch), np.int64)
            used = np.zeros((E, batch), bool)
            for e, r in enumerate(rngs):
                for b, ci in enumerate(orders[e][start:start + batch]):
                    n = sizes[ci]
                    idx = np.unique(np.r_[order_by[ci], r.choice(n, ncand, replace=False)]) if n > ncand \
                        else np.arange(n)
                    picks[e, b, :len(idx)] = idx
                    valid[e, b, :len(idx)] = True
                    ids[e, b] = ci
                    used[e, b] = True
            nb = int(used[0].sum())
            pk = torch.from_numpy(picks).to(dev)
            vd = torch.from_numpy(valid).to(dev) & torch.from_numpy(used).to(dev)[:, :, None]
            ci = torch.from_numpy(ids).to(dev)
            z = Z[ci[:, :, None], pk]                                  # E x batch x width x d
            y = Y[ci[:, :, None], pk].masked_fill(~vd, -np.inf)
            w = [p.detach().requires_grad_(True) for p in ps]
            a = torch.einsum('ebwd,edh->ebwh', z, w[0]) + w[1][:, None, None, :]
            o = torch.einsum('ebwh,eh->ebw', torch.relu(a), w[2]) + w[3][:, None, None, 0]
            # A finite floor, not -inf: a padded slot of the batch (no case) stays without NaN, and weighs exactly 0.
            logp = torch.log_softmax(o.masked_fill(~vd, -1e30), 2)
            target = torch.softmax((y - y.amax(2, keepdim=True).clamp(min=-1e300)) / tau, 2).nan_to_num(0.0)
            loss = -(target * logp.masked_fill(~vd, 0.0)).sum() / nb
            grads = torch.autograd.grad(loss, w)
            t += 1
            with torch.no_grad():
                for i in range(4):
                    g = grads[i] + (wd * ps[i] if i in (0, 2) else 0)
                    m[i] = .9 * m[i] + .1 * g
                    v[i] = .999 * v[i] + .001 * g ** 2
                    ps[i] = ps[i] - lr * (m[i] / (1 - .9 ** t)) / (torch.sqrt(v[i] / (1 - .999 ** t)) + 1e-8)
    out = [p.cpu().numpy() for p in ps]
    return [(mu, sd, [out[0][e], out[1][e], out[2][e], out[3][e]]) for e in range(E)]

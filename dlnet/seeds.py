"""Seed selection: which candidate architectures to train first (or at all) for THIS dataset.

Three independent rankers, aggregated by a weighted Borda count:

1. `prior`   data card (dlnet/datacard.py) + task context -> weights on architecture TRAITS; the
             traits of a candidate are read from the modules in its spec (OP_TRAITS). The rules
             are DLNet heuristics written down with their reason (RULES); they are not taken from
             a paper and are only accepted to the extent scripts/bench_seeds.py shows they help.
2. `meta`    past runs (dlnet/meta.py): the k nearest datasets vote on the family order, and the
             greedy portfolio of Auto-sklearn 2.0 (Feurer et al., JMLR 2022, arXiv:2007.04074,
             Algorithm 1) is built on those neighbours.
2b. `zap`    zero-shot surrogate meta-learned with a pairwise ranking loss (ZAP, arXiv:2206.08476;
             dlnet/surrogate.py) on per-candidate results of past runs.
2c. `ai`     optional AI advisor (dlnet/advisor.py), e.g. Claude Code / Codex CLI.
3. `proxies` zero-cost proxies on the real data, combined by the majority vote of Abdelfattah et
             al. (arXiv:2101.08134, Sec. 4: "majority vote between synflow, jacob_cov and snip";
             DLNet uses NASWOT, the later form of jacob_cov). Off by default: agent.md 16.1 item 4
             forbids pruning by proxies before their rank correlation was measured on DLNet data.

Borda count: every ranker gives n - rank points to each candidate (ties share points); the
weighted sum orders the candidates. `select` keeps the top `n` with a diversity cap per family
and always keeps one plain ResNet as an anchor, so a wrong prior cannot remove every safe choice.
"""

from __future__ import annotations

import math
import re

import numpy as np

# Traits of the modules (leaf ops and decoders). A candidate has a trait if any of its modules has it.
OP_TRAITS = {
    "transformer_block": {"global_attention"}, "attn_seq": {"global_attention"},
    "self_attn": {"global_attention"}, "patchify": {"global_attention", "position_aware"},
    "mixer_block": {"global_attention", "position_aware"}, "window_attn": {"local_attention"},
    "pos": {"position_aware"}, "flatten": {"position_aware"}, "pool_flatten": {"position_aware"},
    "conv_bn_act": {"local_conv"}, "basic_block": {"local_conv"}, "bottleneck": {"local_conv"},
    "preact_block": {"local_conv"}, "vgg_stage": {"local_conv"}, "lenet_stage": {"local_conv", "position_aware"},
    "dense_block": {"local_conv"}, "convnext_block": {"local_conv"}, "regnet_block": {"local_conv"},
    "repvgg_block": {"local_conv", "light"}, "segres_block": {"local_conv", "group_norm"},
    "unet_res_block": {"local_conv", "group_norm"}, "poolformer_block": {"local_conv"},
    "dw_sep": {"local_conv", "light"}, "mbconv": {"local_conv", "light"},
    "fused_mbconv": {"local_conv", "light"}, "ghost_bottleneck": {"local_conv", "light"},
    "shuffle_unit": {"local_conv", "light"}, "fire": {"local_conv", "light"},
    "convmixer_block": {"local_conv", "large_kernel"},
    "inception": {"local_conv", "multi_scale"}, "res2net_block": {"local_conv", "multi_scale"},
    "sk_conv": {"local_conv", "multi_scale", "channel_attention"},
    "split_attn": {"local_conv", "channel_attention"},
    "se": {"channel_attention"}, "eca": {"channel_attention"}, "cbam": {"channel_attention"},
    "coord_attn": {"channel_attention", "position_aware"}, "gc_block": {"channel_attention"},
    "scse": {"channel_attention"}, "grn": {"channel_attention"},
    # segmentation decoders
    "unet": {"fine_detail"}, "unetpp": {"fine_detail"}, "linknet": {"fine_detail", "light"},
    "fpn": {"fine_detail", "multi_scale"}, "deconv": set(), "fcn": {"fine_detail"},
    "deeplabv3": {"context", "multi_scale"}, "deeplabv3plus": {"context", "multi_scale", "fine_detail"},
    "pspnet": {"context"}, "upernet": {"context", "multi_scale"}, "segformer": {"context", "light"},
}  # fmt: skip
TRAITS = sorted({t for v in OP_TRAITS.values() for t in v})

# Task context: keywords (English and Vietnamese) -> trait weights.
CONTEXT = {
    r"medical|y tế|mri|ct\b|x-?ray|histolog|patholog|lesion|tumou?r|khối u": {"fine_detail": 1.0, "group_norm": 1.0},
    r"satellite|aerial|remote sens|vệ tinh|viễn thám|drone": {"multi_scale": 1.0, "context": 0.5},
    r"edge|mobile|embedded|real-?time|fast|nhanh|nhẹ|thiết bị|camera": {"light": 1.5},
    r"texture|material|surface|defect|fabric|kết cấu|bề mặt|lỗi": {"channel_attention": 1.0, "multi_scale": 0.5},
    r"document|text|character|digit|handwrit|chữ|văn bản": {"position_aware": 1.0},
    r"small object|tiny|vật nhỏ|chi tiết": {"fine_detail": 1.0, "multi_scale": 0.5},
    r"natural|photo|scene|ảnh tự nhiên|cảnh": {"local_conv": 0.5, "channel_attention": 0.5},
}  # fmt: skip


def traits(cfg: dict) -> set[str]:
    from .zoo import leaf_ops

    spec = cfg.get("spec")
    ops = leaf_ops(spec) if spec else []
    return set().union(*(OP_TRAITS.get(o, set()) for o in ops)) if ops else set()


def trait_weights(card: dict, context: str = "", goal: str = "accuracy") -> tuple[dict, list[str]]:
    """Weights per trait from the data card and the context, with the reason for each rule fired."""
    w = dict.fromkeys(TRAITS, 0.0)
    why = []

    def add(trait, value, reason):
        w[trait] += value
        why.append(f"{'+' if value > 0 else ''}{value:g} {trait}: {reason}")

    n, spc = card.get("n_train", 0), card.get("samples_per_class_min", 10**9)
    if n < 5000 or spc < 300:
        add(
            "global_attention",
            -1.0,
            f"{n} training images ({spc} per class at least): global attention needs much data",
        )
    gain = card.get("conv_bias_gain")
    if gain is not None and gain > 0.1:
        add(
            "local_conv",
            1.0,
            f"random-conv probe beats pixel probe by {gain:+.2f}: translation-invariant local features help",
        )
        add("position_aware", -0.5, "same probe: absolute position matters little")
    pos = card.get("position_dependence")
    if pos is not None and pos > 0.2 and (gain is None or gain < 0.05):
        add(
            "position_aware",
            0.5,
            f"position dependence {pos:.2f}: classes differ by where things are",
        )
    hf, slope = card.get("high_freq_energy", 0.0), card.get("spectral_slope", 2.0)
    if hf > 0.4 or slope < 1.0:
        add(
            "channel_attention",
            0.5,
            f"high-frequency energy {hf:.2f}, spectral slope {slope:.2f}: texture-like data",
        )
        add("multi_scale", 0.5, "texture-like data: several receptive fields")
    rel = card.get("corr_length_rel")
    if rel is not None and rel < 0.1:
        add("fine_detail", 1.0, f"correlation length {rel:.2f} of the side: small structures")
        add("multi_scale", 0.25, "small structures")
    if rel is not None and rel > 0.25:
        add(
            "context",
            1.0,
            f"correlation length {rel:.2f} of the side: large structures need context",
        )
    cc = card.get("channel_correlation")
    if cc is not None and cc < 0.5:
        add("channel_attention", 0.5, f"channel correlation {cc:.2f}: colour carries information")
    for pattern, tw in CONTEXT.items():
        if context and re.search(pattern, context.lower()):
            for t, v in tw.items():
                add(t, v, f"context matches '{pattern.split('|')[0]}'")
    if goal == "small":
        add("light", 1.5, "goal: small / fast model")
    elif goal == "balanced":
        add("light", 0.75, "goal: balance accuracy and size")
    return w, why


def prior_scores(
    cands: list[dict], card: dict, context="", goal="accuracy"
) -> tuple[list[float], list[str]]:
    """Trait score of every candidate, minus a size penalty when params exceed ~100 per training image."""
    w, why = trait_weights(card, context, goal)
    n = max(card.get("n_train", 1), 1)
    out = []
    for c in cands:
        s = sum(w[t] for t in traits(c))
        p = c.get("params") or 0
        if p > 100 * n:  # DLNet heuristic: over-parameterised for the data size
            s -= 0.5 * math.log10(p / (100 * n))
        if goal in ("small", "balanced") and p:
            s -= (0.5 if goal == "small" else 0.25) * math.log10(max(p, 1))
        out.append(s)
    return out, why


def borda(score_lists: list[tuple[list[float], float]]) -> np.ndarray:
    """Weighted Borda count of several score lists (higher score = better; ties share points)."""
    total = None
    for scores, weight in score_lists:
        if weight == 0:
            continue
        a = np.asarray(scores, float)
        a = np.where(np.isfinite(a), a, -np.inf)
        order = np.argsort(a, kind="stable")
        pts = np.empty(len(a))
        pts[order] = np.arange(len(a), dtype=float)
        for v in np.unique(a):  # average points of ties
            m = a == v
            pts[m] = pts[m].mean()
        total = weight * pts if total is None else total + weight * pts
    return np.zeros(len(score_lists[0][0])) if total is None else total


def proxy_vote(scores: list[dict], names=("synflow", "naswot", "snip")) -> list[float]:
    """Majority vote of zero-cost proxies (Abdelfattah et al., arXiv:2101.08134, Sec. 4).

    The paper says only "majority vote between the three metrics"; DLNet reads it as pairwise:
    A beats B when most of the proxies available for both rank A higher, and a candidate's score
    is its number of pairwise wins (Copeland). Not checked against the authors' code (the
    repository was not reachable).
    """
    n = len(scores)
    wins = [0.0] * n
    for i in range(n):
        for j in range(i + 1, n):
            votes = 0
            for k in names:
                a, b = scores[i].get(k), scores[j].get(k)
                if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)) or a == b:
                    continue
                votes += 1 if a > b else -1
            if votes > 0:
                wins[i] += 1
            elif votes < 0:
                wins[j] += 1
            else:
                wins[i] += 0.5
                wins[j] += 0.5
    return wins


def meta_scores(
    cands: list[dict], rec: dict | None, portfolio: list[str] | None
) -> list[float] | None:
    """Higher for families earlier in the meta order; families of the portfolio get a bonus."""
    if not rec and not portfolio:
        return None
    order = rec["family_order"] if rec else []
    pos = {f: i for i, f in enumerate(order)}
    port = set(portfolio or [])
    return [-pos.get(c["family"], len(pos)) + (len(pos) + 1) * (c["family"] in port) for c in cands]


def _anchor(c: dict) -> bool:
    return c.get("family") in ("resnet", "preset:resnet") or c.get("origin") == "preset:resnet"


def select(cands, card, n=0, context="", goal="accuracy", meta=None, portfolio=None,
           proxy_scores=None, weights=(1.0, 1.0, 0.0), max_per_family=2, order="data", seed=0,
           surrogate_scores=None, surrogate_weight=1.0, advice=None, advice_weight=1.0):  # fmt: skip
    """Order the candidates by the aggregated rank; with n > 0 keep only n of them.

    Returns (candidates, report). `weights` = (prior, meta, proxies); proxies are off by default.
    """
    prior, why = prior_scores(cands, card, context, goal)
    lists = [(prior, weights[0])]
    ms = meta_scores(cands, meta, portfolio)
    if ms is not None:
        lists.append((ms, weights[1]))
    if proxy_scores is not None and weights[2]:
        lists.append((proxy_vote(proxy_scores), weights[2]))
    if surrogate_scores is not None and surrogate_weight:
        lists.append((surrogate_scores, surrogate_weight))
    if advice is not None and advice_weight:  # AI advisor (dlnet/advisor.py)
        lists.append((advice, advice_weight))
    total = borda(lists)
    if order == "random":  # control arm for scripts/bench_seeds.py: same n, no information
        total = np.random.default_rng(seed).permutation(len(cands)).astype(float)
    order = sorted(range(len(cands)), key=lambda i: -total[i])
    if n <= 0 or n >= len(cands):
        chosen = order
    else:
        chosen, per = [], {}
        anchor = next((i for i in order if _anchor(cands[i])), None) if weights[0] else None
        if anchor is not None:
            chosen.append(anchor)
            per[cands[anchor]["family"]] = 1
        for i in order:
            if len(chosen) >= n:
                break
            f = cands[i]["family"]
            if i in chosen or per.get(f, 0) >= max_per_family:
                continue
            chosen.append(i)
            per[f] = per.get(f, 0) + 1
        for i in order:  # the family cap left room: fill by rank
            if len(chosen) >= n:
                break
            if i not in chosen:
                chosen.append(i)
    out = [cands[i] for i in chosen]
    for i, c in enumerate(cands):
        c["seed_prior"] = round(float(prior[i]), 3)
    for rank, i in enumerate(chosen):
        cands[i]["seed_rank"] = rank + 1
        cands[i]["seed_score"] = round(float(total[i]), 2)
        cands[i]["seed_traits"] = sorted(traits(cands[i]))
    report = {
        "rules_fired": why,
        "rankers": ["prior"] + (["meta"] if ms is not None else []) + (
            ["proxy_vote"] if proxy_scores is not None and weights[2] else []) + (
            ["zap_surrogate"] if surrogate_scores is not None and surrogate_weight else []) + (
            ["ai_advisor"] if advice is not None and advice_weight else []),
        "weights": dict(zip(("prior", "meta", "proxies"), weights, strict=True)),
        "kept": len(out), "pool": len(cands),
        "top": [{"family": c["family"], "origin": c.get("origin", "family"), "traits": c["seed_traits"],
                 "score": c["seed_score"]} for c in out[:10]],
        "used_for_pruning": bool(n and n < len(cands)),
    }  # fmt: skip
    return out, report

"""Thesis figures for App. F from torchstrap's examples/spirals.

Reuses the example's data, model, plotting helpers and training loop, and trains the
three configurations of Fig. F.1 on the same 100 points:
  single    -- one replica, every point every pass
  ensemble  -- 100 replicas, different initializations, every point every pass
  bootstrap -- 100 replicas, one fixed with-replacement resample each (the example itself,
               with out-of-bag early stopping and checkpointing)

Run from the torchstrap checkout:
  uv run python <thesis>/figures/torchstrap/make_spirals_figures.py --torchstrap .
"""

import argparse
import sys
import tempfile
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from torch.func import vmap, grad_and_value
from torch.nn.functional import binary_cross_entropy_with_logits

from torchstrap import viz
from torchstrap.callbacks import Checkpoint, EarlyStopping, LRScheduler
from torchstrap.history import History
from torchstrap.metrics import MetricCollection, LogLoss, Accuracy, AUROC
from torchstrap.optimizer import Adam
from torchstrap.stateless import StatelessModule
from torchstrap.utils.data import BootstrapSampler

HERE = Path(__file__).resolve().parent


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--torchstrap", type=Path, required=True, help="torchstrap checkout")
    ap.add_argument("--outdir", type=Path, default=HERE)
    return ap.parse_args()


def train(ex, mode, points, labels, device, outdir, num_replicas=100,
          batch_size=32, num_epochs=20, passes_per_epoch=5):
    num_samples = points.shape[0]
    n = 1 if mode == "single" else num_replicas
    torch.manual_seed(1)
    ensemble, state = StatelessModule.init(
        ex.make_classifier_module, Adam, 2, 512, 512, 1,
        num_replicas=n, device=device, init_randomness="different",
    )

    def loss_fn(params, buffers, x, y):
        return binary_cross_entropy_with_logits(ensemble(params, buffers, x), y)

    grad_loss = vmap(grad_and_value(loss_fn, argnums=0), in_dims=(0, 0, 0, 0))
    forward = vmap(ensemble, in_dims=(0, 0, None))
    sched = LRScheduler("CosineAnnealingLR", T_max=num_epochs)
    history = History()

    if mode == "bootstrap":
        # exactly the example: fixed resample per replica, OOB-monitored callbacks
        sampler = BootstrapSampler(num_samples, num_replicas=n, batch_size=batch_size, device=device)
        oob = sampler.oob_mask()
        metrics = MetricCollection({"oob_loss": LogLoss(ensemble=True),
                                    "oob_acc": Accuracy(ensemble=True),
                                    "oob_auroc": AUROC(ensemble=True)})
        ckpt = Checkpoint(root_dir=Path(tempfile.mkdtemp()), verbose=False)  # scratch, not the thesis tree
        early = EarlyStopping(patience=8, threshold=1e-3, verbose=False)
        batches = lambda: iter(sampler)
    else:
        # every replica sees every point; one shared shuffle per pass
        def batches():
            perm = torch.randperm(num_samples, device=device).expand(n, -1)
            return (perm[:, i:i + batch_size] for i in range(0, num_samples, batch_size))

    for epoch in range(num_epochs):
        history.new_epoch()
        losses = []
        for _ in range(passes_per_epoch):
            for idx in batches():
                X, Y = points[idx], labels[idx].float().unsqueeze(-1)
                grads, loss = grad_loss(state.params_dict, state.buffers_dict, X, Y)
                Adam.apply_gradient(state, grads)
                losses.append(loss.detach())
        train_loss = torch.stack(losses).mean(0)
        if mode == "bootstrap":
            with torch.inference_mode():
                logits = forward(state.params_dict, state.buffers_dict, points)
            metrics.reset()
            metrics.update(logits, labels, mask=oob)
            m = metrics.compute()
            sched(state, m["oob_loss"])
            ckpt(state, m["oob_loss"])
            history.append_epoch(train_loss=train_loss, **m)
            history.record_state(state)
            history.flush_epoch()
            if early(state, m["oob_loss"]):
                break
        else:
            sched(state, train_loss)
            history.append_epoch(train_loss=train_loss)
            history.record_state(state)
            history.flush_epoch()

    if mode == "bootstrap":
        early.restore_best(state)
        ckpt.load_best(state)
    return ex.predict_on_mesh(ensemble, state), history


def main():
    args = parse_args()
    sys.path.insert(0, str(args.torchstrap / "examples" / "spirals"))
    import spirals_parallel as ex  # the example's helpers; its __main__ does not run

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)  # same dataset as the example
    points, labels = ex.make_spirals(100, noise_std=0.05)
    points, labels = points.to(device), labels.to(device)

    titles = {"single": "Single model",
              "ensemble": "Ensemble, same data (100)",
              "bootstrap": "Bootstrap ensemble (100)"}
    results = {m: train(ex, m, points, labels, device, args.outdir) for m in titles}

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    for ax, (mode, title) in zip(axes, titles.items()):
        (xx, yy, z), _ = results[mode]
        im = ex.plot_predictions(ax, xx, yy, z)
        sc = ex.plot_spirals(ax, points.cpu(), labels.cpu())
        ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), xlabel="x", ylabel="y", title=title)
    axes[-1].legend(sc.legend_elements()[0], ["0", "1"], title="label",
                    title_fontproperties=FontProperties(weight="bold"), loc="lower right")
    fig.colorbar(im, ax=axes, label="mean prediction", fraction=0.025, pad=0.01)
    fig.savefig(args.outdir / "spirals_predictions.pdf")

    history = results["bootstrap"][1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    viz.plot_curves(history, ["train_loss", "oob_loss"], ax=axes[0])
    viz.plot_timeline(history, "oob_loss", ax=axes[1])
    fig.savefig(args.outdir / "spirals_training.pdf")
    history.to_file(args.outdir / "spirals_history.json")
    print("wrote", args.outdir / "spirals_predictions.pdf", "and", args.outdir / "spirals_training.pdf")


if __name__ == "__main__":
    main()

# callbacks.py
import os
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from odnn.helper import predict_and_plot, plot_learned_layers_phase, plot_learned_layers_phase_and_gates

def _get_val_loader(trainer, explicit=None):
    if explicit is not None:
        return explicit
    # fall back to datamodule if provided
    dm = getattr(trainer, "datamodule", None)
    return dm.val_dataloader() if dm is not None else None

class ZWarmupCallback(pl.callbacks.Callback):
    def __init__(self, warmup_epochs=100):
        super().__init__()
        self.warmup_epochs = warmup_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        freeze = trainer.current_epoch < self.warmup_epochs
        for layer in getattr(pl_module, "propalayerlist", []):
            if hasattr(layer, "trainable_z") and layer.trainable_z:
                layer.z_raw.requires_grad = not freeze

class LossLoggingCallback(pl.callbacks.Callback):
    def __init__(self, folder_path, plot_each_n_epochs=100):
        super().__init__()
        self.folder_path = folder_path
        self.plot_each_n_epochs = plot_each_n_epochs
        self.history = {"epoch": [], "train_loss": [], "val_loss": []}

    def on_validation_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss  = trainer.callback_metrics.get("val_loss")
        # convert to float if tensors
        tf = float(train_loss) if train_loss is not None else None
        vf = float(val_loss)  if val_loss  is not None else None

        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(tf)
        self.history["val_loss"].append(vf)

        if epoch == 0 or (epoch % self.plot_each_n_epochs) != 0:
            return

        print(f"Epoch {epoch}: train_loss={tf:.6e}, val_loss={vf:.6e}")

        plt.figure(figsize=(6,4))
        plt.plot(self.history["epoch"], self.history["train_loss"], label="train")
        plt.plot(self.history["epoch"], self.history["val_loss"],  label="val")
        plt.yscale("log")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss")
        plt.legend()
        fname = os.path.join(self.folder_path, "loss_curve.png")
        plt.savefig(fname); plt.close()
        print(f"[LossLoggingCallback] Saved loss curve to {fname}")


class PlottingCallback(pl.callbacks.Callback):
    def __init__(self, val_loader=None, plot_each_n_epochs=10, plot_each_n_batches=100, folder_path=None):
        super().__init__()
        self._val_loader = val_loader  # optional; else we’ll pull from datamodule
        self.plot_each_n_epochs = plot_each_n_epochs
        self.plot_each_n_batches = plot_each_n_batches
        self.folder_path = folder_path  # optional; else use trainer.logger.log_dir

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        i_epoch = trainer.current_epoch
        if (batch_idx % self.plot_each_n_batches != 0) or (i_epoch % self.plot_each_n_epochs != 0):
            return

        val_loader = _get_val_loader(trainer, self._val_loader)
        if val_loader is None:
            return  # nothing to plot against

        base = self.folder_path or getattr(trainer.logger, "log_dir", trainer.default_root_dir)

        predict_and_plot(
            pl_module, val_loader, device=pl_module.device, plot_idx=0, plot="phase",
            savename=os.path.join(base, f"phase_e{i_epoch:03d}_b{batch_idx:05d}.png"),
            show=False,
        )
        predict_and_plot(
            pl_module, val_loader, device=pl_module.device, plot_idx=0, plot="intensity",
            savename=os.path.join(base, f"intensity_e{i_epoch:03d}_b{batch_idx:05d}.png"),
            show=False,
        )
        predict_and_plot(
            pl_module, val_loader, device=pl_module.device, plot_idx=0, plot="field",
            savename=os.path.join(base, f"field_e{i_epoch:03d}_b{batch_idx:05d}.png"),
            show=False,
        )
        plot_learned_layers_phase(pl_module, filename=os.path.join(base, f"layers_e{i_epoch:03d}.png"), show=False)
        plot_learned_layers_phase_and_gates(pl_module, filename=os.path.join(base, f"layers_e{i_epoch:03d}.png"), show=False)

class PhaseTrackingCallback(pl.callbacks.Callback):
    def __init__(self, val_loader=None, plot_each_n_epochs=10,
                 save_path="phase_evolution.mp4", folder_path=None):
        super().__init__()
        self._val_loader = val_loader
        self.plot_each_n = plot_each_n_epochs
        self.save_path = save_path
        self.folder_path = folder_path

        self.input_phase = None
        self.expected_output = None
        self.phase_history = None  # allocated in on_fit_start
        self.has_gates = False     # set in on_fit_start

    def _normalize(self, phase):
        return (phase + np.pi) % (2 * np.pi) - np.pi

    def on_fit_start(self, trainer, pl_module):
        # allocate history now that we know N layers
        phaselayers = getattr(pl_module, "phaselayerlist", [])
        n_layers = len(phaselayers)

        # detect whether any layer exposes a gate (LearnablePhaseGateLayer2d)
        self.has_gates = any(hasattr(layer, "gate_hard") for layer in phaselayers)

        self.phase_history = {
            "output_phase": [],
            "layer_phases": [[] for _ in range(n_layers)],
        }
        if self.has_gates:
            self.phase_history["gate_masks"] = [[] for _ in range(n_layers)]

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.plot_each_n != 0:
            return

        val_loader = _get_val_loader(trainer, self._val_loader)
        if val_loader is None:
            return

        x_batch, y_batch = next(iter(val_loader))
        x_batch = x_batch.to(pl_module.device)

        with torch.no_grad():
            y_pred = pl_module(x_batch)

        # output phase at detector
        out_phase = torch.angle(y_pred[:, 0] + 1j * y_pred[:, 1]).cpu().numpy()
        self.phase_history["output_phase"].append(self._normalize(out_phase.copy()))

        # cache input + expected output phase once
        if self.input_phase is None:
            self.input_phase = torch.angle(
                x_batch[:, 0] + 1j * x_batch[:, 1]
            ).cpu().numpy()
        if self.expected_output is None:
            self.expected_output = torch.angle(
                y_batch[:, 0] + 1j * y_batch[:, 1]
            ).cpu().numpy()

        # per-layer phase (and optionally gate) snapshots
        for i, layer in enumerate(pl_module.phaselayerlist):
            ph = layer.phase_weights.detach().cpu().numpy().copy()
            self.phase_history["layer_phases"][i].append(self._normalize(ph))

            if self.has_gates and hasattr(layer, "gate_hard"):
                gate = layer.gate_hard().detach().cpu().numpy().copy()
                self.phase_history["gate_masks"][i].append(gate)

    def on_fit_end(self, trainer, pl_module):
        if self.phase_history is None or len(self.phase_history["output_phase"]) == 0:
            return

        n_layers = len(self.phase_history["layer_phases"])
        T = len(self.phase_history["output_phase"])

        # layout: row 0 = phases at input/expected/predicted
        #         row 1 = learned phase masks per layer
        #         row 2 = learned gate masks per layer (if present)
        ncols = max(3, n_layers)
        nrows = 3 if self.has_gates else 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        # axes is 2D for nrows>=2 and ncols>=3

        ims = []

        # row 0: input / expected (static) + predicted (animated)
        im_in = axes[0, 0].imshow(
            self.input_phase[0], cmap="bwr", vmin=-np.pi, vmax=np.pi
        )
        axes[0, 0].set_title("Input Phase")
        fig.colorbar(im_in, ax=axes[0, 0])

        im_exp = axes[0, 1].imshow(
            self.expected_output[0], cmap="bwr", vmin=-np.pi, vmax=np.pi
        )
        axes[0, 1].set_title("Expected Output Phase")
        fig.colorbar(im_exp, ax=axes[0, 1])

        # prepare titles for layer rows
        for j in range(n_layers):
            axes[1, j].set_title(f"Layer {j+1} Phase")
            if self.has_gates:
                axes[2, j].set_title(f"Layer {j+1} Gate")

        for t in range(T):
            frame = []

            # predicted output phase at epoch t
            im_pred = axes[0, 2].imshow(
                self.phase_history["output_phase"][t][0],
                animated=True,
                cmap="bwr",
                vmin=-np.pi,
                vmax=np.pi,
            )
            axes[0, 2].set_title("Predicted Output Phase")
            frame.append(im_pred)

            # row 1: learned layer phases
            for j in range(n_layers):
                im_phase = axes[1, j].imshow(
                    self.phase_history["layer_phases"][j][t],
                    animated=True,
                    cmap="bwr",
                    vmin=-np.pi,
                    vmax=np.pi,
                )
                frame.append(im_phase)

                # row 2: gate masks (if any)
                if self.has_gates:
                    im_gate = axes[2, j].imshow(
                        self.phase_history["gate_masks"][j][t],
                        animated=True,
                        cmap="gray",
                        vmin=0.0,
                        vmax=1.0,
                    )
                    frame.append(im_gate)

            ims.append(frame)

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)

        # resolve save path relative to logger dir / default_root_dir if not absolute
        base = self.folder_path or getattr(
            trainer.logger, "log_dir", trainer.default_root_dir
        )
        os.makedirs(base, exist_ok=True)

        path = (
            self.save_path
            if os.path.isabs(self.save_path)
            else os.path.join(base, self.save_path)
        )
        ani.save(path, writer="ffmpeg")
        print(f"[PhaseTrackingCallback] Phase evolution animation saved to {path}")

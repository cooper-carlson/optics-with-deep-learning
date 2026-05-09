"""
torch diffractive layer implementation via ASM

helper

author: P. Wiecha, 02/2024
"""
import matplotlib.pyplot as plt
import numpy as np


# helper
def sum_of_list(l, n):
    if n == 0:
        return l[n]
    return l[n] + sum_of_list(l, n - 1)


# plot complex field map
def plot_field(field, title="", extent=None):
    plt.figure(figsize=(8, 3))
    if title:
        plt.suptitle(title)
    plt.subplot(121, title="real")
    plt.imshow(field.real, extent=extent)
    plt.colorbar()

    plt.subplot(122, title="imag")
    plt.imshow(field.imag, extent=extent)
    plt.colorbar()

    plt.tight_layout()
    plt.show()


# helper to predict sample fieldmap and plot original / prediction
def predict_and_plot(
    model,
    val_loader,
    device="cuda",
    plot="field",
    N_load=None,
    plot_idx=None,
    title="",
    savename=None,
    show=True,
):
    x_val = []
    y_val = []
    y_pred = []

    N_load_batches = 9999999 if N_load is None else N_load
    for i, batch in enumerate(val_loader):
        if i >= N_load_batches:
            break

        _x, _y = batch
        _yp = model.to(device)(_x.to(device))

        x_val.append(_x.detach().cpu().numpy())
        y_val.append(_y.detach().cpu().numpy())
        y_pred.append(_yp.detach().cpu().numpy())
    
    if len(y_pred) > 1:
        x_val = np.concatenate(x_val)
        y_val = np.concatenate(y_val)
        y_pred = np.concatenate(y_pred)
    else:
        x_val = np.array(x_val[0])
        y_pred = np.array(y_pred[0])
        y_val = np.array(y_val[0])

    if plot in ["field", "phase", "intensity"]:
        if plot_idx is None:
            plot_idx = np.random.randint(len(y_val))

        if plot == "field":
            if len(plt.get_fignums()) == 0:
                fig = plt.figure(figsize=(8, 8))
            else:
                plt.clf()
                fig = plt.gcf()
                fig.set_size_inches(8, 8)

            if title:
                fig.suptitle(title)

            ax = plt.subplot(221, title="expected-E real")
            im = ax.imshow(y_val[plot_idx, 0], aspect="equal")
            fig.colorbar(im, ax=ax)
            im.set_clim(-1, 1)

            ax = plt.subplot(222, title="expected-E imag")
            im = ax.imshow(y_val[plot_idx, 1], aspect="equal")
            fig.colorbar(im, ax=ax)
            im.set_clim(-1, 1)

            ax = plt.subplot(223, title="out-E real")
            im = ax.imshow(y_pred[plot_idx, 0], aspect="equal")
            fig.colorbar(im, ax=ax)
            im.set_clim(-1, 1)

            ax = plt.subplot(224, title="out-E imag")
            im = ax.imshow(y_pred[plot_idx, 1], aspect="equal")
            fig.colorbar(im, ax=ax)
            im.set_clim(-1, 1)

            plt.tight_layout()

        elif plot == "phase":
            if len(plt.get_fignums()) == 0:
                fig = plt.figure(figsize=(8, 8))
            else:
                plt.clf()
                fig = plt.gcf()
                fig.set_size_inches(8, 8)

            if title:
                fig.suptitle(title)

            phase_in = np.angle(x_val[plot_idx, 0] + 1j * x_val[plot_idx, 1])
            phase_out_val = np.angle(y_val[plot_idx, 0] + 1j * y_val[plot_idx, 1])
            phase_out_pred = np.angle(y_pred[plot_idx, 0] + 1j * y_pred[plot_idx, 1])

            ax = plt.subplot(221, title="in-phase")
            im = ax.imshow(phase_in, cmap="bwr", aspect="equal")
            fig.colorbar(im, ax=ax)
            im.set_clim(-np.pi, np.pi)

            ax = plt.subplot(222, title="expected out-phase")
            im = ax.imshow(phase_out_val, cmap="bwr", aspect="equal")
            fig.colorbar(im, ax=ax)
            im.set_clim(-np.pi, np.pi)

            ax = plt.subplot(223, title="in-phase")
            im = ax.imshow(phase_in, cmap="bwr", aspect="equal")
            fig.colorbar(im, ax=ax)
            im.set_clim(-np.pi, np.pi)

            ax = plt.subplot(224, title="DNN-out-phase")
            im = ax.imshow(phase_out_pred, cmap="bwr", aspect="equal")
            fig.colorbar(im, ax=ax)
            im.set_clim(-np.pi, np.pi)

            plt.tight_layout()

        elif plot == "intensity":
            if len(plt.get_fignums()) == 0:
                plt.figure(figsize=(8, 3))
            else:
                plt.clf()
            if title:
                plt.suptitle(title)
            plt.subplot(121, title="expected-intensity")
            I_in = np.abs(y_val[plot_idx, 0] + 1j * y_val[plot_idx, 1]) ** 2
            plt.imshow(I_in)
            plt.colorbar()
            plt.clim(0, 1)

            plt.subplot(122, title="out-intensity")
            I_out = np.abs(y_pred[plot_idx, 0] + 1j * y_pred[plot_idx, 1]) ** 2
            plt.imshow(I_out)
            plt.colorbar()
            plt.clim(0, 1)

        plt.tight_layout()
        if savename is not None:
            plt.savefig(savename)
        if show:
            plt.show()
    else:
        print('no plot, returning y_val and y_pred.')
        return x_val, y_val, y_pred


# plot learned phases for all diffractive layers
def plot_learned_layers_phase(DNN_model, unwrap=False, filename='', show=True):
    if unwrap:
        from skimage.restoration import unwrap_phase

    diff_l = DNN_model.phaselayerlist

    plt.figure(figsize=(13,2))
    for i in range(len(diff_l)):
        plt.subplot(1,len(diff_l), i+1, title=f'phase layer {i}')
        plt.axis('off')
        phasemap = diff_l[i].phase_weights.detach().cpu().numpy().copy()
        phasemap[phasemap>=np.pi] -= np.pi
        phasemap[phasemap<-np.pi] += np.pi
        if unwrap:
            phasemap = unwrap_phase(phasemap)   # optionally unwrap
        plt.imshow(phasemap, cmap='bwr')
        plt.colorbar()
        plt.clim(-1.05*np.pi, 1.05*np.pi)

    if filename:
        plt.savefig(filename)
    if show:
        plt.show()

# helper.py (add below existing plot_learned_layers_phase)
def plot_learned_layers_phase_and_gates(DNN_model, filename='', show=True):
    import torch, numpy as np, matplotlib.pyplot as plt

    L = DNN_model.phaselayerlist
    n = len(L)

    # Check if ANY layer has gates
    has_any_gates = any(hasattr(layer, 'gate_logits') for layer in L)
    nrows = 2 if has_any_gates else 1

    plt.figure(figsize=(4 * n, 4 * nrows))

    for i, layer in enumerate(L):
        # ---- Phase ----
        plt.subplot(nrows, n, i + 1, title=f'phase {i+1}')
        plt.axis('off')
        ph = layer.phase_weights.detach().cpu().numpy().copy()
        ph[ph >= np.pi] -= np.pi
        ph[ph < -np.pi] += np.pi
        plt.imshow(ph, cmap='bwr')
        plt.colorbar()
        plt.clim(-np.pi, np.pi)

        # ---- Gates (only if at least one layer has them) ----
        if has_any_gates:
            plt.subplot(nrows, n, n + i + 1, title=f'gate {i+1}')
            plt.axis('off')
            if hasattr(layer, 'gate_logits'):
                # hard mask (what forward effectively uses with STE)
                p = torch.sigmoid(layer.gate_logits).detach().cpu().numpy()
                hard = (p >= getattr(layer, 'gate_thresh', 0.5)).astype(float)
                plt.imshow(hard, vmin=0, vmax=1)
                plt.colorbar()
            else:
                # layer has no gates: just leave this subplot blank
                pass

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()

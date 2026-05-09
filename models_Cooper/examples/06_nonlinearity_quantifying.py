"""
nonlinearity quantification test

author: C. Carlson, 02/2026
"""

import os
import csv
import datetime

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
import matplotlib.pyplot as plt

from odnn.diffractivelayer import LearnablePhaseLayer2d, PropagationLayer
from examples.problem_dataloaders import constDataModule
from odnn.helper import predict_and_plot, plot_learned_layers_phase
from odnn.callbacks import LossLoggingCallback, PlottingCallback


# %% main DNN model
class DNN_model_const(pl.LightningModule):
    def __init__(
        self,
        Nx,
        Ny,
        Dx,
        Dy,
        wl,
        propag_z,
        N_layers,
        smoothness_lambda=5e-3,
        init="zeros",
        learning_rate=1e-2,
        nonlinear=True,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.N_layers = N_layers
        self.smoothness_lambda = smoothness_lambda
        self.nonlinear = nonlinear

        # sequence of trainable phase layers
        learnedphase_layers = []
        for _ in range(self.N_layers):
            learnedphase_layers.append(
                LearnablePhaseLayer2d(Nx, Ny, Dx, Dy, init=init)
            )
        self.phaselayerlist = torch.nn.ModuleList(learnedphase_layers)

        # fixed-distance propagation layers
        propa_layers = []
        for _ in range(self.N_layers + 1):
            propa_layers.append(
                PropagationLayer(Nx, Ny, Dx, Dy, wl, propag_z)
            )
        self.propalayerlist = torch.nn.ModuleList(propa_layers)

    def forward(self, x):
        # input is [batch, 2, Nx, Ny] with channels = [real, imag]
        x = x[:, 0, ...] + 1j * x[:, 1, ...]
        x_in = x

        for i in range(self.N_layers):
            # propagate
            x = self.propalayerlist[i](x)

            # structural nonlinearity
            if self.nonlinear:
                x = x * x_in # structural nonlinearity
                # x = np.sigmoid(x)

            # learnable phase
            x = self.phaselayerlist[i](x)

        # final propagation to detector/output plane
        x = self.propalayerlist[-1](x)

        # convert back to 2-channel real/imag format
        x = torch.stack([x.real, x.imag], axis=1)
        return x

    def mse_loss(self, y_pred, y_true):
        return torch.nn.functional.mse_loss(y_pred, y_true)

    def smoothness_loss(self, phase_weights):
        c = torch.exp(-1j * phase_weights)

        dx = c[:, 1:] - c[:, :-1]
        dy = c[1:, :] - c[:-1, :]

        loss = (dx.abs() ** 2).mean() + (dy.abs() ** 2).mean()
        return loss

    def smoothness_loss_all_layers(self):
        loss = 0.0
        for layer in self.phaselayerlist:
            loss += self.smoothness_loss(layer.phase_weights)
        return loss

    def configure_optimizers(self):
        phase_params = []
        for m in self.phaselayerlist:
            if hasattr(m, "phase_weights"):
                phase_params.append(m.phase_weights)

        optimizer = torch.optim.Adam(phase_params, lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self.forward(x)

        loss_data = self.mse_loss(y_pred, y)
        loss_smooth = self.smoothness_lambda * self.smoothness_loss_all_layers()
        loss = loss_data + loss_smooth

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss_data", loss_data, prog_bar=False)
        self.log("train_loss_smooth", loss_smooth, prog_bar=False)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self.mse_loss(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])


# %% helper functions for post-training analysis
def ri_to_complex(x_ri):
    """
    Convert a 2-channel real/imag tensor or array into a complex tensor.

    Input:
        [B, 2, Nx, Ny] or [2, Nx, Ny]
    Output:
        [B, Nx, Ny] or [Nx, Ny]
    """
    if isinstance(x_ri, np.ndarray):
        x_ri = torch.from_numpy(x_ri)

    if x_ri.ndim == 4:
        return x_ri[:, 0, ...] + 1j * x_ri[:, 1, ...]
    elif x_ri.ndim == 3:
        return x_ri[0, ...] + 1j * x_ri[1, ...]
    else:
        raise ValueError(f"Expected ndim 3 or 4, got {x_ri.ndim}")


def circular_mean_phase(field_complex, dims=(-2, -1), eps=1e-12):
    """
    Unweighted circular mean phase over spatial dimensions.
    """
    unit = field_complex / (torch.abs(field_complex) + eps)
    mean_unit = unit.mean(dim=dims)
    return torch.angle(mean_unit)


def intensity_weighted_circular_mean_phase(field_complex, dims=(-2, -1), eps=1e-12):
    """
    Intensity-weighted circular mean phase over spatial dimensions.

    For complex field E:
        weight = |E|^2
        mean phase = angle( sum(weight * exp(i*phase)) / sum(weight) )

    Since exp(i*phase) = E / |E|, this becomes:
        angle( sum( |E|^2 * E/|E| ) / sum(|E|^2) )
      = angle( sum( |E| * E ) / sum(|E|^2) )

    We implement it directly in a numerically stable way.
    """
    amp = torch.abs(field_complex)
    intensity = amp**2
    unit = field_complex / (amp + eps)

    weighted_mean_unit = (intensity * unit).sum(dim=dims) / (intensity.sum(dim=dims) + eps)
    return torch.angle(weighted_mean_unit)

def wrapped_phase_difference(phi_a, phi_b):
    """
    Return wrapped phase difference in [-pi, pi).
    """
    return np.angle(np.exp(1j * (phi_a - phi_b)))


def dataset_mean_mse(model, data_loader, device):
    """
    Mean MSE over an entire dataloader.
    """
    model.eval()
    losses = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = torch.nn.functional.mse_loss(y_pred, y)
            losses.append(loss.item())

    return float(np.mean(losses))


def dataset_samplewise_mean_phase(model, data_loader, device):
    """
    For each sample in the loader, compute the circular-mean output phase
    over the whole NxN grid.

    Returns:
        np.ndarray of shape [Nsamples]
    """
    model.eval()
    phases = []

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            y_pred = model(x)
            y_pred_c = ri_to_complex(y_pred)
            batch_phase = circular_mean_phase(y_pred_c)
            phases.extend(batch_phase.detach().cpu().numpy().tolist())

    return np.asarray(phases, dtype=np.float64)


def get_model_geometry_from_trained_model(trained_model):
    """
    Pull geometry and propagation settings from an already-trained DNN_model_const.
    """
    phase0 = trained_model.phaselayerlist[0]
    prop0 = trained_model.propalayerlist[0]

    Nx = phase0.Nx
    Ny = phase0.Ny
    Dx = phase0.Dx
    Dy = phase0.Dy

    if hasattr(prop0, "current_z"):
        propag_z = float(prop0.current_z().detach().cpu().item())
    else:
        propag_z = float(prop0.propag_z.detach().cpu().item())

    return Nx, Ny, Dx, Dy, propag_z


def clone_model_at_new_wavelength(trained_model, wavelength_new):
    """
    Rebuild the trained model with the same learned phase masks, but with all
    propagation layers recalculated at a new wavelength.
    """
    Nx, Ny, Dx, Dy, propag_z = get_model_geometry_from_trained_model(trained_model)

    model_new = DNN_model_const(
        Nx=Nx,
        Ny=Ny,
        Dx=Dx,
        Dy=Dy,
        wl=wavelength_new,
        propag_z=propag_z,
        N_layers=trained_model.N_layers,
        smoothness_lambda=trained_model.smoothness_lambda,
        init="zeros",
        learning_rate=trained_model.learning_rate,
        nonlinear=trained_model.nonlinear,
    )

    with torch.no_grad():
        for old_layer, new_layer in zip(
            trained_model.phaselayerlist, model_new.phaselayerlist
        ):
            new_layer.phase_weights.copy_(old_layer.phase_weights)

    return model_new


def sweep_wavelength_mse(
    trained_model,
    wavelengths,
    dataset,
    amplitude,
    gauss_width,
    batch_size,
    device,
    phase_function,
):
    """
    Figure type 1:
    train once, then evaluate fixed learned phase masks over a wavelength sweep.

    Returns dict with:
        wavelengths
        mean_mse
    """
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    mean_mse = []

    Nx, _, _, _, _ = get_model_geometry_from_trained_model(trained_model)

    for wl in wavelengths:
        eval_model = clone_model_at_new_wavelength(trained_model, wl).to(device)
        eval_model.eval()

        dm = constDataModule(
            wavelength=wl,
            batch_size=batch_size,
            img_pix_size=Nx,
            dataset=dataset,
            amplitude=amplitude,
            gauss_width=gauss_width,
            test_size=0.0,
            phase_function=phase_function,
        )
        dm.setup()
        val_loader = dm.val_dataloader()

        mean_mse.append(dataset_mean_mse(eval_model, val_loader, device))

    return {
        "wavelengths": wavelengths,
        "mean_mse": np.asarray(mean_mse, dtype=np.float64),
    }


def sweep_input_phase_response(
    trained_model,
    phase_values,
    wavelength,
    amplitude,
    gauss_width,
    batch_size,
    device,
    phase_function,
):
    """
    Figure type 2:
    keep the trained model fixed, vary the uniform input phase, and measure the
    circular-mean output phase over the whole grid.

    Returns dict with:
        input_phase
        output_phase_mean
        target_phase_mean
        phase_error_wrapped
    """
    phase_values = np.asarray(phase_values, dtype=np.float64)
    Nx, _, _, _, _ = get_model_geometry_from_trained_model(trained_model)

    dm = constDataModule(
        wavelength=wavelength,
        batch_size=batch_size,
        img_pix_size=Nx,
        dataset=phase_values,
        amplitude=amplitude,
        gauss_width=gauss_width,
        test_size=0.0,
        phase_function=phase_function,
    )
    dm.setup()
    loader = dm.val_dataloader()

    trained_model.eval()
    output_phase_mean = []
    target_phase_mean = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = trained_model(x)

            y_pred_c = ri_to_complex(y_pred)
            y_true_c = ri_to_complex(y)

            pred_phase = intensity_weighted_circular_mean_phase(y_pred_c)
            true_phase = intensity_weighted_circular_mean_phase(y_true_c)

            output_phase_mean.extend(pred_phase.detach().cpu().numpy().tolist())
            target_phase_mean.extend(true_phase.detach().cpu().numpy().tolist())

    output_phase_mean = np.asarray(output_phase_mean, dtype=np.float64)
    target_phase_mean = np.asarray(target_phase_mean, dtype=np.float64)

    phase_error_wrapped = wrapped_phase_difference(
        output_phase_mean, target_phase_mean
    )

    return {
        "input_phase": phase_values,
        "output_phase_mean": output_phase_mean,
        "target_phase_mean": target_phase_mean,
        "phase_error_wrapped": phase_error_wrapped,
    }

def sweep_phase_response_across_wavelengths(
    trained_model,
    phase_values,
    wavelengths,
    amplitude,
    gauss_width,
    batch_size,
    device,
    phase_function,
):
    """
    Train at one wavelength, then evaluate output phase vs input phase
    for several different test wavelengths.

    Returns dict with:
        input_phase
        target_phase_mean
        wavelength_<nm>_output_phase
    """
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    phase_values = np.asarray(phase_values, dtype=np.float64)

    result = {
        "input_phase": phase_values,
    }

    target_saved = False

    for wl in wavelengths:
        eval_model = clone_model_at_new_wavelength(trained_model, wl).to(device)
        eval_model.eval()

        curve = sweep_input_phase_response(
            trained_model=eval_model,
            phase_values=phase_values,
            wavelength=wl,
            amplitude=amplitude,
            gauss_width=gauss_width,
            batch_size=batch_size,
            device=device,
            phase_function=phase_function,
        )

        wl_nm = wl * 1e9
        key = f"wavelength_{wl_nm:.1f}_nm_output_phase"
        result[key] = curve["output_phase_mean"]

        if not target_saved:
            result["target_phase_mean"] = curve["target_phase_mean"]
            target_saved = True

    return result

def save_curve_to_csv(curve_dict, filename):
    """
    Save dict of equal-length 1D arrays to CSV.
    """
    keys = list(curve_dict.keys())
    n = len(curve_dict[keys[0]])

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([curve_dict[k][i] for k in keys])


def plot_wavelength_mse_curve(curve_dict, training_wavelength=None, filename=None, show=False):
    plt.figure(figsize=(6, 4))
    plt.plot(curve_dict["wavelengths"] * 1e9, curve_dict["mean_mse"], marker="o", markersize=2, label="MSE")

    if training_wavelength is not None:
        plt.axvline(
            training_wavelength * 1e9,
            linestyle="--",
            color="orange",
            linewidth=1.5,
            label=f"training wavelength = {training_wavelength * 1e9:.1f} nm",
        )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mean dataset MSE")
    plt.title("MSE vs wavelength")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

def plot_phase_response_curve(curve_dict, filename=None, show=False):
    plt.figure(figsize=(6, 4))
    plt.plot(
        curve_dict["input_phase"],
        curve_dict["output_phase_mean"],
        marker="o",
        markersize=2,
        label="predicted",
    )
    plt.plot(
        curve_dict["input_phase"],
        curve_dict["target_phase_mean"],
        linestyle="--",
        label="target",
    )
    plt.xlabel("Input phase (rad)")
    plt.ylabel("Average output phase (rad)")
    plt.title("Average output phase vs input phase")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

def plot_phase_response_across_wavelengths(
    curve_dict,
    training_wavelength=None,
    filename=None,
    show=False,
):
    plt.figure(figsize=(7, 5))

    input_phase = curve_dict["input_phase"]

    for key, values in curve_dict.items():
        if key.startswith("wavelength_") and key.endswith("_output_phase"):
            wl_label = key.replace("wavelength_", "").replace("_output_phase", "").replace("_", " ")
            plt.plot(input_phase, values, marker="o", markersize=2, label=wl_label)

    if "target_phase_mean" in curve_dict:
        plt.plot(
            input_phase,
            curve_dict["target_phase_mean"],
            linestyle="--",
            linewidth=2,
            color="black",
            label="target",
        )

    plt.xlabel("Input phase (rad)")
    plt.ylabel("Average output phase (rad)")
    plt.title("Output phase vs input phase at multiple wavelengths")
    plt.grid(True)

    if training_wavelength is not None:
        plt.text(
            0.02,
            0.02,
            f"trained at {training_wavelength * 1e9:.1f} nm",
            transform=plt.gca().transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.legend()
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

def make_run_folder(params, root="runs", prefix="run"):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    folder_name = f"run_{prefix}_{timestamp}"
    folder_path = os.path.join(root, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    params_file = os.path.join(folder_path, "params.csv")
    with open(params_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        for key, value in params.items():
            writer.writerow([key, value])

    return folder_path

def append_summary_row(summary_csv_path, row_dict):
    file_exists = os.path.exists(summary_csv_path)
    fieldnames = list(row_dict.keys())

    with open(summary_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def run_single_experiment(config, root_folder, device):
    wavelength = config["wavelength"]
    Nxy = config["Nxy"]
    Ny = Nxy

    dataset = config["dataset"]
    N_layers = config["N_layers"]
    propag_z = config["propag_z"]
    Dx = config["Dx"]
    Dy = config["Dy"]
    batch_size = config["batch_size"]
    amplitude = config["amplitude"]
    gauss_width = config["gauss_width"]
    learning_rate = config["learning_rate"]
    smoothness_lambda = config["smoothness_lambda"]
    init = config["init"]
    max_epochs = config["max_epochs"]
    print_each_n_epochs = config["print_each_n_epochs"]
    nonlinear = config["nonlinear"]
    wavelengths_eval = config["wavelengths_eval"]
    phase_probe = config["phase_probe"]
    phase_function = config["phase_function"]

    run_name = f"wl_{wavelength*1e9:.1f}nm_Nxy_{Nxy}"
    folder_path = make_run_folder(config, root=root_folder, prefix=run_name)

    data_module = constDataModule(
        wavelength=wavelength,
        batch_size=batch_size,
        img_pix_size=Nxy,
        dataset=dataset,
        amplitude=amplitude,
        gauss_width=gauss_width,
        test_size=0.0,
        phase_function=phase_function,
    )
    data_module.setup()

    val_loader = data_module.val_dataloader()
    x_batch, y_batch = next(iter(val_loader))
    print(f"[{run_name}] input shape: {x_batch.shape}, output shape: {y_batch.shape}")

    DNN_model = DNN_model_const(
        Nx=Nxy,
        Ny=Ny,
        Dx=Dx,
        Dy=Dy,
        wl=wavelength,
        propag_z=propag_z,
        N_layers=N_layers,
        init=init,
        learning_rate=learning_rate,
        smoothness_lambda=smoothness_lambda,
        nonlinear=nonlinear,
    )
    DNN_model.to(device)

    for idx in range(min(batch_size, len(dataset))):
        predict_and_plot(
            DNN_model,
            val_loader,
            device,
            plot_idx=idx,
            plot="phase",
            savename=os.path.join(folder_path, f"init_phase_case{idx}.png"),
            show=False,
        )

    trainer = pl.Trainer(
        accelerator=device.type,
        devices=1,
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=False,
        callbacks=[
            LossLoggingCallback(
                folder_path=folder_path,
                plot_each_n_epochs=print_each_n_epochs,
            ),
            PlottingCallback(
                val_loader,
                plot_each_n_epochs=print_each_n_epochs,
                plot_each_n_batches=100,
                folder_path=folder_path,
            ),
            RichProgressBar(),
        ],
    )

    trainer.fit(DNN_model, data_module)

    for idx in range(min(batch_size, len(dataset))):
        predict_and_plot(
            DNN_model,
            val_loader,
            device,
            plot_idx=idx,
            plot="phase",
            savename=os.path.join(folder_path, f"final_phase_case{idx}.png"),
            show=False,
        )

    plot_learned_layers_phase(
        DNN_model,
        filename=os.path.join(folder_path, "diffractivelayers.png"),
        show=False,
    )

    mean_val_loss = dataset_mean_mse(DNN_model, val_loader, device)

    with open(os.path.join(folder_path, "final_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["mean_val_mse", mean_val_loss])

    wl_curve = sweep_wavelength_mse(
        trained_model=DNN_model,
        wavelengths=wavelengths_eval,
        dataset=dataset,
        amplitude=amplitude,
        gauss_width=gauss_width,
        batch_size=batch_size,
        device=device,
        phase_function=phase_function,
    )
    save_curve_to_csv(
        wl_curve,
        os.path.join(folder_path, "mse_vs_wavelength.csv"),
    )
    plot_wavelength_mse_curve(
        wl_curve,
        training_wavelength=wavelength,
        filename=os.path.join(folder_path, "mse_vs_wavelength.png"),
        show=False,
    )

    phase_curve = sweep_input_phase_response(
        trained_model=DNN_model,
        phase_values=phase_probe,
        wavelength=wavelength,
        amplitude=amplitude,
        gauss_width=gauss_width,
        batch_size=len(phase_probe),
        device=device,
        phase_function=phase_function,
    )
    save_curve_to_csv(
        phase_curve,
        os.path.join(folder_path, "phase_response.csv"),
    )
    plot_phase_response_curve(
        phase_curve,
        filename=os.path.join(folder_path, "phase_response.png"),
        show=False,
    )
        # combined figure: output phase vs input phase for several test wavelengths
    phase_response_wavelengths = np.array([
        450e-9,
        500e-9,
        532e-9,
        600e-9,
        650e-9,
    ])
    multi_phase_curve = sweep_phase_response_across_wavelengths(
        trained_model=DNN_model,
        phase_values=phase_probe,
        wavelengths=phase_response_wavelengths,
        amplitude=amplitude,
        gauss_width=gauss_width,
        batch_size=len(phase_probe),
        device=device,
        phase_function=phase_function,
    )

    save_curve_to_csv(
        multi_phase_curve,
        os.path.join(folder_path, "phase_response_across_wavelengths.csv"),
    )
    plot_phase_response_across_wavelengths(
        multi_phase_curve,
        training_wavelength=wavelength,
        filename=os.path.join(folder_path, "phase_response_across_wavelengths.png"),
        show=False,
    )

    return {
        "run_name": run_name,
        "folder_path": folder_path,
        "wavelength_m": wavelength,
        "wavelength_nm": wavelength * 1e9,
        "Nxy": Nxy,
        "gauss_width": gauss_width,
        "mean_val_mse": mean_val_loss,
        "phase_error_abs_mean": float(np.mean(np.abs(phase_curve["phase_error_wrapped"]))),
        "phase_error_abs_max": float(np.max(np.abs(phase_curve["phase_error_wrapped"]))),
    }

# %% setup and batch execution
def main():
    # dataset of scalar phase values (radians)
    dataset = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    # fixed model/physics params
    N_layers = 5
    propag_z = 50e-6
    Dx = 50e-7
    Dy = Dx
    amplitude = 1.0

    # ANN params
    learning_rate = 1e-2
    smoothness_lambda = 0
    init = "zeros"      # or "rnd"
    max_epochs = 501
    print_each_n_epochs = 500
    nonlinear = False
    
    def phase_function(phi):
        return np.mod(phi ** 2, 2 * np.pi)

    # sweep params
    wavelength_list = [450e-9, 532e-9, 650e-9]
    Nxy_list = [50]

    # post-training sweep params
    wavelengths_eval = np.linspace(450e-9, 650e-9, 201)
    phase_probe = np.linspace(-np.pi, np.pi, 201)

    batch_size = len(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_root = make_run_folder(
        {
            "wavelength_list": wavelength_list,
            "Nxy_list": Nxy_list,
            "dataset": dataset,
            "N_layers": N_layers,
            "propag_z": propag_z,
            "Dx": Dx,
            "Dy": Dy,
            "amplitude": amplitude,
            "learning_rate": learning_rate,
            "smoothness_lambda": smoothness_lambda,
            "init": init,
            "max_epochs": max_epochs,
            "print_each_n_epochs": print_each_n_epochs,
            "nonlinear": nonlinear,
        },
        root="runs",
        prefix="batch_experiment",
    )

    summary_csv_path = os.path.join(experiment_root, "summary.csv")

    for wavelength in wavelength_list:
        for Nxy in Nxy_list:
            gauss_width = Nxy / 6.0

            config = {
                "dataset": dataset,
                "N_layers": N_layers,
                "propag_z": propag_z,
                "Dx": Dx,
                "Dy": Dy,
                "Nxy": Nxy,
                "wavelength": wavelength,
                "batch_size": batch_size,
                "amplitude": amplitude,
                "gauss_width": gauss_width,
                "learning_rate": learning_rate,
                "smoothness_lambda": smoothness_lambda,
                "init": init,
                "max_epochs": max_epochs,
                "print_each_n_epochs": print_each_n_epochs,
                "nonlinear": nonlinear,
                "wavelengths_eval": wavelengths_eval,
                "phase_probe": phase_probe,
                "phase_function": phase_function,
            }

            result = run_single_experiment(
                config=config,
                root_folder=experiment_root,
                device=device,
            )
            append_summary_row(summary_csv_path, result)
            print("Finished:", result["run_name"], "mean_val_mse =", result["mean_val_mse"])


if __name__ == "__main__":
    main()
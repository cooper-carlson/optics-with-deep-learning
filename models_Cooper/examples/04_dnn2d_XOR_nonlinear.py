"""
torch diffractive neural network implementation

author: P. Wiecha, 02/2024
"""

# %%

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import h5py
import numpy as np
import matplotlib.animation as animation

from odnn.diffractivelayer import LearnablePhaseLayer2d, PropagationLayer
from odnn.problem_dataloaders import logicGateXORPhaseDataModule
from odnn.helper import predict_and_plot
from odnn.helper import plot_learned_layers_phase
from odnn.callbacks import ZWarmupCallback, LossLoggingCallback, PlottingCallback, PhaseTrackingCallback

import os, datetime, csv

# %% main DNN model
class DNN_model_imaging(pl.LightningModule):
    def __init__(
        self, Nx, Ny, Dx, Dy, wl, 
        propag_z, N_layers,
        propag_z_list=1e-2,
        smoothness_lambda=5e-2,
        init="zeros",
        learning_rate=0.02,
        trainable_z=False,
        z_min=1e-6,
        z_max=200e-6,
        nonlinear=False
    ):
        super().__init__()
        # pytorch-lightning: configure optimizers in the model class
        self.learning_rate = learning_rate
        self.N_layers = N_layers
        self.smoothness_lambda = smoothness_lambda
        self.nonlinear = nonlinear
        
        # sequence of several trainable phase layers (all same dim.)
        learnedphase_layers = [
            LearnablePhaseLayer2d(Nx, Ny, Dx, Dy, init=init)
            for _ in range(self.N_layers)
        ]
        self.phaselayerlist = torch.nn.ModuleList(learnedphase_layers)

        # sequence of several propagation phase layers between, here: all same dist.
        propa_layers = []
        if isinstance(propag_z_list, (list, tuple)):
            assert len(propag_z_list) == N_layers + 1, "propag_z_list must have N_layers + 1 elements"
            for i in range(N_layers + 1):  # one more than learned weights layers
                propa_layers.append(PropagationLayer(Nx, Ny, Dx, Dy, wl, propag_z_list[i], trainable_z=trainable_z, z_min=z_min, z_max=z_max))
        else:
            for i in range(self.N_layers + 1):  # one more than learned weights layers
                propa_layers.append(PropagationLayer(Nx, Ny, Dx, Dy, wl, propag_z, trainable_z=trainable_z, z_min=z_min, z_max=z_max))

        self.propalayerlist = torch.nn.ModuleList(propa_layers)

    def activation_function(self, x):
        # relu: apply homemade complex relu activation function
        # return x.real * (x.real > 0) + 1j * x.imag * (x.imag > 0)
        # sigmoid: apply homemade complex sigmoid activation function
        # return 1 / (1 + torch.exp(-x.real)) + 1j * 1 / (1 + torch.exp(-x.imag))
        return x.real*x.real + 1j*x.imag*x.imag  # intensity
    
    def forward(self, x):
        # convert scalar 2-channel input to complex for internal calc.
        x = x[:, 0, ...] + 1j * x[:, 1, ...]
        x_in = torch.angle(x)  # keep original input phase for later use

        for i in range(self.N_layers):
            x = self.propalayerlist[i](x)

            # nonlinear: apply input modulation after each hidden layer
            # x = self.activation_function(x)
            if self.nonlinear:
                x = x * torch.exp(-1j * x_in)

            # learnable phase
            x = self.phaselayerlist[i](x)

        # propagation to detector
        x = self.propalayerlist[-1](x)

        # convert complex back to separate real / imag channels
        x = torch.stack([x.real, x.imag], axis=1)
        return x

    def mse_loss(self, y_pred, y_true):
        return torch.nn.functional.mse_loss(y_pred, y_true)
    
    def smoothness_loss(self, phase_weights):
        c = torch.exp(1j * phase_weights)

        dx = c[:, 1:] - c[:, :-1]
        dy = c[1:, :] - c[:-1, :]

        loss = (dx.abs()**2).mean() + (dy.abs()**2).mean()

        return loss
    
    def smoothness_loss_all_layers(self):
        loss = 0
        for layer in self.phaselayerlist:
            phase_weights = layer.phase_weights
            loss += self.smoothness_loss(phase_weights)
        return loss

    def configure_optimizers(self):
        phase_params = []
        z_params = []

        for m in self.propalayerlist:
            if hasattr(m, "z_raw"):
                z_params.append(m.z_raw)
        for m in self.phaselayerlist:
            if hasattr(m, "phase_weights"):
                phase_params.append(m.phase_weights)

        param_groups = []
        if phase_params:
            param_groups.append({
                "params": phase_params
            })
            
        if z_params:
            param_groups.append({
                "params": z_params,
                "lr": self.learning_rate * z_learning_rate_factor,
            })

        optimizer = torch.optim.Adam(param_groups, lr=self.learning_rate)
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

        for i, layer in enumerate(self.propalayerlist):
            if hasattr(layer, "current_z"):
                z_val = layer.current_z()
                self.log(f"layer{i}_z", z_val, prog_bar=False)
                self.log(f"layer{i}_z_um", z_val*1e6, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self.mse_loss(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])

def make_run_folder(params, root="runs"):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    folder_name = f"run_{timestamp}"
    folder_path = os.path.join(root, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Save parameters to a CSV file
    params_file = os.path.join(folder_path, "params.csv")
    with open(params_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for key, value in params.items():
            writer.writerow([key, value])

    return folder_path

# def sum_to_4bit_binary_list(nums):
#     s = nums[0] + nums[1] + nums[2] + nums[3]   # sum of the 4 numbers

#     # convert to 4-bit binary string (e.g., 7 -> "0111")
#     b = format(s, "04b")

#     # turn that string into a list of ints: "0111" -> [0, 1, 1, 1]
#     return [int(ch) for ch in b]
# %% setup

dataset = [
    [[1, 1, 1, 1], [1, 1, 1, 2]],
    [[2, 2, 2, 2], [2, 3, 2, 3]],
    # [[2, 0, 3, 0], [-2, 0, -3, 0]],
    # [[1, 1, 2, 0], [-1, 1, -2, 0]],
    # [[1, 0, 2, 1], [-1, 0, -2, 1]],
    # [[2, 1, 3, 1], [-2, 1, -3, 1]],
    # [[1, 2, 3, 0], [-1, 2, -3, 0]],
    # [[2, 0, 4, 1], [-2, 0, -4, 1]],
    # [[0, 0, 1, 0], [0, 0, -1, 0]],
    # [[0, 1, 2, 0], [0, 1, -2, 0]],
    # [[1, 0, 0, 1], [-1, 0, 0, 1]],
    # [[2, 1, 0, 0], [-2, 1, 0, 0]],
]

# # all ordered 4-tuples (a,b,c,d) where each is 0..4
# for a in range(3):
#     for b in range(3):
#         for c in range(3):
#             for d in range(3):
#                 s = a + b + c + d
#                 if 0 <= s <= 15:
#                     dataset.append([[a, b, c, d], sum_to_4bit_binary_list([a, b, c, d])])

# print(len(dataset))   # 270
# print(dataset[:10])   # peek at the first 10 rows

# nr of layers and propagation distance in between (in meters)
N_layers = 3
propag_z = 50e-6
z_min = 10e-6
z_max = 200e-6

# phase modulation layer size (in meters)
Dx = 50e-7
Dy = Dx

# input data parameters, mostly in pixels
scale=2
batch_size=1
Ngauss=int(512*scale)
wgauss=int(16*scale)
offsetgauss_x=int(32*scale)
offsetgauss_y=int(0*scale)
pixel_phase=np.pi/2
phase_offset=np.pi/2
noise_amplitude=1e-3
Nxy = int(128*scale)
nonlinear = True

Ny = Nxy

# wavelength (in meters)
wavelength = 532e-9

# ANN params
learning_rate = 1e-2
smoothness_lambda = 1e-2
init = "zeros"  # "zeros" or "rnd"
trainable_z = True  # if True, propagation distances are trainable paramfeters
z_freeze_epochs = 2000  # number of epochs with frozen propagation distances at start of training
max_epochs = 1000 # total number of training epochs
print_each_n_epochs = 500  # print loss each n epochs
z_learning_rate_factor = 1e-2
animate_every_n_epochs = 10  # save phase evolution animation every n epochs

params = dict(
    N_layers=N_layers,
    propag_z=propag_z,
    z_min=z_min,
    z_max=z_max,
    Dx=Dx,
    Dy=Dy,
    batch_size=batch_size,
    Ngauss=Ngauss,
    wgauss=wgauss,
    offsetgauss_x=offsetgauss_x,
    offsetgauss_y=offsetgauss_y,
    pixel_phase=pixel_phase,
    phase_offset=phase_offset,
    noise_amplitude=noise_amplitude,
    Nxy=Nxy,
    wavelength=wavelength,
    learning_rate=learning_rate,
    smoothness_lambda=smoothness_lambda,
    init=init,
    trainable_z=trainable_z,
    z_freeze_epochs=z_freeze_epochs,
    max_epochs=max_epochs,
    nonlinear=nonlinear,
    dataset=dataset
)

folder_path = make_run_folder(params, root="runs")

# --- init dataset
data_module = logicGateXORPhaseDataModule(
    wavelength=wavelength,
    img_pix_size=Nxy,
    pixel_phase=pixel_phase,
    phase_offset=phase_offset,
    batch_size=batch_size,
    offsetgauss_x=offsetgauss_x,
    offsetgauss_y=offsetgauss_y,
    noise_amplitude=noise_amplitude,
    Ngauss=Ngauss,
    wgauss=wgauss,
    dataset=dataset
)

data_module.setup()

# get the validation data loader. we'll use it explicitly for testing
val_loader = data_module.val_dataloader()
x_batch, y_batch = next(iter(val_loader))  # testing: get first batch
print("Data: input shape:", x_batch.shape, "output shape:", y_batch.shape)

# --- init DNN model
DNN_model = DNN_model_imaging(
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
    trainable_z=trainable_z,
    z_min=z_min,
    z_max=z_max,
    nonlinear=nonlinear
)

# move the model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DNN_model.to(device)

# %%
# evaluate initialized model
# predict_and_plot(
#     DNN_model,
#     val_loader,
#     device,
#     plot_idx=1,
#     title="init",
# )

# %% train the DNN
trainer = pl.Trainer(
    accelerator=device.type, devices=1, max_epochs=max_epochs,
    callbacks=[
        PhaseTrackingCallback(val_loader, plot_each_n_epochs=animate_every_n_epochs,
                              save_path="phase_evolution.mp4", folder_path=folder_path),
        ZWarmupCallback(warmup_epochs=z_freeze_epochs),
        LossLoggingCallback(folder_path=folder_path, plot_each_n_epochs=print_each_n_epochs),
        PlottingCallback(val_loader, plot_each_n_epochs=print_each_n_epochs,
                         plot_each_n_batches=100, folder_path=folder_path),
        RichProgressBar(),
    ],
)

# the learning rate can be manually set
DNN_model.learning_rate = learning_rate
trainer.fit(DNN_model, data_module)

# %%
for idx in range(batch_size):
    predict_and_plot(
        DNN_model,
        val_loader,
        device,
        plot_idx=idx,
        plot="phase",
        savename=os.path.join(folder_path, f"final_phase_case{idx}.png"),
    )

# %% plot the learned phase maps

plot_learned_layers_phase(
    DNN_model,
    filename=os.path.join(folder_path, "diffractivelayers.png")
    # filename="plots/02_xor_linDNN_diffractivelayers.svg",
)

# # %%
# # single layer plot
# diff_l = DNN_model.difflayer

# plt.figure(figsize=(4,3))
# plt.subplot(1,1,1, title=f'phase layer')
# plt.axis('off')
# phasemap = diff_l.phase_weights.detach().cpu().numpy()
# phasemap[phasemap>=np.pi] -= np.pi
# phasemap[phasemap<-np.pi] += np.pi
# plt.imshow(phasemap, cmap='bwr')
# plt.colorbar()
# plt.clim(-1.05*np.pi, 1.05*np.pi)

# plt.show()

# %%
import diffractsim as df
import torch
import matplotlib.pyplot as plt
import numpy as np

class LearnablePhaseLayer2d(torch.nn.Module):

    def __init__(self, Nx, Ny, Dx, Dy, init="zeros", dataset=None):
        super().__init__()

        # field size / discretization
        self.Nx = Nx
        self.Ny = Ny
        self.Dx = Dx
        self.Dy = Dy
        self.dx = Dx / Nx
        self.dy = Dy / Ny

        self.dataset = dataset

        # initalize the learnable phase
        if init == "zeros":
            self.phase_weights = torch.nn.Parameter(
                torch.zeros(self.Nx, self.Ny, dtype=torch.float32), requires_grad=True
            )
        elif init == "rnd":
            self.phase_weights = torch.nn.Parameter(
                torch.randn(self.Nx, self.Ny, dtype=torch.float32), requires_grad=True
            )

    def forward(self, e_in):
        # add 'learned' phase to incident wavefront
        #learned_phase = torch.exp(-1j * self.phase_weights)
        learned_phase = torch.exp(-1j * self.phase_weights)
        e_plus_phase = e_in * learned_phase

        return e_plus_phase
    
class LearnablePhaseGateLayer2d(torch.nn.Module):

    def __init__(self, Nx, Ny, Dx, Dy,
                 init="zeros",
                 gate_init="on",
                 gate_thresh=0.5,
                 ste_temperature=1.0):
        """2D learnable phase layer with a per-cell binary gate.

        Args:
            Nx, Ny (int): nr of discretization steps
            Dx, Dy (float): physical extension of the field (in meters)
            init (str): "zeros" or "rnd" for phase initialization
            gate_init (str or float): "off" | "on" | "rnd" | p in (0,1)
        """
        super().__init__()

        # field discretization
        self.Nx = Nx
        self.Ny = Ny
        self.Dx = Dx
        self.Dy = Dy

        # --- phase weights ---
        if init == "zeros":
            phase0 = torch.zeros(self.Nx, self.Ny, dtype=torch.float32)
        elif init == "rnd":
            phase0 = torch.randn(self.Nx, self.Ny, dtype=torch.float32)
        else:
            raise ValueError("init must be 'zeros' or 'rnd'")

        self.phase_weights = torch.nn.Parameter(phase0, requires_grad=True)

        # --- gate logits ---
        # these will be set based on desired open probabilities p
        self.gate_logits = torch.nn.Parameter(
            torch.empty(self.Nx, self.Ny, dtype=torch.float32),
            requires_grad=True,
        )

        # choose initial open probabilities p in (0,1)
        if gate_init == "off":
            # mostly closed
            p = torch.full((Nx, Ny), 0.05, dtype=torch.float32)
        elif gate_init == "on":
            # mostly open
            p = torch.full((Nx, Ny), 0.95, dtype=torch.float32)
        elif gate_init == "rnd":
            # fully random per cell in (0.1, 0.9)
            p = 0.1 + 0.8 * torch.rand(Nx, Ny, dtype=torch.float32)
        elif isinstance(gate_init, float) and 0.0 < gate_init < 1.0:
            # user-specified uniform probability
            p = torch.full((Nx, Ny), gate_init, dtype=torch.float32)
        else:
            raise ValueError("gate_init must be 'off'|'on'|'rnd'|float in (0,1)")
    
        # match dtype/device and convert probability to logits
        with torch.no_grad():
            p = p.to(self.gate_logits.device, self.gate_logits.dtype)
            logit = torch.log(p / (1.0 - p))
            self.gate_logits.copy_(logit)

        self.gate_thresh = gate_thresh
        self.ste_temperature = ste_temperature

    # ---------- gate helpers ----------

    def gate_prob(self):
        """Open probability per cell."""
        return torch.sigmoid(self.gate_logits / self.ste_temperature)

    @torch.no_grad()
    def gate_hard(self):
        """Binary gate mask (0/1) using the threshold."""
        p = self.gate_prob()
        return (p >= self.gate_thresh).to(p.dtype)

    def _gate_mask_STE(self):
        """Straight-through-estimator gate mask."""
        p = self.gate_prob()
        y_hard = (p >= self.gate_thresh).to(p.dtype)
        # forward: y_hard ; backward: gradient through p
        return y_hard.detach() - p.detach() + p

    # ---------- forward + regularizers ----------

    def forward(self, e_in):
        # learned phase
        learned_phase = torch.exp(-1j * self.phase_weights)
        # STE gate
        gate = self._gate_mask_STE()
        # broadcast (Nx,Ny) → (batch,Nx,Ny) automatically
        return e_in * learned_phase * gate

    def l1_gate_regularizer(self):
        """Mean gate probability (can be used as sparsity regularizer)."""
        return self.gate_prob().mean()

    @torch.no_grad()
    def gate_utilization(self):
        """Fraction of currently-open cells based on hard mask."""
        return self.gate_hard().mean()
    
# pytorch diffractive layer for propagation of wavefront by distance `propag_z`
class PropagationLayer(torch.nn.Module):

    def __init__(self, Nx, Ny, Dx, Dy, wl, propag_z, init="zeros", trainable_z=False, z_min=1e-6, z_max=200e-6):
        super().__init__()

        # field size / discretization
        self.Nx = Nx
        self.Ny = Ny
        self.Dx = Dx
        self.Dy = Dy
        self.dx = Dx / Nx
        self.dy = Dy / Ny
        self.wl = wl
        self.trainable_z = trainable_z
        self.z_min = z_min
        self.z_max = z_max

        self.k0 = torch.as_tensor(2 * torch.pi / self.wl)

        # precalculate fixed propagation phase angular spectrum (fix propagation distance)
        freq_x = torch.fft.fftfreq(self.Nx, d=self.dx)  # angular spectrum frequencies
        freq_y = torch.fft.fftfreq(self.Ny, d=self.dy)
        freq_x_c = torch.fft.fftshift(freq_x)
        freq_y_c = torch.fft.fftshift(freq_y)
        f_xx_c, f_yy_c = torch.meshgrid(freq_x_c, freq_y_c, indexing="xy")
        f_tot = 1 / self.wl
        f_zz_c_squared = f_tot**2 - (f_xx_c**2 + f_yy_c**2)
        k0_z = 2 * torch.pi * torch.sqrt(torch.abs(f_zz_c_squared))

        # multiply evanescent modes with complex unit
        k0_z = torch.where(f_zz_c_squared >= 0, k0_z, torch.tensor(1j) * k0_z)

        # register propagation phase as non-trainable parameter
        self.register_buffer("k0_z", k0_z)

        if trainable_z:
            raw_init = np.log((propag_z - z_min) / (z_max - propag_z))
            self.z_raw = torch.nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        else:
            self.register_buffer("propag_z", torch.tensor(propag_z, dtype=torch.float32))

    def current_z(self):
        if self.trainable_z:
            z = self.z_min + (self.z_max - self.z_min) * torch.sigmoid(self.z_raw)
            return z
        else:
            return self.propag_z

    def forward(self, e_in):
        # 2D FFT --> angular spectrum
        fft_e = torch.fft.fft2(e_in)
        fft_e_c = torch.fft.fftshift(fft_e)

        # 2D iFFT --> field after propagation
        z = self.current_z()

        phase_term = torch.exp(1j * self.k0_z * z)
        fft_e_propa_c = torch.fft.ifftshift(fft_e_c * phase_term)

        fft_e_propa_c = torch.fft.ifftshift(fft_e_c * phase_term)
        e_out = torch.fft.ifft2(fft_e_propa_c)
        return e_out

# fixed operations, not used
class ConvertToPhaseLayer(torch.nn.Module):
    # phase-modulation of a unit amplitude field
    def __init__(self):
        super().__init__()

    def forward(self, x):
        e_unit = torch.ones(x.shape, dtype=torch.complex64, device=x.device)
        e_with_phase = e_unit * torch.exp(-1j * x)
        return e_with_phase


class ConstantPhaseLayer(torch.nn.Module):
    # use an input scalar array as phase-modulation for a field provided by a second input
    def __init__(self):
        super().__init__()

    def forward(self, e_field, applied_phase):
        #e_modulated = e_field * torch.exp(-1j * applied_phase)
        e_modulated = e_field * torch.exp(1j * applied_phase)
        return e_modulated

# pytorch diffractive layer with learnable phase and fixed propagation pathlength
# (less flexible:
# all-in-one: consists of learnable phase + propagation)
class _DiffractLayer(torch.nn.Module):

    def __init__(self, Nx, Ny, Dx, Dy, wl, propag_z, init="zeros"):
        super().__init__()

        self.Nx = Nx
        self.Ny = Ny
        self.Dx = Dx
        self.Dy = Dy
        self.dx = Dx / Nx
        self.dy = Dy / Ny
        self.wl = wl
        self.propag_z = propag_z

        self.k0 = torch.as_tensor(2 * torch.pi / self.wl)

        # the learnable phase
        if init == "zeros":
            self.phase_weights = torch.nn.Parameter(
                torch.zeros(self.Nx, self.Ny, dtype=torch.float32), requires_grad=True
            )
        elif init == "rnd":
            self.phase_weights = torch.nn.Parameter(
                torch.randn(self.Nx, self.Ny, dtype=torch.float32), requires_grad=True
            )
# precalculate fixed propagation phases (since this is always the same)
        freq_x = torch.fft.fftfreq(self.Nx, d=self.dx)  # angular spectrum frequencies
        freq_y = torch.fft.fftfreq(self.Ny, d=self.dy)
        freq_x_c = torch.fft.fftshift(freq_x)
        freq_y_c = torch.fft.fftshift(freq_y)
        f_xx_c, f_yy_c = torch.meshgrid(freq_x_c, freq_y_c, indexing="xy")
        f_tot = 1 / self.wl
        f_zz_c_squared = f_tot**2 - (f_xx_c**2 + f_yy_c**2)
        k0_z = 2 * torch.pi * torch.sqrt(torch.abs(f_zz_c_squared))

        # multiply evanescent modes with complex unit
        k0_z = torch.where(f_zz_c_squared >= 0, k0_z, torch.tensor(1j) * k0_z)
        phase_term = torch.exp(1j * k0_z * self.propag_z)

        # register propagation phase as non-trainable parameter
        self.register_buffer("phase_term", phase_term, persistent=False)

    def forward(self, e_in):
        # add 'learned' phase to incident wavefront
       # learned_phase = torch.exp(-1j * self.phase_weights)
        learned_phase = torch.exp(1j * self.phase_weights)
        e_plus_phase = e_in * learned_phase

        # 2D FFT --> angular spectrum
        fft_e = torch.fft.fft2(e_plus_phase)
        fft_e_c = torch.fft.fftshift(fft_e)

        # 2D iFFT --> field after propagation
        self.phase_term = torch.as_tensor(
            self.phase_term, device=e_in.device
        )  # adapt device
        fft_e_propa_c = torch.fft.ifftshift(fft_e_c * self.phase_term)
        e_out = torch.fft.ifft2(fft_e_propa_c)
        return e_out
    
# %%
import sys

# Add the directory containing the helper module to Python's path
module_path = r'C:\projects\Peter_pytorch_models\diffractive_nn_asm-main\diffractive_nn_asm-main'
if module_path not in sys.path:
    sys.path.append(module_path)

    
def plot_field(field, title="", extent=None, cmap='plasma'):
    # Check if the field is a CuPy array and explicitly convert to a NumPy array if so
    if hasattr(field, 'get'):
        field_np = field.get()  # Convert CuPy array to NumPy array
    else:
        field_np = field  # Assume it's already a NumPy array if not a CuPy array

    plt.figure(figsize=(8, 3))
    if title:
        plt.suptitle(title)

    plt.subplot(121, title="Real part")
    plt.imshow(field_np.real, extent=extent, cmap=cmap)
    plt.colorbar()

    plt.subplot(122, title="Imaginary part")
    plt.imshow(field_np.imag, extent=extent, cmap=cmap)
    plt.colorbar()

    plt.tight_layout()
    plt.show()    

if __name__ == "__main__":
    # for reference calculation
    from helper import sum_of_list
    #from helper import plot_field


    df.set_backend("CUDA")

    # ---- config
    # vacuum wavelength
    wavelength = 532 * 1e-9  # units: meters

    # propagation distance
    Dz_propa = 50 * 1e-6

    # sampling field size
    E0 = 1.0  # initial field amplitude at aperture

    # screen size
    Dx = 50 * 1e-6
    Dy = Dx

    # discretization
    Nx = 1024
    #Ny = Nx
    Ny = 1
    dx = Dx / Nx
    dy = Dy / Ny

    # aperture size
    D_hole = 0.75 * 1e-6

    # --- calculate a reference example using `diffractsim`
    # - setup test wavefront
    # plane wavefront
    F = df.MonochromaticField(
        wavelength=wavelength, extent_x=Dx, extent_y=Dy, Nx=Nx, Ny=Ny, intensity=E0
    )

    # array of circular apertures
    aperture_list = [
        df.CircularAperture(radius=D_hole, x0=xi * 2 * 1e-6, y0=yi * 2 * 1e-6)
        for xi in range(-4, 5)
        for yi in range(-4, 5)
    ]
    aperture_grid = sum_of_list(aperture_list, len(aperture_list) - 1)
    F.add(aperture_grid)

    # add parabolic phase
    F.add(df.Lens(f=1 * 1e-4))

    field_in = F.get_field()

    # propagate and plot
    F.propagate(Dz_propa)
    field_out = F.get_field()

    plot_field(
        field_in, title="wavefront in", extent=(-Dx / 2, Dx / 2, -Dy / 2, Dy / 2)
    )
    plot_field(
        field_out,
        title="wavefront out - reference ({} micron propagation)".format(
            np.round(Dz_propa * 1e6)
        ),
        extent=(-Dx / 2, Dx / 2, -Dy / 2, Dy / 2),
    )

    # --- test diffractive layer
    difflay = _DiffractLayer(
        Nx=Nx, Ny=Ny, Dx=Dx, Dy=Dy, wl=wavelength, propag_z=Dz_propa, init="zeros"
    )

    e_in = torch.as_tensor(field_in)
    e_out = difflay(e_in)

    plot_field(
        e_out.detach().cpu().numpy(),
        title="wavefront out - pytorch ({} micron propagation)".format(
            np.round(Dz_propa * 1e6)
        ),
        extent=(-Dx / 2, Dx / 2, -Dy / 2, Dy / 2),
    )
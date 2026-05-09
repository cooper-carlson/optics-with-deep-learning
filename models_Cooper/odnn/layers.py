"""
torch diffractive layer implementation using ASM

author: P. Wiecha, 02/2024
"""

# %%
import torch
import matplotlib.pyplot as plt
import numpy as np


# learnable phase layer
import torch

class LearnablePhaseLayer1d(torch.nn.Module):

    def __init__(self, Nx, Ny, Dx, Dy, learnable_axis='x', init="zeros"):
        """a grating-type learnable phase layer (1d)
        
         - one dim learnable (--> orientatioin axis)
         - other dim constant phase along size
        
        Args:
            Nx, Ny (int): nr of discretization steps
            Dx, Dy (int): physical extension of the field (in meters)
            learnable_axis (str, optional): along which axis phase can be learned. 
                Along the other axis phase is constant. Defaults to 'x'.
            init (str, optional): how to initialize the learnable phase weights. 
                can be "zeros" or "rnd". Defaults to "zeros".
        """
        super().__init__()

        # which axis is learnable:
        self.learnable_axis = learnable_axis
        
        # 2d field discretization
        self.Nx = Nx
        self.Ny = Ny

        # field size not used here as this is size-agnostic, it applies just a raw phase
        self.Dx = Dx
        self.Dy = Dy
        
        # 1d learnable grating discretization
        if self.learnable_axis.lower() == 'x':
            self.N = self.Nx
        elif self.learnable_axis.lower() == 'y':
            self.N = self.Ny
        else:
            raise ValueError("Learnable axis needs to be either 'x' or 'y'.")

        # initalize the learnable phase
        if init == "zeros":
            self.phase_weights = torch.nn.Parameter(
                torch.zeros(self.Nx, dtype=torch.float32), requires_grad=True
            )
        elif init == "rnd":
            self.phase_weights = torch.nn.Parameter(
                torch.randn(self.Nx, dtype=torch.float32), requires_grad=True
            )

    def forward(self, e_in):
        # add 'learned' phase to incident wavefront
        learned_phase = torch.exp(-1j * self.phase_weights)
        
        # broadcast 1d learnable phase along constant axis
        if self.learnable_axis.lower() == 'x':
            e_plus_phase = e_in * learned_phase.unsqueeze(-2)
        else:  # y
            e_plus_phase = e_in * learned_phase.unsqueeze(-1)
        
        return e_plus_phase

# pytorch diffractive layer for propagation of wavefront by distance `propag_z`
class PropagationLayer(torch.nn.Module):

    def __init__(self, Nx, Ny, Dx, Dy, wl, propag_z, init="zeros"):
        """propagate n input wavefront along `propag_z` meters
        
         - one dim learnable (--> orientatioin axis)
         - other dim constant phase along size
        
        Args:
            Nx, Ny (int): nr of discretization steps
            Dx, Dy (int): physical extension of the field (in meters)
            learnable_axis (str, optional): along which axis phase can be learned. 
                Along the other axis phase is constant. Defaults to 'x'.
            wl (float): wavelength (in meters)
            propag_z (float): propagation distance (in meters)
            init (str, optional): how to initialize the learnable phase weights. 
                can be "zeros" or "rnd". Defaults to "zeros".
        """
        super().__init__()

        # field size / discretization
        self.Nx = Nx
        self.Ny = Ny
        self.Dx = Dx
        self.Dy = Dy
        self.wl = wl
        self.propag_z = propag_z

        self.k0 = torch.as_tensor(2 * torch.pi / self.wl)

        # precalculate fixed propagation phase angular spectrum (fix propagation distance)
        freq_x = torch.fft.fftfreq(self.Nx, d=Dx / Nx)  # angular spectrum frequencies
        freq_y = torch.fft.fftfreq(self.Ny, d=Dy / Ny)
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
        # 2D FFT --> angular spectrum
        fft_e = torch.fft.fft2(e_in)
        fft_e_c = torch.fft.fftshift(fft_e)

        # 2D iFFT --> field after propagation
        self.phase_term = torch.as_tensor(
            self.phase_term, device=e_in.device
        )  # adapt device

        fft_e_propa_c = torch.fft.ifftshift(fft_e_c * self.phase_term)
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
        e_modulated = e_field * torch.exp(-1j * applied_phase)
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
        learned_phase = torch.exp(-1j * self.phase_weights)
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
if __name__ == "__main__":
    # for reference calculation
    from odnn.helper import sum_of_list
    from odnn.helper import plot_field

    import diffractsim as df

    df.set_backend("CPU")

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
    Ny = Nx
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

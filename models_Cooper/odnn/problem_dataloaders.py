"""
torch diffractive neural network implementation

Data loaders for test problems

author: P. Wiecha, 02/2024
"""

# %%

import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import h5py
import numpy as np

# helper: convert images to amplitude modulated complex field
def images_to_cmplxfield_amplitudemodulation(
    images, amplitude_pixel, amplitude_background, img_pix_outsize=None, zoom_factor=1.0
):
    # optional zoom
    if zoom_factor != 1:
        from scipy.ndimage import zoom

        img = zoom(images, (1, zoom_factor, zoom_factor), order=2)
    else:
        img = images

    # set amplitude of image pixels and background
    e_real_part = (amplitude_pixel - amplitude_background) * img + amplitude_background

    # optionally pad zeros (img shape and target shape must be even numbers!)
    if img_pix_outsize is not None:
        Npad = (img_pix_outsize - img.shape[-1]) // 2
        e_real_part = np.pad(
            e_real_part,
            ((0, 0), (Npad, Npad), (Npad, Npad)),
            constant_values=amplitude_background,
        )

    # convert to complex, separte Re/Im channels
    e_cmplx = e_real_part.astype(np.complex64)
    e_field = np.stack([e_cmplx.real, e_cmplx.imag], axis=1)

    return e_field


# helper: convert images to phase modulated complex field
def images_to_cmplxfield_phasemodulation(
    images, pixel_phase, phase_offset, img_pix_outsize=None, zoom_factor=1.0
):
    # optional zoom
    if zoom_factor != 1:
        from scipy.ndimage import zoom

        img = zoom(images, (1, zoom_factor, zoom_factor), order=2)
    else:
        img = images

    # offset
    phase = (pixel_phase - phase_offset) * img + phase_offset

    # optionally pad zeros (img shape and target shape must be even numbers!)
    if img_pix_outsize is not None:
        Npad = (img_pix_outsize - img.shape[-1]) // 2
        phase = np.pad(
            phase,
            ((0, 0), (Npad, Npad), (Npad, Npad)),
            constant_values=phase_offset,
        )

    # convert to complex e-field (to avoid phase jump in prediction), separte Re/Im channels
    e_cmplx = np.exp(-1j * phase)
    e_field = np.stack([e_cmplx.real, e_cmplx.imag], axis=1)

    return e_field


# %% data loader class - MNIST images, used as amplitude-modulation
class ImagingMNISTAmplitudeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        file_path,
        wavelength,
        batch_size=32,
        img_pix_outsize=28,
        zoom_factor_in=1.0,
        zoom_factor_out=1.0,
        E0_pixel=1,
        E0_background=0.0,
        N_load=-1,
        test_size=0.05,
        random_state=2,
    ):
        # --- implement an image-to-image task
        #   the goal is that the diffractive network reproduces the input image
        #   at the output, hence to act as a kind of lens system
        super().__init__()
        self.file_path = file_path
        self.N_load = N_load
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state

        self.img_pix_outsize = img_pix_outsize
        self.zoom_factor_in = zoom_factor_in
        self.zoom_factor_out = zoom_factor_out
        self.E0_pixel = E0_pixel
        self.E0_background = E0_background
        self.wavelength = wavelength
        self.k0 = 2 * np.pi / self.wavelength

    def setup(self, stage=None):
        # Load data
        print("loading dataset '{}'... ".format(self.file_path), end="")
        with h5py.File(self.file_path, "r") as f_read:
            # grayscale images, dim: Nx28x28
            images = np.array(
                f_read["inputs"][0, : self.N_load, ..., 0], dtype=np.float32
            )  # remove channel dim

            # one hot encoding of mnist classes, dim: Nx10
            classes = np.array(f_read["targets"][0, : self.N_load], dtype=np.float32)
        print("done")

        # --- now we modify the data for our phase-modulation imaging problem:
        e_field_in = images_to_cmplxfield_amplitudemodulation(
            images=images,
            amplitude_pixel=self.E0_pixel,
            amplitude_background=self.E0_background,
            img_pix_outsize=self.img_pix_outsize,
            zoom_factor=self.zoom_factor_in,
        )
        e_field_out = images_to_cmplxfield_amplitudemodulation(
            images=images,
            amplitude_pixel=self.E0_pixel,
            amplitude_background=self.E0_background,
            img_pix_outsize=self.img_pix_outsize,
            zoom_factor=self.zoom_factor_out,
        )

        # Split data into training and validation sets
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            e_field_in,
            e_field_out,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    def _dataloader(self, x, y, shuffle=True):
        # imaging: in- and output are identical
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x), torch.from_numpy(y)
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle
        )

    def train_dataloader(self):
        return self._dataloader(self.x_train, self.y_train)

    def val_dataloader(self):
        return self._dataloader(self.x_val, self.y_val, shuffle=False)

# %% data loader class - MNIST images, used as phase-modulation
class ImagingMNISTPhaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        file_path,
        wavelength,
        batch_size=32,
        img_pix_outsize=28,
        zoom_factor_in=1.0,
        zoom_factor_out=1.0,
        pixel_phase=np.pi,
        phase_offset=0.0,
        N_load=-1,
        test_size=0.05,
        random_state=2,
    ):
        # --- implement an image-to-image task
        #   the goal is that the diffractive network reproduces the input image
        #   at the output, hence to act as a kind of lens system
        super().__init__()
        self.file_path = file_path
        self.N_load = N_load
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state

        self.img_pix_outsize = img_pix_outsize
        self.zoom_factor_in = zoom_factor_in
        self.zoom_factor_out = zoom_factor_out
        self.pixel_phase = pixel_phase
        self.phase_offset = phase_offset
        self.wavelength = wavelength
        self.k0 = 2 * np.pi / self.wavelength

    def setup(self, stage=None):
        # Load data
        print("loading dataset '{}'... ".format(self.file_path), end="")
        with h5py.File(self.file_path, "r") as f_read:
            # grayscale images, dim: Nx28x28
            images = np.array(
                f_read["inputs"][0, : self.N_load, ..., 0], dtype=np.float32
            )  # remove channel dim

            # one hot encoding of mnist classes, dim: Nx10
            classes = np.array(f_read["targets"][0, : self.N_load], dtype=np.float32)
        print("done")

        # --- now we modify the data for our phase-modulation imaging problem:
        e_field_in = images_to_cmplxfield_phasemodulation(
            images=images,
            pixel_phase=self.pixel_phase,
            phase_offset=self.phase_offset,
            img_pix_outsize=self.img_pix_outsize,
            zoom_factor=self.zoom_factor_in,
        )
        e_field_out = images_to_cmplxfield_phasemodulation(
            images=images,
            pixel_phase=self.pixel_phase,
            phase_offset=self.phase_offset,
            img_pix_outsize=self.img_pix_outsize,
            zoom_factor=self.zoom_factor_out,
        )

        # Split data into training and validation sets
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            e_field_in,
            e_field_out,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    def _dataloader(self, x, y, shuffle=True):
        # imaging: in- and output are identical
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x), torch.from_numpy(y)
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle
        )

    def train_dataloader(self):
        return self._dataloader(self.x_train, self.y_train)

    def val_dataloader(self):
        return self._dataloader(self.x_val, self.y_val, shuffle=False)


# %% data loader class - XOR coded with phase-modulated gaussians
class logicGateXORPhaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        wavelength,
        batch_size=4,
        img_pix_size=64,
        Ngauss=240,
        wgauss=12,
        offsetgauss_x=48,
        offsetgauss_y=0,
        pixel_phase=np.pi,
        phase_offset=0.0,
        noise_amplitude=1e-3,
        dataset=None
    ):
        # --- implement an image-to-image task
        #   the goal is that the diffractive network reproduces the input image
        #   at the output, hence to act as a kind of lens system
        super().__init__()
        self.batch_size = batch_size

        self.Nxy = img_pix_size
        self.Ngauss = Ngauss
        self.wgauss = wgauss
        self.offsetgauss_x = offsetgauss_x
        self.offsetgauss_y = offsetgauss_y

        self.pixel_phase = pixel_phase
        self.phase_offset = phase_offset
        self.wavelength = wavelength
        self.k0 = 2 * np.pi / self.wavelength

        self.noise_amplitude = noise_amplitude

        self.dataset = dataset

    def setup(self, stage=None):
        from scipy import signal

        def gen_OAM_mode_2d(imgsize, width, l, n):
            x = np.linspace(-imgsize // 2, imgsize // 2 - 1, imgsize)
            y = np.linspace(-imgsize // 2, imgsize // 2 - 1, imgsize)
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X**2 + Y**2)
            Phi = np.arctan2(Y, X)
            # Laguerre polynomial
            # from scipy.special import genlaguerre

            # L = genlaguerre(n, np.abs(l))
            
            # # mode field
            # mode = (
            #     (R * np.sqrt(2) / width) ** np.abs(l)
            #     * L(2 * R**2 / width**2)
            #     * np.exp(-R**2 / width**2)
            #     * np.exp(1j * l * Phi)
            # )

            w0 = width / np.sqrt(abs(l) + 1)

            r_safe = np.where(R == 0, 1e-12, R)
            t_rad = abs(l) * np.log(r_safe) - (R**2 / w0**2)

            amp = np.exp(t_rad)
            A = amp / np.max(amp)

            F = l * Phi - (np.pi * A)
            phase = np.mod(F, 2 * np.pi)

            mode = A * np.exp(1j * phase)

            return mode.astype(np.complex64)
    
        import numpy as np

        def paste_center(big, small, offset_y=0, offset_x=0):
            out = big.copy()

            # centres of each array
            cy_big, cx_big = np.array(big.shape) // 2
            cy_sml, cx_sml = np.array(small.shape) // 2

            # top-left corner in big where small[0,0] should go
            y0_big = cy_big - cy_sml + offset_y
            x0_big = cx_big - cx_sml + offset_x

            # clip to valid ranges
            y0_sml = max(0, -y0_big)
            x0_sml = max(0, -x0_big)
            y1_sml = min(small.shape[0], big.shape[0] - y0_big)
            x1_sml = min(small.shape[1], big.shape[1] - x0_big)

            y0_big = max(0, y0_big)
            x0_big = max(0, x0_big)
            y1_big = y0_big + (y1_sml - y0_sml)
            x1_big = x0_big + (x1_sml - x0_sml)

            out[y0_big:y1_big, x0_big:x1_big] += small[y0_sml:y1_sml, x0_sml:x1_sml]
            return out

        def gen_double_OAM_2d(channels12):

            # create the two OAM mode patterns
            oam1 = gen_OAM_mode_2d(self.Ngauss, self.wgauss,
                                l=channels12[0], n=channels12[1])
            oam2 = gen_OAM_mode_2d(self.Ngauss, self.wgauss,
                                l=channels12[2], n=channels12[3])

            full = np.zeros((self.Nxy, self.Nxy), dtype=np.complex64)

            # place first mode offset to the left (negative x offset)
            full = paste_center(full, oam1, offset_y=self.offsetgauss_y,
                                offset_x=-self.offsetgauss_x)
            # place second mode offset to the right (positive x offset)
            full = paste_center(full, oam2, offset_y=self.offsetgauss_y,
                                offset_x=+self.offsetgauss_x)
            
            # full += 1e-5  # avoid zeros

            return full
        
        def gen_gaussian_2d(imgsize, width, phase_scale=np.pi):
            g1d = signal.windows.gaussian(imgsize, std=width).reshape(imgsize, 1)
            g2d = np.outer(g1d, g1d)
            # g2d = g2d / np.max(g2d)  # normalize max to 1

            phase = phase_scale * g2d

            efield = np.exp(1j * phase).astype(np.complex64)

            return efield

        def gen_double_gauss_2d(channels12, phase_scale=np.pi):
            gauss_small = gen_gaussian_2d(self.Ngauss, self.wgauss, phase_scale=phase_scale)
            full = np.zeros((self.Nxy, self.Nxy), dtype=np.complex64)

            # left Gaussian (negative x offset)
            if channels12[0]:
                full = paste_center(full, gauss_small,
                                    offset_y=0, offset_x=-self.offsetgauss_x)

            # right Gaussian (positive x offset)
            if channels12[1]:
                full = paste_center(full, gauss_small,
                                    offset_y=0, offset_x=+self.offsetgauss_x)

            return full

        # implement XOR

        efields_in = []
        efields_out = []
        for sample in self.dataset:
            in12 = sample[0]
            out12 = sample[1]

            e_in_clean = gen_double_OAM_2d(in12)
            # e_out_clean = gen_double_gauss_2d(out12, phase_scale=self.pixel_phase)
            e_out_clean = gen_double_OAM_2d(out12)

            noise = (np.random.normal(scale=self.noise_amplitude, size=e_in_clean.shape) +
                     1j * np.random.normal(scale=self.noise_amplitude, size=e_in_clean.shape)).astype(np.complex64)
            e_in_noisy = e_in_clean + noise
            
            e_in = np.stack([e_in_noisy.real, e_in_noisy.imag], axis=0)
            e_out = np.stack([e_out_clean.real, e_out_clean.imag], axis=0)

            efields_in.append(e_in)
            efields_out.append(e_out)
        
        self.x_train = self.x_val = np.array(efields_in, dtype=np.float32)
        self.y_train = self.y_val = np.array(efields_out, dtype=np.float32)

    def _dataloader(self, x, y, shuffle=True):
        # imaging: in- and output are identical

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x), torch.from_numpy(y)
        )

        def collate_with_noise(batch):
            x_batch, y_batch = zip(*batch)
            x_batch = torch.stack(x_batch)
            y_batch = torch.stack(y_batch)
            noise_real = torch.randn_like(x_batch[:, 0]) * self.noise_amplitude
            noise_imag = torch.randn_like(x_batch[:, 1]) * self.noise_amplitude
            x_batch = torch.stack([x_batch[:, 0] + noise_real, x_batch[:, 1] + noise_imag], dim=1)
            return x_batch, y_batch
        
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=collate_with_noise
        )

    def train_dataloader(self):
        return self._dataloader(self.x_train, self.y_train)

    def val_dataloader(self):
        return self._dataloader(self.x_val, self.y_val, shuffle=False)

class constDataModule(pl.LightningDataModule):
    def __init__(
        self,
        wavelength,
        batch_size=2,
        img_pix_size=64,
        dataset=None,
        amplitude=1.0,
        gauss_width=None,
        test_size=0.0,
        random_state=2,
        shuffle_train=True,
        phase_function = None,
    ):
        """
        Data module for uniform-phase Gaussian-beam fields.

        Each entry in `dataset` is interpreted as a scalar phase value phi (radians).
        A sample is converted into a centered NxN complex field

            E(x, y) = A(x, y) * exp(-1j * phi)

        where A(x, y) is a 2D Gaussian envelope.

        Target output is identical to input.

        Args:
            wavelength (float): Wavelength in meters.
            batch_size (int): Batch size for train/val loaders.
            img_pix_size (int): Grid size N so each field is N x N.
            dataset (array-like): Iterable of scalar phase values in radians.
            amplitude (float): Peak amplitude of the Gaussian beam.
            gauss_width (float or None): Gaussian std in pixels. If None, uses N/6.
            test_size (float): Fraction for validation split. If <= 0, use full set for both.
            random_state (int): Random seed for optional split.
            shuffle_train (bool): Whether to shuffle training loader.
        """
        super().__init__()

        self.batch_size = batch_size
        self.Nxy = img_pix_size
        self.wavelength = wavelength
        self.k0 = 2 * np.pi / self.wavelength

        self.dataset = dataset if dataset is not None else []
        self.amplitude = amplitude
        self.gauss_width = gauss_width if gauss_width is not None else self.Nxy / 6.0

        self.test_size = test_size
        self.random_state = random_state
        self.shuffle_train = shuffle_train

        self.phase_values = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.phase_function = phase_function if phase_function is not None else (lambda phi: phi)

    def setup(self, stage=None):
        from scipy import signal

        def gen_gaussian_amplitude_2d():
            g1d = signal.windows.gaussian(self.Nxy, std=self.gauss_width).astype(np.float32)
            g2d = np.outer(g1d, g1d).astype(np.float32)
            g2d /= np.max(g2d)
            g2d *= self.amplitude
            return g2d

        def gen_const_phase_gaussian_field(phi):
            amp = gen_gaussian_amplitude_2d()
            efield = amp * np.exp(-1j * phi)
            return efield.astype(np.complex64)

        phase_values = np.asarray(self.dataset, dtype=np.float32).reshape(-1)
        if phase_values.size == 0:
            raise ValueError("constDataModule requires a non-empty `dataset` of phase values.")

        efields_in = []
        efields_out = []

        for phi_in in phase_values:
            phi_out = self.phase_function(phi_in)

            e_in_c = gen_const_phase_gaussian_field(phi_in)
            e_out_c = gen_const_phase_gaussian_field(phi_out)

            e_in = np.stack([e_in_c.real, e_in_c.imag], axis=0).astype(np.float32)
            e_out = np.stack([e_out_c.real, e_out_c.imag], axis=0).astype(np.float32)

            efields_in.append(e_in)
            efields_out.append(e_out)

        efields_in = np.asarray(efields_in, dtype=np.float32)
        efields_out = np.asarray(efields_out, dtype=np.float32)

        self.phase_values = phase_values

        if self.test_size is None or self.test_size <= 0:
            self.x_train = self.x_val = efields_in
            self.y_train = self.y_val = efields_out
        else:
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
                efields_in,
                efields_out,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=True,
            )

    def _dataloader(self, x, y, shuffle=True):
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y),
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._dataloader(self.x_train, self.y_train, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return self._dataloader(self.x_val, self.y_val, shuffle=False)

# testing
if __name__ == "__main__":
    wavelength = 532 * 1e-9
    Nx = 64
    
    # XOR Dataset test
    data_module = logicGateXORPhaseDataModule(
        wavelength=wavelength,
        img_pix_size=Nx,
        pixel_phase=np.pi / 2,
        phase_offset=-np.pi / 2,
        batch_size=4,
    )
    data_module.setup()

    # # MNIST Dataset test
    # data_module = ImagingMNISTPhaseDataModule(
    #     "data/mnist_dataset.h5",
    #     wavelength=wavelength,
    #     img_pix_outsize=Nx,
    #     pixel_phase=np.pi / 2,
    #     zoom_factor_in=1,
    #     zoom_factor_out=2,
    #     phase_offset=0,
    #     batch_size=32,
    #     N_load=2000,  # partial data loading for rapid testing
    # )
    # data_module.setup()


    # evaluate and plot a sample
    val_loader = data_module.val_dataloader()
    x_val = []
    y_val = []
    for i, batch in enumerate(val_loader):
        _x, _y = batch

        x_val.append(_x.detach().cpu().numpy())
        y_val.append(_y.detach().cpu().numpy())

    if len(x_val) > 1:
        x_val = np.concatenate(x_val)
        y_val = np.concatenate(y_val)
    else:
        x_val = np.array(x_val[0])
        y_val = np.array(y_val[0])
    
    
    idx = 2
    plt.figure(figsize=(8,3))
    plt.subplot(121)
    plt.imshow(np.angle(x_val[idx, 0]+1j*x_val[idx, 1]))
    plt.colorbar()
    plt.clim(-np.pi, np.pi)
    
    plt.subplot(122)
    plt.imshow(np.angle(y_val[idx, 0]+1j*y_val[idx, 1]))
    plt.colorbar()
    plt.clim(-np.pi, np.pi)
    plt.show()

# %%
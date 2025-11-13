# shear_correction.py

import math

import torch

torch.backends.cuda.matmul.allow_tf32 = False

# ============================================================
# Core: DFT registration (ported 1:1 from your Julia versions)
# ============================================================

import math
from typing import Tuple

from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp
import numpy as np


@torch.no_grad()
def dftreg_subpix_gpu(
    img1_f_g: torch.Tensor,                 # complex64 (H, W) FFT(I1)
    img2_f_g: torch.Tensor,                 # complex64 (H, W) FFT(I2)
    CC2x_g: torch.Tensor,                   # complex64 (2H, 2W) scratch
    up_fac: int = 10,
) -> Tuple[float, Tuple[float, float], float]:
    """
    Faithful port of the Julia function:
      dftreg_subpix_gpu!(img1_f_g, img2_f_g, CC2x_g, up_fac=10)

    Returns:
      error: float
      shift: (dy, dx) in pixels (Python 0-based convention)
      diffphase: float
    """
    device = img1_f_g.device
    dtypec = img1_f_g.dtype
    H, W = img1_f_g.shape

    # --------------------------
    # Initial estimate via 2x upsampled correlation (center-embed)
    # --------------------------
    CC2x_g.zero_()
    prod_shifted = torch.fft.fftshift(img1_f_g) * torch.conj(torch.fft.fftshift(img2_f_g))
    r0 = H - (H // 2)
    c0 = W - (W // 2)
    CC2x_g[r0:r0 + H, c0:c0 + W].copy_(prod_shifted)

    CC2x_spatial = torch.fft.ifft2(torch.fft.ifftshift(CC2x_g))
    loc_flat = torch.argmax(torch.abs(CC2x_spatial))
    r_loc, c_loc = torch.unravel_index(loc_flat, CC2x_spatial.shape)
    CC2x_max = CC2x_spatial[r_loc, c_loc]

    # Map 2x-grid argmax to original grid shift (Julia 1-based -> Python 0-based)
    def _map_shift(idx: int, dim: int) -> float:
        return (idx - 2 * dim) / 2.0 if idx >= dim else (idx / 2.0)

    shift_y = _map_shift(int(r_loc), H)
    shift_x = _map_shift(int(c_loc), W)
    shift = torch.tensor([shift_y, shift_x], dtype=torch.float32, device=device)

    # --------------------------
    # Subpixel refinement (matrix-multiply DFT) if requested
    # --------------------------
    ind2H, ind2W = H, W   # Julia: ind2 = size(CC2x_g) ./ 2 == (H, W)

    if up_fac > 2:
        # round initial estimate to 1/up_fac grid
        shift = torch.round(shift * up_fac) / float(up_fac)
        dft_shift = math.ceil(up_fac * 1.5) / 2.0  # center of output at dft_shift + 1 (Julia 1-based)

        # CC_refine around current estimate; NOTE: order img2 * conj(img1) matches Julia
        Fcp = img2_f_g * torch.conj(img1_f_g)
        outsz = int(math.ceil(up_fac * 1.5))
        roff = dft_shift - shift[0] * up_fac
        coff = dft_shift - shift[1] * up_fac

        # CC_refine = dftups(Fcp, outsz, outsz, up_fac, roff, coff) / (float(ind2H * ind2W) * (up_fac ** 2))
        # (optional) run only the dftups branch in complex128, then cast back:
        Fcp = (img2_f_g * torch.conj(img1_f_g)).to(torch.complex128)
        CC_refine = dftups(Fcp, outsz, outsz, up_fac, roff, coff) / (H*W*up_fac**2)
        CC_refine = CC_refine.to(torch.complex64)

        # locate max and map back to original grid
        loc_flat = torch.argmax(torch.abs(CC_refine))
        rr, cc = torch.unravel_index(loc_flat, CC_refine.shape)
        CC_refine_max = CC_refine[rr, cc]

        # Julia: locI = locI .- dft_shift .- 1  (1-based)
        # Python (0-based): rr,cc - dft_shift
        loc_vec = torch.tensor([rr, cc], dtype=torch.float32, device=device) - float(dft_shift)
        shift = shift + (loc_vec / float(up_fac))

        # power terms at (0,0), normalized identically to Julia
        P1 = img1_f_g * torch.conj(img1_f_g)
        P2 = img2_f_g * torch.conj(img2_f_g)
        img1_00 = dftups(P1, 1, 1, up_fac, 0.0, 0.0)[0, 0] / (float(ind2H * ind2W) * (up_fac ** 2))
        img2_00 = dftups(P2, 1, 1, up_fac, 0.0, 0.0)[0, 0] / (float(ind2H * ind2W) * (up_fac ** 2))
        CC_max = CC_refine_max
    else:
        # Coarse-only normalization (uses size(CC2x_g))
        img1_00 = torch.sum(img1_f_g * torch.conj(img1_f_g)) / float((2 * H) * (2 * W))
        img2_00 = torch.sum(img2_f_g * torch.conj(img2_f_g)) / float((2 * H) * (2 * W))
        CC_max = CC2x_max

    # --------------------------
    # Error and diffphase
    # --------------------------
    numer = CC_max * torch.conj(CC_max)
    denom = img1_00 * img2_00
    error = torch.sqrt(torch.abs(1.0 - (numer / denom))).real.item()
    diffphase = math.atan2(CC_max.imag.item(), CC_max.real.item())

    return float(error), (float(shift[0].item()), float(shift[1].item())), float(diffphase)


def dftups(
    F: torch.Tensor, nor: int, noc: int, usfac: int, roff: float, coff: float
) -> torch.Tensor:
    """
    Matrix-multiply upsampled DFT (Guizar-Sicairos) with correct signs/centering.

    Args:
      F:     complex64 (nr, nc) frequency-domain input (e.g., FT2 .* conj(FT1))
      nor:   rows of upsampled region
      noc:   cols of upsampled region
      usfac: upsampling factor
      roff:  row offset of the region center (Julia dft_shift - shift_y*usfac)
      coff:  col offset of the region center (Julia dft_shift - shift_x*usfac)

    Returns:
      complex64 (nor, noc)
    """
    device = F.device
    dtypec = F.dtype
    nr, nc = F.shape

    # centered integer frequency indices, reordered with ifftshift (matches Julia/FFT layout)
    Nr_vals = torch.arange(-math.floor(nr / 2), math.ceil(nr / 2), device=device, dtype=torch.float32)
    Nc_vals = torch.arange(-math.floor(nc / 2), math.ceil(nc / 2), device=device, dtype=torch.float32)
    Nr = torch.fft.ifftshift(Nr_vals)  # (nr,)
    Nc = torch.fft.ifftshift(Nc_vals)  # (nc,)

    # upsampled coordinate grids (0..nor-1 minus offset), (0..noc-1 minus offset)
    r = torch.arange(nor, device=device, dtype=torch.float32) - float(roff)   # (nor,)
    c = torch.arange(noc, device=device, dtype=torch.float32) - float(coff)   # (noc,)

    # kernels (note the NEGATIVE sign in the exponent, as in Guizar-Sicairos)
    twopi = 2.0 * math.pi
    kernr = torch.exp(
        (-1j * twopi / (nr * usfac))
        * (r.reshape(nor, 1) @ Nr.reshape(1, nr)),
    ).to(dtypec)  # (nor, nr)

    kernc = torch.exp(
        (-1j * twopi / (nc * usfac))
        * (Nc.reshape(nc, 1) @ c.reshape(1, noc)),
    ).to(dtypec)  # (nc, noc)

    # upsampled region
    return (kernr @ F) @ kernc

@torch.no_grad()
def dftreg_resample_gpu(
    img_f_g: torch.Tensor,
    N_g: torch.Tensor,  # scratch (ignored; kept for signature parity)
    shift: Tuple[float, float],
    diffphase: float,
) -> torch.Tensor:
    """
    Port of:
      dftreg_resample_gpu!(img_f_g, N_g, shift, diffphase) = real(ifft(subpix_shift_gpu!(...)))

    Returns:
      float32 real image on CUDA of shape (H, W)
    """
    F_shifted = _subpix_shift_in_fourier(img_f_g, shift, diffphase)
    out = torch.fft.ifft2(F_shifted).real.to(torch.float32)
    return out


def _subpix_shift_in_fourier(
    F: torch.Tensor,
    shift: Tuple[float, float],
    diffphase: float,
) -> torch.Tensor:
    """
    Apply spatial-domain shift (dy, dx) and global phase in Fourier domain.
    """
    device = F.device
    H, W = F.shape
    dy, dx = float(shift[0]), float(shift[1])

    # torch.fft.fftfreq gives frequencies in cycles/sample: [0, 1/N, ..., -1/2]
    fy = torch.fft.fftfreq(H, device=device).reshape(H, 1)  # (H,1)
    fx = torch.fft.fftfreq(W, device=device).reshape(1, W)  # (1,W)

    twopi = 2.0 * math.pi
    phase_ramp = torch.exp(-1j * twopi * (dy * fy + dx * fx))
    global_phase = torch.exp(1j * torch.tensor(diffphase, device=device))
    return F * phase_ramp * global_phase


# ============================================================
# Your two pipeline functions (unchanged API, CUDA-first)
# ============================================================

@torch.no_grad()
def reg_stack_translate(
    img_stack_reg_g: torch.Tensor,
    img1_f_g: torch.Tensor,
    img2_f_g: torch.Tensor,
    CC2x_g: torch.Tensor,
    N_g: torch.Tensor,
    *,
    reg_param,
) -> None:
    """
    In-place stack registration (Z by translation), mirrors your Julia logic.

    Tensors:
      img_stack_reg_g: float32 (H, W, Z) on CUDA
      img1_f_g/img2_f_g: complex64 (H, W) on CUDA
      CC2x_g: complex64 (2H, 2W) on CUDA
      N_g: float32 (H, W) on CUDA  (scratch)
    reg_param:
      dict z -> (error:float, (dy:float, dx:float), diffphase:float)
    """
    assert img_stack_reg_g.is_cuda and img1_f_g.is_cuda and img2_f_g.is_cuda
    assert CC2x_g.is_cuda and N_g.is_cuda

    Z, H, W = img_stack_reg_g.shape

    CC2x_g.zero_()
    N_g.zero_()
    # print(reg_param)
    for z in range(1, Z): # I made this 0 which should have broken it but didn't?
        z1, z2 = z - 1, z
        img1_g = img_stack_reg_g[z1, :, :]
        img2_g = img_stack_reg_g[z2, :, :]

        img1_f_g.copy_(torch.fft.fft2(img1_g).to(torch.complex64))
        img2_f_g.copy_(torch.fft.fft2(img2_g).to(torch.complex64))

        if z not in reg_param:
            error, shift, diffphase = dftreg_subpix_gpu(img1_f_g, img2_f_g, CC2x_g)
            reg_param[z] = (error, (shift[0], shift[1]), diffphase)
        else:
            error, shift, diffphase = reg_param[z]

        out_z = dftreg_resample_gpu(img2_f_g, N_g, shift, diffphase)
        img_stack_reg_g[z, :, :].copy_(out_z)

        CC2x_g.zero_()
        N_g.zero_()


# @torch.no_grad()
# def shear_correction_torch(volume_torch: torch.Tensor, shear_params_dict = {}) -> torch.Tensor:
#     """
#     CUDA-first port of the ANTSUN Julia shear correction routine.
#     shear_params_dict -  # Need to have this to apply to green. Maybe is possible to run them both together?
#      COuld also consinder using a tensor instead of a dict for speed but this is fine for now.
#     """
#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA is required to mirror the CuArray workflow.")

#     assert volume_torch.is_cuda, "Input volume must be on CUDA device"
#     assert len(volume_torch.shape) == 3, "Input volume must be 3D (H, W, Z). Batches not supported."
#     H, W, Z = volume_torch.shape

#     # device = torch.device("cuda")
#     device = volume_torch.device

#     img1_f_g = torch.empty((H, W), dtype=torch.complex64, device=device)
#     img2_f_g = torch.empty((H, W), dtype=torch.complex64, device=device)
#     CC2x_g   = torch.empty((2 * H, 2 * W), dtype=torch.complex64, device=device)
#     N_g      = torch.empty((H, W), dtype=torch.float32, device=device)
#     # img_stack_reg_g = torch.empty((H, W, Z), dtype=torch.float32, device=device)


#     # Register stack in place
#     reg_stack_translate(
#         volume_torch, img1_f_g, img2_f_g, CC2x_g, N_g,
#         reg_param=shear_params_dict,
#     )

#     # cleanup
#     del img1_f_g, img2_f_g, CC2x_g, N_g
#     torch.cuda.empty_cache()

#     return volume_torch, shear_params_dict
    # return img_i32

@torch.no_grad()
def shear_correction_torch(volume_torch: torch.Tensor, shear_params_dict = {}) -> torch.Tensor:
    """
    CUDA-first port of the ANTSUN Julia shear correction routine.
    shear_params_dict -  # Need to have this to apply to green. Maybe is possible to run them both together?
     COuld also consinder using a tensor instead of a dict for speed but this is fine for now.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to mirror the CuArray workflow.")

    assert volume_torch.is_cuda, "Input volume must be on CUDA device"
    assert len(volume_torch.shape) == 3, "Input volume must be 3D (H, W, Z). Batches not supported."
    Z, H, W = volume_torch.shape

    # device = torch.device("cuda")
    device = volume_torch.device

    img1_f_g = torch.empty((H, W), dtype=torch.complex64, device=device)
    img2_f_g = torch.empty((H, W), dtype=torch.complex64, device=device)
    CC2x_g   = torch.empty((2 * H, 2 * W), dtype=torch.complex64, device=device)
    N_g      = torch.empty((H, W), dtype=torch.float32, device=device)
    # img_stack_reg_g = torch.empty((H, W, Z), dtype=torch.float32, device=device)


    # Register stack in place
    reg_stack_translate(
        volume_torch, img1_f_g, img2_f_g, CC2x_g, N_g,
        reg_param=shear_params_dict,
    )

    # cleanup
    del img1_f_g, img2_f_g, CC2x_g, N_g
    torch.cuda.empty_cache()

    return volume_torch, shear_params_dict


@torch.no_grad()
# def shear_correction_scikit(volume_torch: torch.Tensor, shear_params_dict = {}) -> torch.Tensor:
def shear_correction_scikit(volume_np: np.array, shear_params_dict = {}) -> torch.Tensor:
    """
    CUDA-first port of the ANTSUN Julia shear correction routine.
    shear_params_dict -  # Need to have this to apply to green. Maybe is possible to run them both together?
     COuld also consinder using a tensor instead of a dict for speed but this is fine for now.

     [Z, H, W]
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to mirror the CuArray workflow.")

    assert len(volume_np.shape) == 3, "Input volume must be 3D (Z, H, W). Batches not supported."
    Z, H, W = volume_np.shape

    # device = torch.device("cuda")

    img1_f_g = np.empty((H, W))
    img2_f_g = np.empty((H, W))

    for z in range(1, Z): # I made this 0 which should have broken it but didn't?
        z1, z2 = z - 1, z
        img1_g = volume_np[z1, :, :]
        img2_g = volume_np[z2, :, :]

        if z not in shear_params_dict:
            shift, error, diffphase = phase_cross_correlation(img1_g, img2_g, upsample_factor=10)
            shear_params_dict[z] = (error, (shift[0], shift[1]), diffphase)
        else:
            error, shift, diffphase = shear_params_dict[z]

        # out_z = dftreg_resample_gpu(img2_f_g, N_g, shift, diffphase)
        tform = AffineTransform(translation=(-shift[1], -shift[0]))  # negative to move moving->reference
        aligned_Z = warp(img2_g, tform, preserve_range=True)
        volume_np[z, :, :] = aligned_Z

    return volume_np, shear_params_dict
    # return img_i32

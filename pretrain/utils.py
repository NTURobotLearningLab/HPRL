import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import wandb

import logging
logging.getLogger('matplotlib.font_manager').disabled = True


def analyze_z_bz(z, bz, save_path="pca_z_bz.png"):
    # print("GOOGOOGOO!!!~~~~")
    if not isinstance(z, torch.Tensor):
        z_tensor = torch.tensor(z, dtype=torch.float32)
        bz_tensor = torch.tensor(bz, dtype=torch.float32)
    else:
        z_tensor = z
        bz_tensor = bz

    # Merge first two dimensions: [66, 128, 64] -> [8448, 64]
    z_flat = z_tensor.view(-1, z_tensor.size(-1))
    bz_flat = bz_tensor.view(-1, bz_tensor.size(-1))

    # Compute L2 norms
    z_norm = torch.norm(z_flat, dim=1)
    bz_norm = torch.norm(bz_flat, dim=1)

    # Cosine similarity
    cosine_sim = F.cosine_similarity(z_flat, bz_flat, dim=1)
    angles_rad = torch.acos(torch.clamp(cosine_sim, -1.0, 1.0))
    angles_deg = angles_rad.mean().item() * 180 / torch.pi

    # Scale (length ratio)
    scale_ratio = z_norm / (bz_norm + 1e-8)

    # Summary statistics
    print(f"Z norm mean/std: {z_norm.mean():.4f} / {z_norm.std():.4f}")
    print(f"bZ norm mean/std: {bz_norm.mean():.4f} / {bz_norm.std():.4f}")
    print(f"Mean angle (deg): {angles_deg:.4f}")
    print(f"Mean scale ratio: {scale_ratio.mean():.4f}")

    # wandb.log({
    #     'eval/z_norm_mean': z_norm.mean(),
    #     'eval/z_norm_std': z_norm.std(),
    #     'eval/bz_norm_mean': bz_norm.mean(),
    #     'eval/bz_norm_std': bz_norm.std(),
    #     'eval/angle_rad_mean': angles_rad.mean(),
    #     'eval/angle(deg)': angles_rad.mean().item() * 180 / torch.pi,
    #     'eval/scale_ratio_mean': scale_ratio.mean()
    # })

    # Mix into a pool and do PCA
    # pool = torch.cat([z_flat, bz_flat], dim=0).numpy()  # shape: (2 * 8448, 64)
    # labels = np.array([0] * z_flat.size(0) + [1] * bz_flat.size(0))  # 0: z, 1: bz

    # pca = PCA(n_components=2)
    # reduced = pca.fit_transform(pool)  # shape: (16896, 2)

    # # Plot and save
    # plt.figure(figsize=(8, 6))
    # plt.scatter(reduced[labels == 0, 0], reduced[labels == 0, 1], alpha=0.5, label='z')
    # plt.scatter(reduced[labels == 1, 0], reduced[labels == 1, 1], alpha=0.5, label='bz')
    # plt.title("PCA projection of z and bz vectors")
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()

    # # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.savefig(save_path)
    # plt.close()

    return {
        'z_norm': z_norm.mean(),
        'bz_norm': bz_norm.mean(),
        'angle_rad': angles_rad,
        'angle_deg': angles_deg,
        'scale_ratio': scale_ratio.mean(),

        # 'pca_components': reduced
    }

# import torch
# import torch.nn.functional as F

# def convert_to_POMDP(s_h):
#     """
#     Extract 3x3 local grid centered on Karel for each state.
    
#     Input:
#         s_h: [R, T, C=8, H=8, W=8]
#     Output:
#         [R, T, C=8, 3, 3]
#     """
#     R, T, C, H, W = s_h.shape
#     assert (H, W, C) == (8, 8, 8), "Expected shape [R, T, 8, 8, 8] in CHW layout"

#     # Pad spatial dimensions by 1 for safe 3x3 extraction
#     s_h_padded = F.pad(s_h, pad=(1, 1, 1, 1))  # [R, T, C, 10, 10]

#     # Find Karel position per [R, T] — only one per frame in channels 0–3
#     karel_mask = s_h[:, :, 0:4, :, :].sum(dim=2)  # [R, T, H, W]
#     karel_mask_flat = karel_mask.view(R * T, -1)
#     karel_indices = karel_mask_flat.argmax(dim=1)  # [R*T]
#     y = (karel_indices // W + 1).long()  # +1 due to padding
#     x = (karel_indices % W + 1).long()
#     print(f"y: {y}, x: {x}")

#     # Use unfold to efficiently extract 3x3 patches from s_h_padded
#     patches = F.unfold(
#         s_h_padded.view(R * T, C, 10, 10), kernel_size=3, padding=0
#     )  # [R*T, C*3*3, L], where L=64 (sliding over all 8x8 positions)

#     # Find the linear index into 8x8 grid for each Karel location
#     unfold_idx = (y - 1) * 8 + (x - 1)  # subtract 1 because unfold slides on original 8x8

#     # Gather relevant patches
#     idx = unfold_idx.view(-1, 1).expand(-1, C * 3 * 3)  # [R*T, C*9]
#     selected = patches.gather(dim=2, index=idx.unsqueeze(-1)).squeeze(-1)  # [R*T, C*9]

#     # Reshape back to [R, T, C, 3, 3]
#     result = selected.view(R, T, C, 3, 3)
#     return result




import torch
import torch.nn.functional as F

def convert_to_POMDP(s_h: torch.Tensor) -> torch.Tensor:
    """
    Extract a 3×3×8 local observation centred on Karel.

    Parameters
    ----------
    s_h : torch.Tensor
        Shape [R, T, 8, 8, 8]  (rollouts, timesteps, channels, height, width).

    Returns
    -------
    torch.Tensor
        Shape [R, T, 8, 3, 3]  with Karel at (1,1) of every 3×3 patch.
    """
    R, T, C, H, W = s_h.shape
    assert (H, W, C) == (8, 8, 8), "Expected [R, T, 8, 8, 8]"

    # --- 1) pad the spatial dims so crops at the border are valid -----------
    s_h_pad = F.pad(s_h, pad=(1, 1, 1, 1))        # → [R, T, 8, 10, 10]

    # --- 2) locate Karel in every frame (sum over facing-direction channels)
    karel_mask = s_h[:, :, 0:4].sum(dim=2)         # [R, T, 8, 8]  (0/1 values)
    flat_idx    = karel_mask.view(R, T, -1).argmax(-1)   # [R, T] linear index 0-63
    y_idx, x_idx = flat_idx // 8, flat_idx % 8            # 0-based in 8×8 map
    y_idx, x_idx = y_idx + 1, x_idx + 1                   # shift for the padding

    # --- 3) gather the 3×3 patches ------------------------------------------------
    out = torch.zeros(R, T, C, 3, 3, device=s_h.device, dtype=s_h.dtype)

    for r in range(R):
        for t in range(T):
            y, x = y_idx[r, t].item(), x_idx[r, t].item()
            out[r, t] = s_h_pad[r, t, :, y-1:y+2, x-1:x+2]   # (C,3,3)

    return out

import numpy as np

def convert_to_POMDP_np(s_h: np.ndarray):
    """
    Robustly extract 3×3×8 patches centred on Karel.

    Parameters
    ----------
    s_h : np.ndarray
        Shape (R, T, 8, 8, 8).

    Returns
    -------
    patches : np.ndarray
        Shape (R, T, 8, 3, 3).
    centred_ok : np.ndarray
        Boolean mask (R, T) – True where Karel was uniquely centred.
    """
    R, T, C, H, W = s_h.shape
    assert (H, W, C) == (8, 8, 8)

    # 1) pad 1 cell all round (zeros)
    s_pad = np.pad(s_h, ((0,0),(0,0),(0,0),(1,1),(1,1)),
                   mode='constant', constant_values=0)  # (R,T,8,10,10)

    patches     = np.zeros((R, T, 8, 3, 3), dtype=s_h.dtype)
    centred_ok  = np.zeros((R, T),          dtype=bool)

    # 2) loop is still plenty fast for 10×18
    for r in range(R):
        for t in range(T):
            # orientation channels 0-3
            orient = s_h[r, t, 0:4]                   # (4,8,8)
            yy, xx = np.where(orient.sum(0) == 1)     # positions with a single flag
            if len(yy) == 1:                          # exactly one Karel
                y, x = int(yy[0] + 1), int(xx[0] + 1) # shift for padding
                patches[r, t] = s_pad[r, t, :, y-1:y+2, x-1:x+2]
                centred_ok[r, t] = True
            # else: leave the patch zeros (indicates invalid / end-of-rollout)

    return patches, centred_ok

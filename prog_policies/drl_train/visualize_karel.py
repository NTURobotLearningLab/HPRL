"""
Helpers for recording evaluation roll-outs.
Assumes state tensors are CHANNEL-FIRST:  (C, H, W)
"""

from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import pickle, os.path as osp
from PIL import Image


# -------------------------------------------------------------------- #
#  STATE  →  IMAGE                                                     #
# -------------------------------------------------------------------- #
def state2image(s: np.ndarray, grid_size: int = 100, root_dir: str = "./") -> np.ndarray:
    """
    Convert a (C, H, W) boolean tensor into a uint8 grayscale image
    sized (H*grid, W*grid, 1).  NO HWC support—CHW only.
    """
    c, h, w = s.shape

    img = np.ones((h * grid_size, w * grid_size, 1), dtype=np.uint8)

    tex = pickle.load(open(osp.join(root_dir, "karel_env/asset/texture.pkl"), "rb"))
    blank_img, wall_img = tex["blank"], tex["wall"]
    marker_img = tex["marker"]
    agent_imgs = [tex[f"agent_{i}"] for i in range(4)]

    # blanks
    for y in range(h):
        for x in range(w):
            img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = blank_img

    # walls
    y, x = np.where(s[4])                          # plane 4 = wall
    for i in range(len(x)):
        img[y[i]*grid_size:(y[i]+1)*grid_size, x[i]*grid_size:(x[i]+1)*grid_size] = wall_img

    # markers (any quantity ≥1 → show a generic marker sprite)
    y, x = np.where(np.sum(s[6:], axis=0))
    for i in range(len(x)):
        img[y[i]*grid_size:(y[i]+1)*grid_size, x[i]*grid_size:(x[i]+1)*grid_size] = marker_img

    # agent (one-hot across planes 0-3)
    y, x = np.where(np.sum(s[:4], axis=0))
    if len(y) == 1:
        y, x = y[0], x[0]
        direction_idx = np.argmax(s[:4, y, x])
        img[y*grid_size:(y+1)*grid_size,
            x*grid_size:(x+1)*grid_size] = agent_imgs[direction_idx]

    return img


# -------------------------------------------------------------------- #
#  GIF WRITER                                                          #
# -------------------------------------------------------------------- #
def save_gif(path: str | Path, state_history, fps: int = 5):
    """
    state_history: list[ np.ndarray ]  where each array is (C, H, W)
    """
    frames = [state2image(s) for s in state_history]
    imageio.mimsave(str(path), frames, format="GIF-PIL", fps=fps)

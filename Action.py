import torch
import numpy as np


class Action:
    """Utility helpers for converting between (row,col) action tuples
    and 8x8 one-hot action planes.

    Methods accept either Python lists of (r,c) tuples or torch tensors
    of shape (B,2)."""

    @staticmethod
    def coords_to_planes(coords, device=None, dtype=torch.float32):
        # coords: list[(r,c)] or torch tensor (B,2)
        if isinstance(coords, (list, tuple)):
            if len(coords) == 0:
                return torch.empty((0, 8, 8), dtype=dtype, device=device)
            coords = torch.tensor(coords, dtype=torch.long, device=device)
        else:
            # assume tensor-like
            coords = coords.to(device) if device is not None else coords
            coords = coords.long()

        B = coords.size(0)
        planes = torch.zeros((B, 8, 8), dtype=dtype, device=device)
        if B == 0:
            return planes
        rows = coords[:, 0]
        cols = coords[:, 1]
        idx = torch.arange(B, device=device)
        planes[idx, rows, cols] = 1.0
        return planes

    @staticmethod
    def plane_to_coord(plane):
        # plane: torch (8,8) or numpy array
        if isinstance(plane, torch.Tensor):
            if plane.numel() == 0:
                return None
            if float(plane.max()) == 0.0:
                return None
            flat_idx = int(plane.reshape(-1).argmax().item())
        else:
            arr = np.asarray(plane)
            if arr.size == 0:
                return None
            if arr.max() == 0:
                return None
            flat_idx = int(arr.reshape(-1).argmax())
        return (flat_idx // 8, flat_idx % 8)

    @staticmethod
    def planes_to_coords(planes):
        # planes: torch (N,8,8) or numpy array
        if isinstance(planes, torch.Tensor):
            if planes.numel() == 0:
                return []
            cpu = planes.detach().cpu()
            N = cpu.shape[0]
            flat = cpu.view(N, -1)
            maxvals = flat.max(dim=1).values
            has = maxvals != 0
            idxs = flat.argmax(dim=1)
            coords = []
            for h, idx in zip(has, idxs):
                if h:
                    i = int(idx.item())
                    coords.append((i // 8, i % 8))
            return coords
        else:
            arr = np.asarray(planes)
            if arr.size == 0:
                return []
            N = arr.shape[0]
            flat = arr.reshape(N, -1)
            maxvals = flat.max(axis=1)
            has = maxvals != 0
            idxs = flat.argmax(axis=1)
            coords = []
            for h, idx in zip(has, idxs):
                if h:
                    coords.append((int(idx // 8), int(idx % 8)))
            return coords

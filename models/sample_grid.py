import torch
import numpy as np
from tqdm.auto import tqdm
from .common import split_tensor_by_batch


def get_grids(centers, radial_limits=[1.0, 1.5], context=None, device=None, resolution=0.1):
    '''
    Args:
        radial_limits (list of float): list with lower distance limit as first entry
            and upper distance limit as second entry
    Returns:
        grid (numpy.ndarray): 2d array of grid positions (n_grid_positions x 3) for all
            generation steps except the first
        start_grid (numpy.ndarray): 2d array of grid positions (n_grid_positions x 3)
            for the first generation step
    '''
    if device is None:
        device = centers.device
    n_dims = 3  # make grid in 3d space
    grid_max = radial_limits[1]
    grid_steps = int(grid_max * 2 * int(1/resolution)) + 1  # gives steps of length 0.05
    coords = np.linspace(-grid_max, grid_max, grid_steps)
    grid = np.meshgrid(*[coords for _ in range(n_dims)])
    grid = np.stack(grid, axis=-1)  # stack to array (instead of list)
    # reshape into 2d array of positions
    shape_a0 = np.prod(grid.shape[:n_dims])
    grid = np.reshape(grid, (shape_a0, -1))
    # cut off cells that are out of the spherical limits
    grid_dists = np.sqrt(np.sum(grid**2, axis=-1))
    grid_mask = np.logical_and(grid_dists >= radial_limits[0],
                               grid_dists <= radial_limits[1])

    grid = torch.FloatTensor(grid[grid_mask]).to(device)   # (N, 3)

    grid_all = []
    for i, pos in enumerate(centers):
        grid_this = grid + pos.view(1, n_dims)  # (N, 3)
        dists = torch.norm(
            grid_this.view(-1, 1, n_dims) - torch.cat([centers[:i],centers[i+1:]], dim=0).view(1, -1, n_dims),
            dim=-1, p=2,
        )   # (N, num_centers-1)
        mask = (dists > radial_limits[0]).all(dim=1)

        if context is not None:
            dists = torch.norm(
                grid_this.view(-1, 1, n_dims) - context.view(1, -1, n_dims),
                dim=-1, p=2,
            )   # (N, num_contexts-1)
            mask = torch.logical_and(mask, (dists > (radial_limits[0]+radial_limits[1])/2).all(dim=1))

        grid_this = grid_this[mask]

        grid_all.append(grid_this)

    return torch.cat(grid_all, dim=0)


def get_grids_batch(centers, batch_center, context=None, batch_context=None, **kwargs):
    center_list = split_tensor_by_batch(centers, batch_center)
    num_graphs = len(center_list)

    if context is not None:
        context_list = split_tensor_by_batch(context, batch_context)
    else:
        context_list = [None] * num_graphs

    grid_list = []
    for i in range(num_graphs):
        grid_list.append(get_grids(
            centers = center_list[i],
            context = context_list[i],
            **kwargs,
        ))
    return grid_list


def remove_triangles(grid, context, threshold=1.7, n_dims=3):
    num_nodes = context.size(0)
    row, col = torch.triu_indices(num_nodes,num_nodes,offset=1)
    mask = torch.norm(context[row] - context[col], dim=-1) < threshold
    row, col = row[mask], col[mask]

    dist1 = torch.norm(grid.view(-1, 1, n_dims) - context[row].view(1, -1, n_dims), dim=-1) # (N, num_pairs)
    dist2 = torch.norm(grid.view(-1, 1, n_dims) - context[col].view(1, -1, n_dims), dim=-1) # (N, num_pairs)
    mask_grid = torch.logical_not(torch.logical_and(dist1 < threshold, dist2 < threshold).any(dim=-1))
    return grid[mask_grid]



def grid_refine(pos_init, batch, model, radius=0.5, resolution=0.1, device=None):
    num_graphs = len(pos_init)

    pos_results, y_cls_results, y_ind_results, type_results = [], [], [], []
    with torch.no_grad():
        for i in range(num_graphs):
            pos_refined, y_cls_refined, y_ind_refined, type_refined = [], [], [], []
            for center in pos_init[i]:
                pos_query = get_grids(center.view(1, 3), radial_limits=(0, radius))
                model.eval()
                y_cls, y_ind = model.query_batch([pos_query], batch, limit=10000)
                y_cls, y_ind = y_cls[0], y_ind[0]
                y_flat = y_cls.flatten()
                p = (y_flat - y_flat.logsumexp(dim=0)).exp()
                p_argmax = torch.multinomial(p, 1)[0]

                # pos_idx, type_idx = p_argmax // y_cls.size(1), p_argmax % y_cls.size(1)   
                # [NOTE] operator // is deprecated by the latest version of PyTorch, use the following torch.div instead
                pos_idx = torch.div(p_argmax, y_cls.size(1), rounding_mode='floor')
                type_idx = p_argmax % y_cls.size(1)

                pos_refined.append(pos_query[pos_idx].view(1, 3))
                y_cls_refined.append(y_cls[pos_idx].view(1, -1))
                y_ind_refined.append(y_ind[pos_idx].view(1, -1))
                type_refined.append(type_idx.view(1))

            pos_refined = torch.cat(pos_refined, dim=0)
            pos_results.append(pos_refined)

            y_cls_refined = torch.cat(y_cls_refined, dim=0)
            y_cls_results.append(y_cls_refined)

            y_ind_refined = torch.cat(y_ind_refined, dim=0)
            y_ind_results.append(y_ind_refined)

            type_refined = torch.cat(type_refined, dim=0)
            type_results.append(type_refined)

    return pos_results, y_cls_results, y_ind_results, type_results

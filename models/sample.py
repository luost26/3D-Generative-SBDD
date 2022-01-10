import random
import torch
import numpy as np
from torch.nn import functional as F
from torch_geometric.data import Batch
from tqdm.auto import tqdm
from sklearn.cluster import DBSCAN, KMeans, OPTICS

from .common import split_tensor_by_batch, concat_tensors_to_batch


DEFAULT_FOLLOW_BATCH = ['protein_element', 'ligand_context_element',]


def uniform_ball_sample(num_points, r_min, r_max, device):
    phi = torch.rand(size=(num_points, 1), device=device) * (2*np.pi)
    costheta = torch.rand(size=(num_points, 1), device=device) * 2 - 1
    u = (r_max**3 - r_min**3) * torch.rand(size=(num_points, 1), device=device) + r_min**3

    theta = torch.arccos(costheta)
    r = u**(1/3)

    samples = torch.cat([
        r * torch.sin(theta) * torch.cos(phi),
        r * torch.sin(theta) * torch.sin(phi),
        r * torch.cos(theta),
    ], dim=1)
    return samples
    

def filter_too_close_points(x, y, r):
    """
    Filter out points in `x` which are too close to some point in `y`
    Args:
        x:  (N, 3)
        y:  (M, 3)
    """
    dist = torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1)  # (N, M)
    mask = (dist > r).all(dim=-1)
    return x[mask]


def sample_init(batch, model, num_points=1000, noise_std=2.0, follow_batch=DEFAULT_FOLLOW_BATCH):
    """
    Sample `num_points` positions which are more likely to have atoms for each graph.
    """

    device = batch.protein_pos.device
    data_list = batch.to_data_list()
    for i, data in enumerate(data_list):
        data._internal = i

    # Random starting points
    pos_query = []
    batch_query = []
    for i, data in enumerate(data_list):
        pos_query_modes = torch.cat([data.ligand_context_pos, data.protein_pos], dim=0)   # (N_ctx+N_prot, 3)
        mode_idx = np.random.choice(np.arange(pos_query_modes.size(0)), size=[num_points, ])
        pos_query_modes = pos_query_modes[mode_idx]
        # Append to `pos_query` and `batch_query`
        pos_query.append(pos_query_modes + torch.randn_like(pos_query_modes) * noise_std)
        batch_query.append(torch.ones([pos_query_modes.size(0)], device=device).long() * i)
    pos_query = torch.cat(pos_query, dim=0)
    batch_query = torch.cat(batch_query, dim=0)

    pos_results = [None] * batch.num_graphs
    y_cls_results = [None] * batch.num_graphs
    y_ind_results = [None] * batch.num_graphs
    num_finished = 0
    # Start sampling points
    with torch.no_grad():
        while len(data_list) > 0:
            batch = Batch.from_data_list(data_list, follow_batch=follow_batch)
            # print('InternalID:', batch._internal, )
            y_cls, y_ind = model(
                pos_query = pos_query,
                protein_pos = batch.protein_pos,
                protein_atom_feature = batch.protein_atom_feature.float(),
                ligand_pos = batch.ligand_context_pos,
                ligand_atom_feature = batch.ligand_context_feature_full.float(), 
                batch_query = batch_query,
                batch_protein = batch.protein_element_batch,
                batch_ligand = batch.ligand_context_element_batch,
            )

            has_atom = (y_cls.logsumexp(-1) > 0).cpu()
            batch_result = batch_query[has_atom]
            pos_result_list = split_tensor_by_batch(pos_query[has_atom], batch_result)
            y_cls_result_list = split_tensor_by_batch(y_cls[has_atom], batch_result)
            y_ind_result_list = split_tensor_by_batch(y_ind[has_atom], batch_result)

            pos_query_next = []
            data_list_next = []
            for i in range(len(pos_result_list)):
                if pos_result_list[i].size(0) >= num_points:
                    idx = data_list[i]._internal
                    pos_results[idx] = pos_result_list[i][:num_points]
                    y_cls_results[idx] = y_cls_result_list[i][:num_points]
                    y_ind_results[idx] = y_ind_result_list[i][:num_points]
                    num_finished += 1
                    # print('Finished: %d' % idx)
                else:
                    pos_next = pos_result_list[i].repeat(2, 1)
                    noise = torch.randn_like(pos_next) * noise_std
                    noise[:pos_result_list[i].size(0)] = 0
                    pos_next = pos_next + noise
                    pos_query_next.append(pos_next[:num_points])
                    data_list_next.append(data_list[i])
            
            data_list = data_list_next
            if len(data_list) > 0:
                pos_query, batch_query = concat_tensors_to_batch(pos_query_next)
            #     print('Next PosQuery:', [p.size() for p in pos_query_next])
            #     print('DataList Length:', len(data_list))
            # else:
            #     print('Ending')

    batch._internal_id = None
    return pos_results, y_cls_results, y_ind_results


def sample_bonded(batch, model, mask=None, num_points_per_mode=100, d_min=0.9, d_max=1.5, follow_batch=DEFAULT_FOLLOW_BATCH):
    device = batch.protein_pos.device
    data_list = batch.to_data_list()
    
    if mask is not None:
        mask_list = split_tensor_by_batch(mask, batch.ligand_context_element_batch)

    pos_results = []
    y_results = []

    with torch.no_grad():
        for i, data in enumerate(data_list):
            pos_centroids = data.ligand_context_pos     # (N_l, 3)

            if mask is not None:
                pos_centroids = pos_centroids[mask_list[i]]

            # Mask non-frontiers if needed
            assert pos_centroids.size(0) > 0
            noise = uniform_ball_sample(pos_centroids.size(0)*num_points_per_mode, d_min, d_max, device=device)
            pos_query = torch.repeat_interleave(pos_centroids, repeats=num_points_per_mode, dim=0) + noise  # (N*n, 3)
            pos_query = filter_too_close_points(pos_query, pos_centroids, r=d_min)
            batch_query = torch.zeros([pos_query.size(0)], device=device, dtype=torch.long)

            # print(pos_query.size(), data.protein_pos.size(), data.ligand_context_pos.size())
            batch_protein = torch.zeros([data.protein_pos.size(0)], device=device, dtype=torch.long)
            batch_ligand = torch.zeros([data.ligand_context_pos.size(0)], device=device, dtype=torch.long)

            y, _ = model(
                pos_query = pos_query,
                protein_pos = data.protein_pos,     # Note: not batch-wise
                protein_atom_feature = data.protein_atom_feature.float(),
                ligand_pos = data.ligand_context_pos,
                ligand_element = data.ligand_context_element,
                batch_query = batch_query,
                # batch_protein = data.protein_element_batch,
                # batch_ligand = data.ligand_context_element_batch,
                batch_protein = batch_protein,
                batch_ligand = batch_ligand,
            )

            has_atom = (y.logsumexp(-1) > 0).cpu()
            pos_results.append(pos_query[has_atom])
            y_results.append(y[has_atom])

    return pos_results, y_results


def sample_refine(batch, model, pos_results, num_iters=10, noise_std=0.1):
    """
    Refine samples by discarding low-probability half each iteration.
    Args:
        pos_results:  List of position matrices.
    """
    pos_list = pos_results

    with torch.no_grad():
        for it in range(num_iters):
            pos_query, batch_query = concat_tensors_to_batch(pos_results)
            y_cls, y_ind = model(
                pos_query = pos_query,
                protein_pos = batch.protein_pos,
                protein_atom_feature = batch.protein_atom_feature.float(),
                ligand_pos = batch.ligand_context_pos,
                ligand_atom_feature = batch.ligand_context_feature_full.float(), 
                batch_query = batch_query,
                batch_protein = batch.protein_element_batch,
                batch_ligand = batch.ligand_context_element_batch,
            )
            y_cls_results = split_tensor_by_batch(y_cls, batch_query)
            y_ind_results = split_tensor_by_batch(y_ind, batch_query)
            energy_list = split_tensor_by_batch(-y_cls.logsumexp(-1), batch_query)

            if it < num_iters - 1:
                pos_next = []
                for i in range(batch.num_graphs):
                    energy = energy_list[i]
                    top_idx = energy.argsort()[:energy.size(0) // 2]
                    top_energy = energy[:energy.size(0) // 2]
                    pos_top = pos_results[i][top_idx]
                    pos_next.append(torch.cat([
                        pos_top,
                        pos_top + torch.randn_like(pos_top) * noise_std,
                    ]))
                pos_results = pos_next
    
    return pos_results, y_cls_results, y_ind_results


def sample_refine_split(batch, model, pos_results, num_iters=10, noise_std=0.1, num_clusters=4):
    """
    Refine samples by discarding low-probability half each iteration.
    Args:
        pos_results:  List of position matrices.
    """
    pos_list = pos_results

    with torch.no_grad():
        for it in range(num_iters):
            pos_query, batch_query = concat_tensors_to_batch(pos_results)
            y, _ = model(
                pos_query = pos_query,
                protein_pos = batch.protein_pos,
                protein_atom_feature = batch.protein_atom_feature.float(),
                ligand_pos = batch.ligand_context_pos,
                ligand_atom_feature = batch.ligand_context_feature_full.float(), 
                batch_query = batch_query,
                batch_protein = batch.protein_element_batch,
                batch_ligand = batch.ligand_context_element_batch,
            )
            y_list = split_tensor_by_batch(y, batch_query)
            energy_list = split_tensor_by_batch(-y.logsumexp(-1), batch_query)

            pos_next = []
            y_results = []
            for i in range(batch.num_graphs):
                clusterer = KMeans(n_clusters=num_clusters)
                clusterer.fit(pos_results[i].clone().cpu().numpy())
                pos_top, y_top = [], []
                for j in range(clusterer.labels_.max() + 1):
                    # Cluster mask
                    cmask = (clusterer.labels_ == j)
                    # Position
                    pos_cluster = pos_results[i][cmask]
                    energy = energy_list[i][cmask]
                    top_idx = energy.argsort()[:energy.size(0) // 2]
                    top_energy = energy[:energy.size(0) // 2]
                    pos_top_cluster = pos_cluster[top_idx]
                    pos_top.append(pos_top_cluster)
                    # Y: atom type
                    y_cluster = y_list[i][cmask]
                    y_top.append(y_cluster[top_idx])

                pos_top = torch.cat(pos_top, dim=0)
                y_top = torch.cat(y_top, dim=0)

                if it < num_iters - 1:
                    pos_next.append(torch.cat([
                        pos_top,
                        pos_top + torch.randn_like(pos_top) * noise_std,
                    ]))
                else:
                    pos_next.append(pos_top)
                y_results.append(y_top)
            pos_results = pos_next
    
    return pos_results, y_results


def cluster_and_select_best(pos_results, y_cls_results, y_ind_results, eps=0.2):
    """
    Args:
        pos_results:  List of position tensors.
        y_results:    List of `y` tensors.
    """
    num_graphs = len(pos_results)
    num_types = y_cls_results[0].size(1)
    pos_selected = []
    y_cls_selected = []
    y_ind_selected = []
    type_selected = []
    for i in range(num_graphs):
        clustering = DBSCAN(eps=eps, min_samples=1).fit(pos_results[i].clone().detach().cpu().numpy())
        num_clusters = clustering.labels_.max() + 1

        # print(pos_results[i].size(), y_results[i].size(), clustering.labels_.shape)

        pos_cluster = []
        y_cls_cluster = []
        y_ind_cluster = []
        type_cluster = []
        for clus_id in range(num_clusters):
            
            clus_pos = pos_results[i][clustering.labels_ == clus_id]
            clus_y_cls = y_cls_results[i][clustering.labels_ == clus_id]
            clus_y_ind = y_ind_results[i][clustering.labels_ == clus_id]
            type_id = clus_y_cls[clus_y_cls.argmax(0), range(num_types)].argmax().view(1)
            type_cluster.append(type_id)
            point_id = clus_y_cls.argmax(0)[type_id]
            pos_cluster.append(clus_pos[point_id].view(1, 3))
            y_cls_cluster.append(clus_y_cls[point_id].view(1, -1))
            y_ind_cluster.append(clus_y_ind[point_id].view(1, -1))

        if len(pos_cluster) > 0:
            pos_selected.append(torch.cat(pos_cluster, dim=0))
            y_cls_selected.append(torch.cat(y_cls_cluster, dim=0))
            y_ind_selected.append(torch.cat(y_ind_cluster, dim=0))
            type_selected.append(torch.cat(type_cluster, dim=0))
        else:
            pos_selected.append(None)
            y_cls_selected.append(None)
            y_ind_selected.append(None)
            type_selected.append(None)

    return pos_selected, y_cls_selected, y_ind_selected, type_selected


def sample_frontier_and_bond(batch, model):
    ligand_context_element_list = split_tensor_by_batch(
        batch.ligand_context_element,
        batch = batch.ligand_context_element_batch,
    )
    query_bond = []
    query_bond_list = []
    query_bond_batch = []
    cum_nodes = 0
    for i, element in enumerate(ligand_context_element_list):
        num_prev_nodes = element.size(0) - 1
        bond_index = torch.cat([
            torch.full([1, num_prev_nodes], num_prev_nodes, device=element.device, dtype=torch.long),
            torch.arange(num_prev_nodes, device=element.device).view(1, -1),
        ], dim=0)
        query_bond_list.append(bond_index)
        query_bond.append(bond_index + cum_nodes)
        query_bond_batch.append(torch.full([bond_index.size(1)], i))
        cum_nodes += element.size(0)

    query_bond = torch.cat(query_bond, dim=1)
    query_bond_batch = torch.cat(query_bond_batch, dim=0)

    y_frontier, y_bond = model(
        protein_pos = batch.protein_pos,
        protein_atom_feature = batch.protein_atom_feature.float(),
        ligand_pos = batch.ligand_context_pos,
        ligand_atom_feature = batch.ligand_context_feature_full.float(), 
        query_bond_index = query_bond,
        batch_protein = batch.protein_element_batch,
        batch_ligand = batch.ligand_context_element_batch,
    )

    y_frontier = split_tensor_by_batch(y_frontier, batch.ligand_context_element_batch)

    y_bond = split_tensor_by_batch(y_bond, query_bond_batch, num_graphs=batch.num_graphs)

    return y_frontier, y_bond, query_bond_list


def add_ligand_atom_to_data(data, pos, atom_type, y_ind, type_map=[1,6,7,8,9,15,16,17]):
    """
    """
    data = data.clone()

    data.ligand_context_pos = torch.cat([
        data.ligand_context_pos, pos.view(1, 3).to(data.ligand_context_pos)
    ], dim=0)

    data.ligand_context_feature_full = torch.cat([
        data.ligand_context_feature_full,
        torch.cat([
            F.one_hot(atom_type.view(1), len(type_map)).to(data.ligand_context_feature_full), # (1, num_elements)
            # y_ind.sigmoid().bernoulli().to(data.ligand_context_feature_full).view(1, -1),     # (n, num_indicators)
            (y_ind >= 0).to(data.ligand_context_feature_full).view(1, -1),     # (n, num_indicators)
        ], dim=1)
    ], dim=0)

    element = torch.LongTensor([type_map[atom_type.item()]])
    data.ligand_context_element = torch.cat([
        data.ligand_context_element, element.view(1).to(data.ligand_context_element)
    ])

    return data
    

def enum_conbination(options, limit=3):
    if limit == 0:
        return
    for i in range(len(options)):
        yield [options[i], ]
        for item_j in enum_conbination(options[i+1:], limit-1):
            yield [options[i], ] + item_j


def get_next_step_comb(batch, pos_selected, y_selected, type_selected, type_map=[1,6,7,8,9,15,16,17], follow_batch=DEFAULT_FOLLOW_BATCH, num_data_limit=20, max_next_atoms=1, dist_thres=0.9):
    """
    """
    data_list = batch.to_data_list()
    results = []
    for i in range(len(data_list)):
        pos_next, y_next, type_next = pos_selected[i], y_selected[i], type_selected[i]
        if pos_next is None:
            results.append(data_list[i])
            continue
        # print(pos_next, y_next)

        for pos_comb in enum_conbination(list(range(pos_next.size(0))), max_next_atoms):

            if len(pos_comb) > 1:
                pdist = torch.norm(pos_next[pos_comb].view(1, -1, 3) - pos_next[pos_comb].view(-1, 1, 3), dim=-1, p=2)
                row, col = torch.triu_indices(pdist.size(0), pdist.size(1), offset=1)
                if pdist[row, col].min() < dist_thres:
                    continue 

            data_new = data_list[i]
            for j in pos_comb:
                data_new = add_ligand_atom_to_data(
                    data_new,
                    pos = pos_next[j],
                    atom_type = type_next[j],
                    type_map = type_map
                )

                energy_next = -1 * y_next[j][type_next[j]].item()
                if 'total_energy' not in data_new:
                    data_new.total_energy = energy_next
                else:
                    data_new.total_energy += energy_next
                data_new.average_energy = data_new.total_energy / data_new.ligand_context_pos.size(0)
            results.append(data_new)

    # results.sort(key=lambda data: data.average_energy)
    random.shuffle(results)
    results = results[:num_data_limit]
    return results


def get_next_step(batch, pos_selected, y_cls_selected, y_ind_selected, type_selected, type_map=[1,6,7,8,9,15,16,17], follow_batch=DEFAULT_FOLLOW_BATCH, num_data_limit=20):
    """
    """
    data_list = batch.to_data_list()
    results = []
    for i in range(len(data_list)):
        pos_next, y_cls_next, y_ind_next, type_next = pos_selected[i], y_cls_selected[i], y_ind_selected[i], type_selected[i]
        if pos_next is None:
            results.append(data_list[i])
            continue

        for j in range(len(pos_next)):
            data_new = add_ligand_atom_to_data(
                data_list[i],
                pos = pos_next[j],
                atom_type = type_next[j],
                y_ind = y_ind_next[j],
                type_map = type_map
            )

            logp = y_cls_next[j][type_next[j]].item()            
            if 'logp_history' not in data_new:
                data_new.logp_history = [logp]
            else:
                data_new.logp_history.append(logp)
            data_new.total_logp = np.sum(data_new.logp_history)
            data_new.average_logp = np.mean(data_new.logp_history)
            results.append(data_new)

    # results.sort(key=lambda data: -1 * data.average_logp)
    random.shuffle(results)
    results = results[:num_data_limit]
    return results

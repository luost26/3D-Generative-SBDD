import copy
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.transforms import Compose
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

from .data import ProteinLigandData
from .protein_ligand import ATOM_FAMILIES


class FeaturizeProteinAtom(object):

    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])    # H, C, N, O, S, Se
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data:ProteinLigandData):
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(1, -1)   # (N_atoms, N_elements)
        amino_acid = F.one_hot(data.protein_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        data.protein_atom_feature = x
        return data


class FeaturizeLigandAtom(object):
    
    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1,6,7,8,9,15,16,17])

    @property
    def num_properties(self):
        return len(ATOM_FAMILIES)

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + len(ATOM_FAMILIES)

    def __call__(self, data:ProteinLigandData):
        element = data.ligand_element.view(-1, 1) == self.atomic_numbers.view(1, -1)   # (N_atoms, N_elements)
        x = torch.cat([element, data.ligand_atom_feature], dim=-1)
        data.ligand_atom_feature_full = x
        return data


class FeaturizeLigandBond(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data:ProteinLigandData):
        data.ligand_bond_feature = F.one_hot(data.ligand_bond_type - 1 , num_classes=3)    # (1,2,3) to (0,1,2)-onehot
        return data


class LigandCountNeighbors(object):

    @staticmethod
    def count_neighbors(edge_index, symmetry, valence=None, num_nodes=None):
        assert symmetry == True, 'Only support symmetrical edges.'

        if num_nodes is None:
            num_nodes = maybe_num_nodes(edge_index)

        if valence is None:
            valence = torch.ones([edge_index.size(1)], device=edge_index.device)
        valence = valence.view(edge_index.size(1))

        return scatter_add(valence, index=edge_index[0], dim=0, dim_size=num_nodes).long()

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.ligand_num_neighbors = self.count_neighbors(
            data.ligand_bond_index, 
            symmetry=True,
            num_nodes=data.ligand_element.size(0),
        )
        data.ligand_atom_valence = self.count_neighbors(
            data.ligand_bond_index, 
            symmetry=True, 
            valence=data.ligand_bond_type,
            num_nodes=data.ligand_element.size(0),
        )
        return data


class LigandRandomMask(object):

    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked

    def __call__(self, data:ProteinLigandData):
        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_atoms = data.ligand_element.size(0)
        num_masked = int(num_atoms * ratio)

        if num_masked < self.min_num_masked:
            num_masked = self.min_num_masked
        if (num_atoms - num_masked) < self.min_num_unmasked:
            num_masked = num_atoms - self.min_num_unmasked

        idx = np.arange(num_atoms)
        np.random.shuffle(idx)
        idx = torch.LongTensor(idx)
        masked_idx = idx[:num_masked]
        context_idx = idx[num_masked:]
        
        data.ligand_masked_element = data.ligand_element[masked_idx]
        data.ligand_masked_feature = data.ligand_atom_feature[masked_idx]   # For Prediction
        data.ligand_masked_pos = data.ligand_pos[masked_idx]

        data.ligand_context_element = data.ligand_element[context_idx]
        data.ligand_context_feature_full = data.ligand_atom_feature_full[context_idx]   # For Input
        data.ligand_context_pos = data.ligand_pos[context_idx]

        data.ligand_context_bond_index, data.ligand_context_bond_feature = subgraph(
            context_idx,
            data.ligand_bond_index,
            edge_attr = data.ligand_bond_feature,
            relabel_nodes = True,
        )
        data.ligand_context_num_neighbors = LigandCountNeighbors.count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            num_nodes = context_idx.size(0),
        )

        # print(context_idx)
        # print(data.ligand_context_bond_index)

        # mask = torch.logical_and(
        #     (data.ligand_bond_index[0].view(-1, 1) == context_idx.view(1, -1)).any(dim=-1),
        #     (data.ligand_bond_index[1].view(-1, 1) == context_idx.view(1, -1)).any(dim=-1),
        # )
        # print(data.ligand_bond_index[:, mask])

        # print(data.ligand_context_num_neighbors)
        # print(data.ligand_num_neighbors[context_idx])


        data.ligand_frontier = data.ligand_context_num_neighbors < data.ligand_num_neighbors[context_idx]

        data._mask = 'random'

        return data


class LigandBFSMask(object):
    
    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0, inverse=False):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked
        self.inverse = inverse

    @staticmethod
    def get_bfs_perm(nbh_list):
        num_nodes = len(nbh_list)
        num_neighbors = torch.LongTensor([len(nbh_list[i]) for i in range(num_nodes)])

        bfs_queue = [random.randint(0, num_nodes-1)]
        bfs_perm = []
        num_remains = [num_neighbors.clone()]
        bfs_next_list = {}
        visited = {bfs_queue[0]}   

        num_nbh_remain = num_neighbors.clone()
        
        while len(bfs_queue) > 0:
            current = bfs_queue.pop(0)
            for nbh in nbh_list[current]:
                num_nbh_remain[nbh] -= 1
            bfs_perm.append(current)
            num_remains.append(num_nbh_remain.clone())
            next_candid = []
            for nxt in nbh_list[current]:
                if nxt in visited: continue
                next_candid.append(nxt)
                visited.add(nxt)
                
            random.shuffle(next_candid)
            bfs_queue += next_candid
            bfs_next_list[current] = copy.copy(bfs_queue)

        return torch.LongTensor(bfs_perm), bfs_next_list, num_remains

    def __call__(self, data):
        bfs_perm, bfs_next_list, num_remaining_nbs = self.get_bfs_perm(data.ligand_nbh_list)

        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_atoms = data.ligand_element.size(0)
        num_masked = int(num_atoms * ratio)
        if num_masked < self.min_num_masked:
            num_masked = self.min_num_masked
        if (num_atoms - num_masked) < self.min_num_unmasked:
            num_masked = num_atoms - self.min_num_unmasked

        if self.inverse:
            masked_idx = bfs_perm[:num_masked]
            context_idx = bfs_perm[num_masked:]
        else:
            masked_idx = bfs_perm[-num_masked:]
            context_idx = bfs_perm[:-num_masked]
        
        data.ligand_masked_element = data.ligand_element[masked_idx]
        data.ligand_masked_feature = data.ligand_atom_feature[masked_idx]   # For Prediction
        data.ligand_masked_pos = data.ligand_pos[masked_idx]

        data.ligand_context_element = data.ligand_element[context_idx]
        data.ligand_context_feature_full = data.ligand_atom_feature_full[context_idx]   # For Input
        data.ligand_context_pos = data.ligand_pos[context_idx]

        data.ligand_context_bond_index, data.ligand_context_bond_feature = subgraph(
            context_idx,
            data.ligand_bond_index,
            edge_attr = data.ligand_bond_feature,
            relabel_nodes = True,
        )
        data.ligand_context_num_neighbors = LigandCountNeighbors.count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            num_nodes = context_idx.size(0),
        )

        # print(context_idx)
        # print(data.ligand_context_bond_index)

        # mask = torch.logical_and(
        #     (data.ligand_bond_index[0].view(-1, 1) == context_idx.view(1, -1)).any(dim=-1),
        #     (data.ligand_bond_index[1].view(-1, 1) == context_idx.view(1, -1)).any(dim=-1),
        # )
        # print(data.ligand_bond_index[:, mask])

        # print(data.ligand_context_num_neighbors)
        # print(data.ligand_num_neighbors[context_idx])

        data.ligand_frontier = data.ligand_context_num_neighbors < data.ligand_num_neighbors[context_idx]

        data._mask = 'invbfs' if self.inverse else 'bfs'

        return data


class LigandMaskAll(LigandRandomMask):

    def __init__(self):
        super().__init__(min_ratio=1.0)


class LigandMixedMask(object):

    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0, p_random=0.5, p_bfs=0.25, p_invbfs=0.25):
        super().__init__()

        self.t = [
            LigandRandomMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked),
            LigandBFSMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked, inverse=False),
            LigandBFSMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked, inverse=True),
        ]
        self.p = [p_random, p_bfs, p_invbfs]

    def __call__(self, data):
        f = random.choices(self.t, k=1, weights=self.p)[0]
        return f(data)


def get_mask(cfg):
    if cfg.type == 'bfs':
        return LigandBFSMask(
            min_ratio=cfg.min_ratio, 
            max_ratio=cfg.max_ratio, 
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
        )
    elif cfg.type == 'random':
        return LigandRandomMask(
            min_ratio=cfg.min_ratio, 
            max_ratio=cfg.max_ratio, 
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
        )
    elif cfg.type == 'mixed':
        return LigandMixedMask(
            min_ratio=cfg.min_ratio, 
            max_ratio=cfg.max_ratio, 
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
            p_random = cfg.p_random,
            p_bfs = cfg.p_bfs,
            p_invbfs = cfg.p_invbfs,
        )
    elif cfg.type == 'all':
        return LigandMaskAll()
    else:
        raise NotImplementedError('Unknown mask: %s' % cfg.type)


class ContrastiveSample(object):

    def __init__(self, num_real=50, num_fake=50, pos_real_std=0.05, pos_fake_std=2.0, elements=None):
        super().__init__()
        self.num_real = num_real
        self.num_fake = num_fake
        self.pos_real_std = pos_real_std
        self.pos_fake_std = pos_fake_std
        if elements is None:
            # elements = torch.LongTensor([
            #     1, 3, 5, 6, 7, 8, 9, 
            #     12, 13, 14, 15, 16, 17, 
            #     21, 23, 24, 26, 27, 29, 33, 34, 35, 
            #     39, 42, 44, 50, 53, 74, 79, 80
            # ])
            elements = [1,6,7,8,9,15,16,17]
        self.elements = torch.LongTensor(elements)

    @property
    def num_elements(self):
        return self.elements.size(0)

    def __call__(self, data:ProteinLigandData):
        # Positive samples
        pos_real_mode = data.ligand_masked_pos
        element_real = data.ligand_masked_element
        ind_real = data.ligand_masked_feature
        cls_real = data.ligand_masked_element.view(-1, 1) == self.elements.view(1, -1)
        if not (cls_real.sum(-1) > 0).all():
            print(data.ligand_element)
        assert (cls_real.sum(-1) > 0).all(), 'Unexpected elements.'

        real_sample_idx = np.random.choice(np.arange(pos_real_mode.size(0)), size=self.num_real)
        data.pos_real = pos_real_mode[real_sample_idx]
        data.pos_real += torch.randn_like(data.pos_real) * self.pos_real_std
        data.element_real = element_real[real_sample_idx]
        data.cls_real = cls_real[real_sample_idx]
        data.ind_real = ind_real[real_sample_idx]

        # Negative samples
        pos_fake_mode = torch.cat([data.ligand_context_pos, data.protein_pos], dim=0)
        fake_sample_idx = np.random.choice(np.arange(pos_fake_mode.size(0)), size=self.num_fake)
        data.pos_fake = pos_fake_mode[fake_sample_idx]
        data.pos_fake += torch.randn_like(data.pos_fake) * self.pos_fake_std

        return data


def get_contrastive_sampler(cfg):
    return ContrastiveSample(
        num_real = cfg.num_real,
        num_fake = cfg.num_fake,
        pos_real_std = cfg.pos_real_std,
        pos_fake_std = cfg.pos_fake_std,
    )

import torch
from torch.nn import Module, Linear, Embedding
from torch.nn import functional as F

from .encoders import get_encoder
from .common import MultiLayerPerceptron, compose_context_stable


class FrontierNetwork(Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        self.config = config
        
        self.protein_atom_emb = Linear(protein_atom_feature_dim, config.hidden_channels)
        self.ligand_atom_emb = Linear(ligand_atom_feature_dim, config.hidden_channels)

        self.encoder = get_encoder(config.encoder)

        self.frontier_pred = MultiLayerPerceptron(
            input_dim = self.encoder.out_channels,
            hidden_dims = [128, 128, 64, 1],
        )


    def forward(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, batch_protein, batch_ligand):
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        h_ctx, pos_ctx, batch_ctx, mask_protein = compose_context_stable(
            h_protein = h_protein,
            h_ligand = h_ligand,
            pos_protein = protein_pos,
            pos_ligand = ligand_pos,
            batch_protein = batch_protein,
            batch_ligand = batch_ligand,
        )
        mask_ligand = torch.logical_not(mask_protein)

        h_ctx = self.encoder(
            node_attr = h_ctx,
            pos = pos_ctx,
            batch = batch_ctx,
        )   # (N_p+N_l, H)
        h_ligand = h_ctx[mask_ligand]   # (N_l, H)

        y_frontier = self.frontier_pred(h_ligand)

        return y_frontier


    def get_loss(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, ligand_bond_index, y_bond, y_frontier, batch_protein, batch_ligand):
        """
        Args:
            ligand_frontier:    (N_l, 1)
        """

        y_frontier_pred = self(
            protein_pos = protein_pos,
            protein_atom_feature = protein_atom_feature,
            ligand_pos = ligand_pos,
            ligand_atom_feature = ligand_atom_feature,
            batch_protein = batch_protein,
            batch_ligand = batch_ligand,
        )

        loss_frontier = F.binary_cross_entropy_with_logits(
            input = y_frontier_pred,
            target = y_frontier.view(-1, 1).float()
        )

        return loss_frontier


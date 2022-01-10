import torch
from torch.nn import Module, Linear, Embedding
from torch.nn import functional as F

from .encoders import get_encoder
from .fields import get_field
from .common import *


class MaskFillModel(Module):

    def __init__(self, config, num_classes, num_indicators, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        self.config = config
        
        self.protein_atom_emb = Linear(protein_atom_feature_dim, config.hidden_channels)
        self.ligand_atom_emb = Linear(ligand_atom_feature_dim, config.hidden_channels)

        self.encoder = get_encoder(config.encoder)
        self.field = get_field(config.field, num_classes=num_classes, num_indicators=num_indicators, in_channels=self.encoder.out_channels)

        self.smooth_cross_entropy = SmoothCrossEntropyLoss(reduction='mean', smoothing=0.1)
        
    def forward(self, pos_query, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, batch_query, batch_protein, batch_ligand):
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        h_ctx, pos_ctx, batch_ctx = compose_context(
            h_protein = h_protein,
            h_ligand = h_ligand,
            pos_protein = protein_pos,
            pos_ligand = ligand_pos,
            batch_protein = batch_protein,
            batch_ligand = batch_ligand,
        )

        h_ctx = self.encoder(
            node_attr = h_ctx,
            pos = pos_ctx,
            batch = batch_ctx,        
        )   # (N_p+N_l, H)

        y_cls, y_ind = self.field(
            pos_query = pos_query,
            pos_ctx = pos_ctx,
            node_attr_ctx = h_ctx,
            batch_query = batch_query,
            batch_ctx = batch_ctx,
        )   # (N_query, num_classes), (N_query, num_indicators)

        return y_cls, y_ind

    def get_loss(self, pos_real, y_real, p_real, pos_fake, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, batch_real, batch_fake, batch_protein, batch_ligand):
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        h_ctx, pos_ctx, batch_ctx = compose_context(
            h_protein = h_protein,
            h_ligand = h_ligand,
            pos_protein = protein_pos,
            pos_ligand = ligand_pos,
            batch_protein = batch_protein,
            batch_ligand = batch_ligand,
        )

        h_ctx = self.encoder(
            node_attr = h_ctx,
            pos = pos_ctx,
            batch = batch_ctx,        
        )   # (N_p+N_l, H)    
    
        # Positive samples
        y_real_pred, p_real_pred = self.field(
            pos_query = pos_real,
            pos_ctx = pos_ctx,
            node_attr_ctx = h_ctx,
            batch_query = batch_real,
            batch_ctx = batch_ctx,
        )   # (N_real, num_classes)
        energy_real = -1 *  torch.logsumexp(y_real_pred, dim=-1)  # (N_real)
        loss_cls = self.smooth_cross_entropy(y_real_pred, y_real.argmax(-1))    # Classes
        loss_ind = F.binary_cross_entropy_with_logits(p_real_pred, p_real)      # Indicators

        # Negative samples
        y_fake_pred, _ = self.field(
            pos_query = pos_fake,
            pos_ctx = pos_ctx,
            node_attr_ctx = h_ctx,
            batch_query = batch_fake,
            batch_ctx = batch_ctx,
        )   # (N_fake, num_classes)

        energy_fake = -1 * torch.logsumexp(y_fake_pred, dim=-1)   # (N_fake)
        loss_nce_real = -1 * torch.log(torch.sigmoid(-energy_real)).mean()
        loss_nce_fake = -1 * torch.log(torch.sigmoid(energy_fake)).mean()
        # print(energy_fake.abs().max().item(), torch.sigmoid(energy_fake).abs().max().item(), torch.sigmoid(energy_fake).abs().min().item())
        loss_nce = loss_nce_real + loss_nce_fake

        loss = loss_cls + loss_nce + loss_ind

        return loss, loss_cls, loss_nce_real, loss_nce_fake, loss_ind


    def query_batch(self, pos_query_list, batch, limit=10000):
        pos_query, batch_query = concat_tensors_to_batch(pos_query_list)
        num_query = pos_query.size(0)

        y_cls_all, y_ind_all = [], []
        for pos_query_partial, batch_query_partial in zip(split_tensor_to_segments(pos_query, limit), split_tensor_to_segments(batch_query, limit)):
            PM = batch_intersection_mask(batch.protein_element_batch, batch_query_partial)
            LM = batch_intersection_mask(batch.ligand_context_element_batch, batch_query_partial)
            
            y_cls_partial, y_ind_partial = self(
                # Query
                pos_query = pos_query_partial,
                batch_query = batch_query_partial,
                # Protein
                protein_pos = batch.protein_pos[PM],
                protein_atom_feature = batch.protein_atom_feature.float()[PM],
                batch_protein = batch.protein_element_batch[PM],
                # Ligand
                ligand_pos = batch.ligand_context_pos[LM],
                ligand_atom_feature = batch.ligand_context_feature_full.float()[LM], 
                batch_ligand = batch.ligand_context_element_batch[LM],
            )
            y_cls_all.append(y_cls_partial)
            y_ind_all.append(y_ind_partial)
        
        y_cls_all = torch.cat(y_cls_all, dim=0)
        y_ind_all = torch.cat(y_ind_all, dim=0)

        lengths = [x.size(0) for x in pos_query_list]
        y_cls_list = split_tensor_by_lengths(y_cls_all, lengths)
        y_ind_list = split_tensor_by_lengths(y_ind_all, lengths)

        return y_cls_list, y_ind_list


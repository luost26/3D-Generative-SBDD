import os
import shutil
import argparse
from tqdm.auto import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch_geometric.loader import DataLoader

from models.maskfill import MaskFillModel
from utils.datasets import *
from utils.transforms import *
from utils.misc import *
from utils.train import *

import torch_geometric.data.collate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    # Logging
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Transforms
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    masking = get_mask(config.train.transform.mask)
    contrastive_sampler = get_contrastive_sampler(config.train.transform.contrastive)
    transform = Compose([
        LigandCountNeighbors(),
        protein_featurizer,
        ligand_featurizer,
        FeaturizeLigandBond(),
        masking,
        contrastive_sampler,
    ])

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config = config.dataset,
        transform = transform,
    )
    train_set, val_set = subsets['train'], subsets['test']
    follow_batch = ['protein_element', 'ligand_context_element', 'pos_real', 'pos_fake']
    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = inf_iterator(DataLoader(
        train_set, 
        batch_size = config.train.batch_size, 
        shuffle = True,
        num_workers = config.train.num_workers,
        follow_batch = follow_batch,
        exclude_keys = collate_exclude_keys,
    ))
    val_loader = DataLoader(
        val_set, 
        config.train.batch_size, 
        shuffle=False, 
        follow_batch=follow_batch,
        exclude_keys = collate_exclude_keys,
    )

    # Model
    logger.info('Building model...')
    model = MaskFillModel(
        config.model, 
        num_classes = contrastive_sampler.num_elements,
        num_indicators = ligand_featurizer.num_properties,
        protein_atom_feature_dim = protein_featurizer.feature_dim,
        ligand_atom_feature_dim = ligand_featurizer.feature_dim,
    ).to(args.device)

    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    def train(it):
        model.train()
        optimizer.zero_grad()
        batch = next(train_iterator).to(args.device)

        protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
        ligand_noise = torch.randn_like(batch.ligand_context_pos) * config.train.pos_noise_std

        loss, loss_cls, loss_nce_real, loss_nce_fake, loss_ind = model.get_loss(
            pos_real = batch.pos_real,
            y_real = batch.cls_real.long(),
            p_real = batch.ind_real.float(),    # Binary indicators: float
            pos_fake = batch.pos_fake,
            protein_pos = batch.protein_pos + protein_noise,
            protein_atom_feature = batch.protein_atom_feature.float(),
            ligand_pos = batch.ligand_context_pos + ligand_noise,
            ligand_atom_feature = batch.ligand_context_feature_full.float(), 
            batch_real = batch.pos_real_batch,
            batch_fake = batch.pos_fake_batch,
            batch_protein = batch.protein_element_batch,
            batch_ligand = batch.ligand_context_element_batch,
        )
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        logger.info('[Train] Iter %d | Loss %.6f | Loss(Cls) %.6f | Loss(Ind) %.6f | Loss(Real) %.6f | Loss(Fake) %.6f' % (
            it, loss.item(), loss_cls.item(), loss_ind.item(), loss_nce_real.item(), loss_nce_fake.item()
        ))
        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/loss_cls', loss_cls, it)
        writer.add_scalar('train/loss_ind', loss_ind, it)
        writer.add_scalar('train/loss_real', loss_nce_real, it)
        writer.add_scalar('train/loss_fake', loss_nce_fake, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad', orig_grad_norm, it)
        writer.flush()

    def validate(it):
        sum_loss, sum_n = 0, 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                loss, loss_cls, loss_nce_real, loss_nce_fake, loss_ind = model.get_loss(
                    pos_real = batch.pos_real,
                    y_real = batch.cls_real.long(),
                    p_real = batch.ind_real.float(),    # Binary indicators: float
                    pos_fake = batch.pos_fake,
                    protein_pos = batch.protein_pos,
                    protein_atom_feature = batch.protein_atom_feature.float(),
                    ligand_pos = batch.ligand_context_pos,
                    ligand_atom_feature = batch.ligand_context_feature_full.float(), 
                    batch_real = batch.pos_real_batch,
                    batch_fake = batch.pos_fake_batch,
                    batch_protein = batch.protein_element_batch,
                    batch_ligand = batch.ligand_context_element_batch,
                )
                sum_loss += loss.item()
                sum_n += 1
        avg_loss = sum_loss / sum_n

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info('[Validate] Iter %05d | Loss %.6f' % (
            it, avg_loss,
        ))
        writer.add_scalar('val/loss', avg_loss, it)
        writer.flush()
        return avg_loss

    try:
        for it in range(1, config.train.max_iters+1):
            # with torch.autograd.detect_anomaly():
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
        

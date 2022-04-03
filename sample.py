import os
import shutil
import argparse
import random
import torch
import numpy as np
from torch_geometric.data import Batch
from easydict import EasyDict
from tqdm.auto import tqdm
from rdkit import Chem
from scipy.special import softmax

from models.maskfill import MaskFillModel
from models.frontier import FrontierNetwork
from models.sample import *
from models.sample_grid import *
from utils.transforms import *
from utils.datasets import get_dataset
from utils.misc import *
from utils.data import FOLLOW_BATCH
from utils.reconstruct import *
from utils.chem import *


STATUS_RUNNING = 'running'
STATUS_FINISHED = 'finished'
STATUS_FAILED = 'failed'


def get_init_samples(data, model, batch_size=1, num_points=8000, refine_using_grid=True, default_max_retry=5):
    batch = Batch.from_data_list([data] * batch_size, follow_batch=FOLLOW_BATCH)
    with torch.no_grad():
        pos_results, y_cls, y_ind = sample_init(batch, model, num_points=num_points)
        pos_results, y_cls, y_ind = sample_refine(batch, model, pos_results)
        pos_selected, y_cls_selected, y_ind_selected, type_selected = cluster_and_select_best(pos_results, y_cls, y_ind, eps=0.2)
        if refine_using_grid:
            pos_selected, y_cls_selected, y_ind_selected, type_selected = grid_refine(pos_selected, batch, model)
    data_list = get_next_step(batch, pos_selected, y_cls_selected, y_ind_selected, type_selected, num_data_limit=20)
    for data in data_list:
        data.remaining_retry = default_max_retry
        data.status = STATUS_RUNNING
    return data_list


@torch.no_grad()
def get_next(data, ftnet, model, num_next, factor_frontier=1.0, factor_cls=5.0, logger=BlackHole()):
    batch = Batch.from_data_list([data], follow_batch=FOLLOW_BATCH)

    ### Predict which atoms are frontiers
    ftnet.eval()
    y_frontier = ftnet(
        protein_pos = batch.protein_pos,
        protein_atom_feature = batch.protein_atom_feature.float(),
        ligand_pos = batch.ligand_context_pos,
        ligand_atom_feature = batch.ligand_context_feature_full.float(), 
        batch_protein = batch.protein_element_batch,
        batch_ligand = batch.ligand_context_element_batch,
    )
    # frontier_mask = (y_frontier * factor_frontier).flatten().sigmoid().bernoulli().bool()
    frontier_mask = (y_frontier >= 0).flatten()
    # If no frontiers, mark as success
    if frontier_mask.sum().item() == 0:
        # logger.info('[%s] Finished' % data.ligand_filename)
        data.status = STATUS_FINISHED
        return [data]

    ### Sample from the neighborhood of frontiers
    # Generate meshgrid to discretize the probability
    pos_query = get_grids_batch(
        batch.ligand_context_pos[frontier_mask],
        batch.ligand_context_element_batch[frontier_mask],
    )[0]    # This function only handles 1 data at a call
    # pos_query = remove_triangles(pos_query, batch.ligand_context_pos, threshold=1.5)

    # Evaluate probabilites on the meshgrid
    model.eval()
    y_cls_list, y_ind_list = model.query_batch([pos_query], batch, limit=10000)
    y_cls, y_ind = y_cls_list[0], y_ind_list[0] # This function only handles 1 data at a call
    y_flat = y_cls.flatten() * factor_cls
    # Sample the index of next position and type
    p = (y_flat - y_flat.logsumexp(dim=0)).exp()
    p_argmax = torch.multinomial(p, num_next)  # OR  p_argmax = p.argsort(descending=True)[0]
    pos_idx, type_idx = p_argmax // y_cls.size(1), p_argmax % y_cls.size(1)

    pos_next = [pos_query[pos_idx].view(-1,3)]
    y_cls_next = [y_cls[pos_idx].view(-1, y_cls.size(1))]
    y_ind_next = [y_ind[pos_idx].view(-1, y_ind.size(1))]
    type_next = [type_idx.view(-1)]

    # Next state
    data_next_list = get_next_step(
        batch,
        pos_selected = pos_next,
        y_cls_selected = y_cls_next,
        y_ind_selected = y_ind_next,
        type_selected = type_next,
    )

    # logger.info('[%s] logp=%.6f' % (data.ligand_filename, logp))

    return [data.to('cpu') for data in data_next_list]


def print_pool_status(pool, logger):
    logger.info('[Pool] Queue %d | Finished %d | Failed %d' % (
        len(pool.queue), len(pool.finished), len(pool.failed)
    ))


def random_roll_back(data):
    num_steps = len(data.logp_history)
    back_to = random.randint(1, max(1, num_steps-1))
    data.ligand_context_element = data.ligand_context_element[:back_to]
    data.ligand_context_feature_full = data.ligand_context_feature_full[:back_to]
    data.ligand_context_pos = data.ligand_context_pos[:back_to]
    data.logp_history = data.logp_history[:back_to]

    data.total_logp = np.sum(data.logp_history)
    data.average_logp = np.mean(data.logp_history)
    
    return data


def data_exists(data, prevs):
    for other in prevs:
        if len(data.logp_history) == len(other.logp_history):
            if (data.ligand_context_element == other.ligand_context_element).all().item() and \
                (data.ligand_context_feature_full == other.ligand_context_feature_full).all().item() and \
                torch.allclose(data.ligand_context_pos, other.ligand_context_pos):
                return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('-i', '--data_id', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--outdir', type=str, default='./outputs')
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.sample.seed)

    # Logging
    log_dir = get_new_log_dir(args.outdir, prefix='%s-%d' % (config_name, args.data_id))
    logger = get_logger('sample', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))    

    # Data
    logger.info('Loading data...')
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    contrastive_sampler = ContrastiveSample(num_real=0, num_fake=0)
    masking = LigandMaskAll()
    transform = Compose([
        LigandCountNeighbors(),
        protein_featurizer,
        ligand_featurizer,
        FeaturizeLigandBond(),
        masking,
    ])
    dataset, subsets = get_dataset(
        config = config.dataset,
        transform = transform,
    )
    testset = subsets['test']
    data = testset[args.data_id]

    with open(os.path.join(log_dir, 'pocket_info.txt'), 'a') as f:
        f.write(data.protein_filename + '\n')

    # Model (Main)
    logger.info('Loading main model...')
    ckpt = torch.load(config.model.main.checkpoint, map_location=args.device)
    model = MaskFillModel(
        ckpt['config'].model, 
        num_classes = contrastive_sampler.num_elements,
        protein_atom_feature_dim = protein_featurizer.feature_dim,
        ligand_atom_feature_dim = ligand_featurizer.feature_dim,
        num_indicators = len(ATOM_FAMILIES)
    ).to(args.device)
    model.load_state_dict(ckpt['model'])

    # Model (Frontier Network)
    logger.info('Loading frontier model...')
    ckpt_ft = torch.load(config.model.frontier.checkpoint, map_location=args.device)
    ftnet = FrontierNetwork(
        ckpt_ft['config'].model,
        protein_atom_feature_dim = protein_featurizer.feature_dim,
        ligand_atom_feature_dim = ligand_featurizer.feature_dim,
    ).to(args.device)
    ftnet.load_state_dict(ckpt_ft['model'])


    pool = EasyDict({
        'queue': [],
        'failed': [],
        'finished': [],
        'duplicate': [],
        'smiles': set(),
    })

    logger.info('Initialization')
    pbar = tqdm(total=config.sample.num_samples, desc='InitSample')
    while len(pool.queue) < config.sample.num_samples:
        queue_size_before = len(pool.queue)
        pool.queue += get_init_samples(
            data = data.to(args.device), 
            model = model,
            default_max_retry = config.sample.num_retry,
        )
        if len(pool.queue) > config.sample.num_samples:
            pool.queue = pool.queue[:config.sample.num_samples]
        pbar.update(len(pool.queue) - queue_size_before)
    pbar.close()

    print_pool_status(pool, logger)
    logger.info('Saving samples...')
    torch.save(pool, os.path.join(log_dir, 'samples_init.pt'))

    logger.info('Start sampling')

    global_step = 0

    try:
        while len(pool.finished) < config.sample.num_samples:
            global_step += 1
            queue_size = len(pool.queue)

            queue_tmp = []
            for data in tqdm(pool.queue):
                nexts = []
                data_next_list = get_next(
                    data.to(args.device), 
                    ftnet = ftnet,
                    model = model,
                    logger = logger,
                    num_next = 5,
                )

                for data_next in data_next_list:
                    if data_next.status == STATUS_FINISHED:
                        try:
                            rdmol = reconstruct_from_generated(data_next)
                            smiles = Chem.MolToSmiles(rdmol)
                            data_next.smiles = smiles
                            data_next.rdmol = rdmol
                            valid = filter_rd_mol(rdmol)
                            if not valid:
                                logger.warning('Ignoring invalid molecule: %s' % smiles)
                                pool.failed.append(data_next)
                            elif smiles in pool.smiles:
                                logger.warning('Ignoring duplicate molecule: %s' % smiles)
                                pool.duplicate.append(data_next)
                            else:   # Pass checks
                                logger.info('Success: %s' % smiles)
                                pool.finished.append(data_next)
                                pool.smiles.add(smiles)
                        except MolReconsError:
                            logger.warning('Ignoring, because reconstruction error encountered.')
                            pool.failed.append(data_next)
                    else:
                        if data_next.logp_history[-1] < config.sample.logp_thres:
                            if data_next.remaining_retry > 0:
                                data_next.remaining_retry -= 1
                                logger.info('[%s] Retrying, remaining %d retries' % (data.ligand_filename, data_next.remaining_retry))
                                nexts.append(random_roll_back(data_next))
                            else:
                                logger.info('[%s] Failed' % (data.ligand_filename,))
                                pool.failed.append(data_next)
                        else:
                            nexts.append(data_next)

                queue_tmp += nexts

            next_factor = 1.0
            p_next = softmax(np.array([np.mean(data.logp_history) for data in queue_tmp]) * next_factor)
            # print(np.arange(len(queue_tmp)), config.sample.beam_size)
            next_idx = np.random.choice(
                np.arange(len(queue_tmp)),
                size=config.sample.beam_size,
                replace=True,
                p=p_next,
            )
            pool.queue = [queue_tmp[idx] for idx in next_idx]

            print_pool_status(pool, logger)
            torch.save(pool, os.path.join(log_dir, 'samples_%d.pt' % global_step))
    except KeyboardInterrupt:
        logger.info('Terminated. Generated molecules will be saved.')

    torch.save(pool, os.path.join(log_dir, 'samples_all.pt'))

    sdf_dir = os.path.join(log_dir, 'SDF')
    os.makedirs(sdf_dir)
    with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
        for i, data_finished in enumerate(pool['finished']):
            smiles_f.write(data_finished.smiles + '\n')
            writer = Chem.SDWriter(os.path.join(sdf_dir, '%d.sdf' % i))
            writer.SetKekulize(False)
            writer.write(data_finished.rdmol, confId=0)
            writer.close()

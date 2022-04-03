import os
import argparse
import warnings
from easydict import EasyDict
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Selection import unfold_entities
from rdkit import Chem

from utils.protein_ligand import PDBProtein
from sample import *    # Import everything from `sample.py`


def pdb_to_pocket_data(pdb_path, center, bbox_size):
    center = torch.FloatTensor(center)
    warnings.simplefilter('ignore', BiopythonWarning)
    ptable = Chem.GetPeriodicTable()
    parser = PDBParser()
    model = parser.get_structure(None, pdb_path)[0]

    protein_dict = EasyDict({
        'element': [],
        'pos': [],
        'is_backbone': [],
        'atom_to_aa_type': [],
    })
    for atom in unfold_entities(model, 'A'):
        res = atom.get_parent()
        resname = res.get_resname()
        if resname == 'MSE': resname = 'MET'
        if resname not in PDBProtein.AA_NAME_NUMBER: continue   # Ignore water, heteros, and non-standard residues.

        element_symb = atom.element.capitalize()
        if element_symb == 'H': continue
        x, y, z = atom.get_coord()
        pos = torch.FloatTensor([x, y, z])
        if (pos - center).abs().max() > (bbox_size / 2): 
            continue

        protein_dict['element'].append( ptable.GetAtomicNumber(element_symb))
        protein_dict['pos'].append(pos)
        protein_dict['is_backbone'].append(atom.get_name() in ['N', 'CA', 'C', 'O'])
        protein_dict['atom_to_aa_type'].append(PDBProtein.AA_NAME_NUMBER[resname])
        
    if len(protein_dict['element']) == 0:
        raise ValueError('No atoms found in the bounding box (center=%r, size=%f).' % (center, bbox_size))

    protein_dict['element'] = torch.LongTensor(protein_dict['element'])
    protein_dict['pos'] = torch.stack(protein_dict['pos'], dim=0)
    protein_dict['is_backbone'] = torch.BoolTensor(protein_dict['is_backbone'])
    protein_dict['atom_to_aa_type'] = torch.LongTensor(protein_dict['atom_to_aa_type'])

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict = protein_dict,
        ligand_dict = {
            'element': torch.empty([0,], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0,], dtype=torch.long),
        }
    )
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str,
                        default='./example/4yhj.pdb')
    parser.add_argument('--center', type=lambda s: list(map(float, s.split(','))),
                        default=[32.0, 28.0, 36.0], 
                        help='Center of the pocket bounding box, in format x,y,z')
    parser.add_argument('--bbox_size', type=float, default=23.0, 
                        help='Pocket bounding box size')
    parser.add_argument('--config', type=str, default='./configs/sample_for_pdb.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--outdir', type=str, default='./outputs')
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.sample.seed)

    # Logging
    log_dir = get_new_log_dir(args.outdir, prefix='%s-%s' % (
        config_name, 
        os.path.basename(args.pdb_path),
    ))
    logger = get_logger('sample', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))    
    shutil.copyfile(args.pdb_path, os.path.join(log_dir, os.path.basename(args.pdb_path)))    


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
    data = pdb_to_pocket_data(args.pdb_path, args.center, args.bbox_size)
    data = transform(data)

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



    # Sampling
    # The algorithm is the same as the one `sample.py`.

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

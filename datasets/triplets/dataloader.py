
import torch
import logging
from datasets.triplets.mulran_dataset import MulRanTupleDataset
from models.MinkLoc3dv2.mink_params import TrainingParams
from datasets.triplets.base_dataset import TrainingDataset
from datasets.triplets.samplers import BatchSampler
from torch.utils.data.sampler import Sampler
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from datasets.triplets.scannetpr_dataset import ScannetTripleDataset

def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e

class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.
      Arguments:
          data_source (Dataset): dataset to sample from
          shuffle: use random permutation
      """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        else:
            perm = torch.arange(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop(0)

    def __len__(self):
        return len(self.data_source)

def collate_tuple(list_data):
    queries = []
    positives = []
    negatives = []
    other_neg = []
    for k in range(len(list_data)):
        queries.append(list_data[k][0])
        positives.append(list_data[k][1])
        negatives.append(list_data[k][2])
        other_neg.append(list_data[k][3])
    
    queries, positives, negatives, other_neg = np.array(queries), np.array(positives), np.array(negatives), np.array(other_neg)
    queries_tensor = torch.from_numpy(queries).unsqueeze(1).float()
    positives_tensor = torch.from_numpy(positives).float()
    negatives_tensor = torch.from_numpy(negatives).float()
    other_neg_tensor = torch.from_numpy(other_neg).unsqueeze(1).float()
    feed_tensor = torch.cat(
        (queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor), 1)
    feed_tensor = feed_tensor.view((-1, 1, 4096, 3)).squeeze(1)
    # for turple_data in inputs:
    #     for data in turple_data:
    #         if isinstance(data, np.ndarray):
    #             outputs.append(data)  
    #         elif isinstance(data, list):
    #             outputs.extend(data)    
    
    # outputs = np.array(outputs)
    
    return feed_tensor


def make_datasets(params: TrainingParams, validation: bool = True):
    # Create training and validation datasets
    datasets = {}
    train_set_transform = None

    # PoinNetVLAD datasets (RobotCar and Inhouse)
    # PNV datasets have their own transform
    train_transform = None
    datasets['train'] = TrainingDataset(params.dataset_folder, transform=train_transform, set_transform=train_set_transform)
    if validation:
        datasets['val'] = TrainingDataset(params.dataset_folder, params.val_file)

    return datasets
  
def make_dataloaders(params: TrainingParams, validation=False):
    
    ######################################################################
    datasets = make_datasets(params, validation=validation)

    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)

    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    quantizer = params.model_params.quantizer
    train_collate_fn = make_collate_fn(datasets['train'],  quantizer, params.batch_split_size)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler,
                                     collate_fn=train_collate_fn, num_workers=params.num_workers,
                                     pin_memory=True)
    if validation and 'val' in datasets:
        val_collate_fn = make_collate_fn(datasets['val'], quantizer, params.batch_split_size)
        val_sampler = BatchSampler(datasets['val'], batch_size=params.val_batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=params.num_workers, pin_memory=True)

    return dataloders


def make_collate_fn(dataset: TrainingDataset, quantizer, batch_split_size=None):
    # quantizer: converts to polar (when polar coords are used) and quantizes
    # batch_split_size: if not None, splits the batch into a list of multiple mini-batches with batch_split_size elems
    def collate_fn(data_list):
        # Constructs a batch object
        clouds = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]

        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            lens = [len(cloud) for cloud in clouds]
            clouds = torch.cat(clouds, dim=0)
            clouds = dataset.set_transform(clouds)
            clouds = clouds.split(lens)

        # Compute positives and negatives mask
        # dataset.queries[label]['positives'] is bitarray
        positives_mask = [[in_sorted_array(e, dataset.files[label][2]) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.files[label][3]) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Convert to polar (when polar coords are used) and quantize
        # Use the first value returned by quantizer
        coords = [quantizer(e)[0] for e in clouds]

        if batch_split_size is None or batch_split_size == 0:
            coords = ME.utils.batched_coordinates(coords)
            # Assign a dummy feature equal to 1 to each point
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            batch = {'coords': coords, 'features': feats}

        else:
            # Split the batch into chunks
            batch = []
            for i in range(0, len(coords), batch_split_size):
                temp = coords[i:i + batch_split_size]
                c = ME.utils.batched_coordinates(temp)
                f = torch.ones((c.shape[0], 1), dtype=torch.float32)
                minibatch = {'coords': c, 'features': f}
                batch.append(minibatch)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and negatives_mask which are
        # batch_size x batch_size boolean tensors
        #return batch, positives_mask, negatives_mask, torch.tensor(sampled_positive_ndx), torch.tensor(relative_poses)
        return batch, positives_mask, negatives_mask

    return collate_fn



def make_data_loader_for_scannet(args):
    dset = ScannetTripleDataset(args,'training')
    sampler = RandomSampler(dset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(dset,
                                        num_workers=16,
                                        batch_size=args["BATCH_NUM_QUERIES"],
                                        sampler=sampler,
                                        pin_memory=True)
    return data_loader
    
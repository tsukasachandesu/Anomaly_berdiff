import os
from PIL import Image
import yaml
import torch
import torchvision
from torch.utils.data.dataset import Subset
from torchvision.transforms import (CenterCrop, Compose, RandomHorizontalFlip, Resize, ToTensor, RandomResizedCrop, RandomCrop )
import sys
sys.path.append('./Binary_AE')

import misc
import numpy as np
import PIL.Image
import random
from torch.utils.data import Dataset, DataLoader

augmentation_transform = Compose([
    RandomHorizontalFlip(p=0.5)
])


class MapTransformOverNumpyArrayChannels:
    """Maps a torchvision.transforms transform over the dimension 0 of a numpy.ndarray

    Takes a numpy C x H x W array and converts each channel to a PIL.Image. Applies
    the transform to each PIL.Image and converts them back to numpy  H x W x C

    Can be used just like torchvision.transforms
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, numpyArray):
        rng_state = random.getstate()  # resetting the RNG for each layer
        np_rng_state = np.random.get_state()
        outArray = np.empty_like(numpyArray)

        for k, channel in enumerate(numpyArray):
            random.setstate(rng_state)
            np.random.set_state(np_rng_state)
            channel = np.array(channel)
            img = PIL.Image.fromarray(channel)
            img = self.transform(img)
            outChannel = np.array(img)
            outArray[k, :, :] = outChannel
        return outArray

    def __repr__(self):
        return "MapTransformOverNumpyArrayChannels.__repr__() not implemented"
        pass


class TransposeNumpy:
    """Transposes a numpy.ndarray

    Can be used just like torchvision.transforms
    """
    def __init__(self, transposition=None):
        self.transposition = transposition

    def __call__(self, numpyArray):
       # outArray = numpyArray.transpose(self.transposition)
        outArray=numpyArray
        arrays = [outArray[:,:,0], outArray[:,:,0], outArray[:,:,0]]  #stacking for RGB

        outArray2=np.stack(arrays, axis=-1)
        return outArray2

    def __repr__(self):
        return "TransposeNumpy.__repr__() not implemented"
        pass

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample
def my_split_by_worker(urls):
    wi = torch.utils.data.get_worker_info()
    if wi is None:
        return urls
    else:
        return urls[wi.id::wi.num_workers]

def my_split_by_node(urls):
    node_id, node_count = torch.distributed.get_rank(), torch.distributed.get_world_size()
    return urls[node_id::node_count]


def dataset_to_dataloader(dataset, batch_size, num_prepro_workers, input_format):
    """Create a pytorch dataloader from a dataset"""

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)
    print('dataset', dataset)
    data = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_prepro_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn if input_format == "files" else None,
    )
    return data


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class BigDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.image_paths = os.listdir(folder)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = Image.open(os.path.join(self.folder, path))
        img = np.array(img)
        img = torch.from_numpy(img).permute(2, 0, 1)  # -> channels first
        return img

    def __len__(self):
        return len(self.image_paths)


class NoClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, length=None):
        self.dataset = dataset
        self.length = length if length is not None else len(dataset)

    def __getitem__(self, index):
        return self.dataset[index][0].mul(255).clamp_(0, 255).to(torch.uint8)

    def __len__(self):
        return self.length


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def get_default_dataset_paths():
    with open("datasets.yml") as yaml_file:
        read_data = yaml.load(yaml_file, Loader=yaml.FullLoader)

    paths = {}
    for i in range(len(read_data)):
        paths[read_data[i]["dataset"]] = read_data[i]["path"]

    return paths


def train_val_split(dataset, train_val_ratio):
    indices = list(range(len(dataset)))
    split_index = int(len(dataset) * train_val_ratio)
    train_indices, val_indices = indices[:split_index], indices[split_index:]
    train_dataset, val_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices)
    return train_dataset, val_dataset


def get_datasets(
    dataset_name,
    img_size,
    get_val_dataset=False,
    get_flipped=False,
    train_val_split_ratio=0.95,
    custom_dataset_path=None,
    random=False
):
    # if random:
    #     transform = Compose([  ToTensor()])
    # else:
    #     transform = Compose([   RandomHorizontalFlip(p=0.5), ToTensor()])
    # transform_with_flip = transform
    transform = Compose([
        MapTransformOverNumpyArrayChannels(augmentation_transform),
        TransposeNumpy(),
        ToTensor()
    ])
    transform_with_flip = transform

    print(transform)
    
    default_paths = get_default_dataset_paths()

    if dataset_name in default_paths:
        dataset_path = default_paths[dataset_name]
    elif dataset_name == "custom":
        if custom_dataset_path:
            dataset_path = custom_dataset_path
        else:
            raise ValueError("Custom dataset selected, but no path provided")
    else:
        raise ValueError(f"Invalid dataset chosen: {dataset_name}. To use a custom dataset, set --dataset \
            flag to 'custom'.")


    if dataset_name == "churches":
        train_dataset = torchvision.datasets.LSUN(
            dataset_path,
            classes=["church_outdoor_train"],
            transform=transform
        )
        if get_flipped:
            train_dataset_flip = torchvision.datasets.LSUN(
                dataset_path,
                classes=["church_outdoor_train"],
                transform=transform_with_flip,
            )
        if get_val_dataset:
            val_dataset = torchvision.datasets.LSUN(
                dataset_path,
                classes=["church_outdoor_val"],
              #  transform=transform
            )

    elif dataset_name == "bedrooms":
        train_dataset = torchvision.datasets.LSUN(
            dataset_path,
            classes=["bedroom_train"],
           # transform=transform,
        )
        if get_val_dataset:
            val_dataset = torchvision.datasets.LSUN(
                dataset_path,
                classes=["bedroom_val"],
                transform=transform,
            )

        if get_flipped:
            train_dataset_flip = torchvision.datasets.LSUN(
                dataset_path,
                classes=["bedroom_train"],
                transform=transform_with_flip,
            )

    elif dataset_name == "custom":
        train_dataset = torchvision.datasets.ImageFolder(
            dataset_path,
            transform=transform,
        )

        if get_flipped:
            train_dataset_flip = torchvision.datasets.ImageFolder(
                dataset_path,
                transform=transform_with_flip,
            )

        if get_val_dataset:
            train_dataset, val_dataset = train_val_split(train_dataset, train_val_split_ratio)
            if get_flipped:
                train_dataset_flip, _ = train_val_split(train_dataset_flip, train_val_split_ratio)

    elif dataset_name == "chexpert":
        print('chexpert path', dataset_path)
        path=os.path.join(dataset_path, 'train') #get all samples with labels
        train_dataset = torchvision.datasets.DatasetFolder(
            root=path,
            loader=npy_loader,
            transform=transform,
            extensions=('.npy',)
        )
        print('train Dataset', len(train_dataset))


        if get_flipped:
            train_dataset_flip = torchvision.datasets.DatasetFolder(
                root=dataset_path,
                loader=npy_loader,
                transform=transform_with_flip,
                extensions=('.npy',)
            )


            print('Dataset', len(train_dataset_flip))

        if get_val_dataset:
            print('chexpert path', dataset_path)
            path = os.path.join(dataset_path, 'validate') #only get diseased samples
            val_dataset = torchvision.datasets.DatasetFolder(
                root=path,
                loader=npy_loader,
                transform=transform,
                extensions=('.npy',)
            )
            print('val Dataset', len(val_dataset))



    if get_flipped:
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_flip])

    if not get_val_dataset:
        val_dataset = None

    return train_dataset, val_dataset


def get_data_loaders(
    dataset_name,
    img_size,
    batch_size,
    get_flipped=False,
    train_val_split_ratio=0.95,
    custom_dataset_path=None,
    num_workers=1,
    drop_last=True,
    shuffle=True,
    get_val_dataloader=False,
    distributed=False,
    random=False,
    args=None, 
    
):


    if dataset_name in ['bedrooms', 'churches', 'custom', 'chexpert', 'brats', 'OCT']:
        train_dataset, val_dataset = get_datasets(
            dataset_name,
            img_size,
            get_flipped=get_flipped,
            get_val_dataset=get_val_dataloader,
            train_val_split_ratio=train_val_split_ratio,
            custom_dataset_path=custom_dataset_path,
            random=random
        )

        # if distributed:
        #     print('distributed')
        #     num_tasks = misc.get_world_size()
        #     global_rank = misc.get_rank()
        #    # sampler_train = torch.utils.data.DistributedSampler(
        #     #    train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        #   #  )
        #    # print("Sampler_train = %s" % str(sampler_train))
        #
        # else:
        #     print('not distributed')

           # sampler_train = torch.utils.data.RandomSampler(train_dataset)
        train_loader= torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True)


    if get_val_dataloader:
        if distributed:
            if args.dist_eval:
                if len(val_dataset) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
               # sampler_val = torch.utils.data.DistributedSampler(
                  #  val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
            else:
          #      sampler_val = torch.utils.data.SequentialSampler(val_dataset)
                print('ski')

           # sampler_val = torch.utils.data.SequentialSampler(val_dataset)

        val_loader = torch.utils.data.DataLoader(
                val_dataset,
                num_workers=num_workers,
              #  sampler=sampler_val,
                batch_size=batch_size,
                pin_memory=True,
                drop_last=drop_last
            )
    else:
        val_loader = None

    return train_loader, val_loader

from .cifar import CIFAR, CIFAR_C
from .imagenet import IMAGENET_C, IMAGENET_


def get_dataset(dataset_name, root_dir, opt, load_source=False, aug_mult=0, hard_augment="randaugment"):

    if "cifar" in dataset_name:
        if load_source:
            return CIFAR(root_dir, dataset_name, False)
        else:
            return CIFAR_C(root_dir, dataset_name, opt.corruption, opt.corruption_level, aug_mult=aug_mult)
    elif "imagenet" in dataset_name:
        if load_source:
            return IMAGENET_(root_dir)
        else:
            return IMAGENET_C(root_dir, corruption=opt.corruption, level=opt.corruption_level, aug_mult=aug_mult)
    else:
        raise NotImplementedError(f"The dataset {dataset_name} is not implemented yet !")

from torch.utils.data import DataLoader

from dataloaders import dataset


def make_data_loader(args, **kwargs):

    if args.dataset == 'RS':
        train_set = dataset.RSSet(args, split='train')
        val_set = dataset.RSSet(args, split='val')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, val_loader
    else:
        raise NotImplementedError


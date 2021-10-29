from torch.utils.data import Dataset



class SimpleDataset(Dataset):
    def __init__(self, iterable_data):
        self.data = iterable_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

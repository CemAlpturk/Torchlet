import torchlet
from torchlet.data import DataLoader, Dataset


class DummyDataset(Dataset):
    def __init__(self, data) -> None:
        if not isinstance(data, torchlet.Tensor):
            data = torchlet.tensor(data)
        self.data = data

    def __getitem__(self, index: int) -> torchlet.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.data.shape[0]


class DummyLabeledDataset(Dataset):
    def __init__(self, data, labels) -> None:
        if not isinstance(data, torchlet.Tensor):
            data = torchlet.tensor(data)

        if not isinstance(labels, torchlet.Tensor):
            labels = torchlet.tensor(labels)

        self.data = data
        self.labels = labels

    def __getitem__(self, index: int) -> tuple[torchlet.Tensor, torchlet.Tensor]:
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return self.data.shape[0]


def test_dataloader_no_shuffle_no_drop_last():
    dataset = DummyDataset([1, 2, 3, 4, 5])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)

    batches = list(dataloader)
    assert batches == [
        torchlet.tensor([1, 2]),
        torchlet.tensor([3, 4]),
        torchlet.tensor([5]),
    ]


def test_dataloader_no_shuffle_drop_last():
    dataset = DummyDataset([1, 2, 3, 4, 5])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=True)

    batches = list(dataloader)
    assert batches == [torchlet.tensor([1, 2]), torchlet.tensor([3, 4])]


def test_dataloader_shuffle():
    dataset = DummyDataset([1, 2, 3, 4, 5])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=False)

    batches = list(dataloader)
    assert len(batches) == 3
    assert sum(batch.shape[0] for batch in batches) == 5


def test_dataloader_len_no_drop_last():
    dataset = DummyDataset([1, 2, 3, 4, 5])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)

    assert len(dataloader) == 3


def test_dataloader_len_drop_last():
    dataset = DummyDataset([1, 2, 3, 4, 5])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=True)

    assert len(dataloader) == 2


def test_dataloader_tuple_batch():
    data = [1, 2, 3, 4, 5, 6]
    labels = [0, 1, 0, 1, 0, 1]
    dataset = DummyLabeledDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)

    batches = list(dataloader)
    for batch in batches:
        x, y = batch
        assert isinstance(x, torchlet.Tensor)
        assert isinstance(y, torchlet.Tensor)
        assert x.shape == (2,)
        assert y.shape == (2,)

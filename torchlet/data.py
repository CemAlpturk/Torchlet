from typing import TYPE_CHECKING, Any, Iterator
import random

if TYPE_CHECKING:
    from torchlet import Tensor


class Dataset:
    """
    An iterable dataset.
    """

    def __getitem__(self, index: int) -> Tensor | tuple[Tensor, ...]: ...

    def __len__(self) -> int: ...


class DataLoader:
    """
    An iterable over a dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = list(range(len(dataset)))

    def __iter__(self) -> Iterator[Any]:
        if self.shuffle:
            random.shuffle(self.indices)

        for start_idx in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[start_idx : start_idx + self.batch_size]

            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            batch = [self.dataset[idx] for idx in batch_indices]

            if isinstance(batch[0], tuple):
                # Transpose the batch to seperate each element
                yield tuple(zip(*batch))
            else:
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

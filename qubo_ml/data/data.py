import threading
from random import Random
from typing import Dict, Iterator, List, Optional, Union, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from rdkit import Chem

from .scaler import StandardScaler
from .features_generators import get_features_generator

# Cache of RDKit molecules
CACHE_MOL = True
SMILES_TO_MOL: Dict[str, Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]] = {}
def empty_cache():
    r"""Empties the cache of :class:`~chemprop.features.MolGraph` and RDKit molecules."""
    SMILES_TO_MOL.clear()

def cache_mol() -> bool:
    r"""Returns whether RDKit molecules will be cached."""
    return CACHE_MOL

def set_cache_mol(cache_mol: bool) -> None:
    r"""Sets whether RDKit molecules will be cached."""
    global CACHE_MOL
    CACHE_MOL = cache_mol

def make_mol(s: str,):
    """
    Builds an RDKit molecule from a SMILES string.

    :param s: SMILES string.
    :return: RDKit molecule.
    """
    mol = Chem.MolFromSmiles(s)
    return mol

def make_mols(smiles: List[str], is_reaction: bool):
    """
    Builds a list of RDKit molecules (or a list of tuples of molecules if reaction is True) for a list of smiles.

    :param smiles: List of SMILES strings.
    :param is_reaction: A boolean whether the SMILES strings are to be treated as a reaction.
    :return: List of RDKit molecules or list of tuple of molecules.
    """
    mol = []
    for s in smiles:
        if is_reaction:
            mol.append(SMILES_TO_MOL[s] if s in SMILES_TO_MOL else (make_mol(s.split(">")[0]), make_mol(s.split(">")[-1])))
        else:
            mol.append(SMILES_TO_MOL[s] if s in SMILES_TO_MOL else make_mol(s))
    return mol

class MoleculeDatapoint:
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets."""

    def __init__(self,
                 smiles: List[str],
                 targets: List[Optional[float]] = None,
                 extra_features: np.ndarray = None,
                 features_generator: List[str] = None,
                 is_reaction: bool = None,
                 reaction_mode: str = None,
                 limits: List[Optional[float]] = [0, 100]):
        """
        :param smiles: A list of the SMILES strings for the molecules.
        :param targets: A list of targets for the molecule.
        :param extra_features: A numpy array containing additional features.
        :param features_generator: A list of features generators to use.
        """
        self.smiles = smiles
        self.targets = targets
        self.extra_features = extra_features
        self.features_generator = features_generator
        self.is_reaction = is_reaction
        self.reaction_mode = reaction_mode
        self.limits = limits

        # Generate additional features if given a generator
        self.features = []
        r_features, p_features = [], []
        if self.features_generator is not None:
            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                if not self.is_reaction:
                    for m in self.mol:
                        if m is not None and m.GetNumHeavyAtoms() > 0:
                            self.features.extend(features_generator(m))
                        # for H2
                        elif m is not None and m.GetNumHeavyAtoms() == 0:
                            # not all features are equally long, so use methane as dummy molecule to determine length
                            self.features.extend(np.zeros(len(features_generator(Chem.MolFromSmiles('C')))))
                else:
                    for r_m, p_m in self.mol:
                        if r_m is not None and r_m.GetNumHeavyAtoms() > 0:
                            r_features.extend(features_generator(r_m))
                        # for H2
                        elif r_m is not None and r_m.GetNumHeavyAtoms() == 0:
                            # not all features are equally long, so use methane as dummy molecule to determine length
                            r_features.extend(np.zeros(len(features_generator(Chem.MolFromSmiles('C')))))
                        
                        if p_m is not None and p_m.GetNumHeavyAtoms() > 0:
                            p_features.extend(features_generator(p_m))
                        # for H2
                        elif p_m is not None and p_m.GetNumHeavyAtoms() == 0:
                            # not all features are equally long, so use methane as dummy molecule to determine length
                            p_features.extend(np.zeros(len(features_generator(Chem.MolFromSmiles('C')))))
                        
                    if self.reaction_mode  == 'reac_prod':
                        self.features.extend(r_features + p_features)
                    elif self.reaction_mode  == 'reac_diff':
                        features_diff = list(map(lambda x, y: x - y, p_features, r_features))
                        self.features.extend(r_features + features_diff)
                    elif self.reaction_mode  == 'prod_diff':
                        features_diff = list(map(lambda x, y: x - y, p_features, r_features))
                        self.features.extend(p_features + features_diff)
                    elif self.reaction_mode  == 'reac_only':
                        self.features.extend(r_features)
                    elif self.reaction_mode  == 'prod_only':
                        self.features.extend(p_features)
                    elif self.reaction_mode  == 'diff_only':
                        features_diff = list(map(lambda x, y: x - y, p_features, r_features))
                        self.features.extend(features_diff)
                    else:
                        raise ValueError(f"Not supported reaction mode: {reaction_mode}")

        if self.extra_features is not None:
            self.features.extend(extra_features)

        self.features = np.array(self.features)

        # Save a copy of the raw features, targets, and limits to enable different scaling later on
        self.raw_features, self.raw_targets, self.raw_limits = self.features, self.targets, self.limits

    @property
    def mol(self) -> List[Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]]:
        """Gets the corresponding list of RDKit molecules for the corresponding SMILES list."""
        mol = make_mols(self.smiles, self.is_reaction)
        if cache_mol():
            for s, m in zip(self.smiles, mol):
                SMILES_TO_MOL[s] = m

        return mol

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in the :class:`MoleculeDatapoint`.

        :return: The number of molecules.
        """
        return len(self.smiles)

    def set_features(self, features: np.ndarray) -> None:
        """
        Sets the features of the molecule.

        :param features: A 1D numpy array of features for the molecule.
        """
        self.features = features

    def extend_features(self, features: np.ndarray) -> None:
        """
        Extends the features of the molecule.

        :param features: A 1D numpy array of extra features for the molecule.
        """
        self.features = np.append(self.features, features) if self.features is not None else features

    def set_targets(self, targets: List[Optional[float]]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets

    def set_limits(self, limits: List[Optional[float]]):
        """
        Sets the limits of a molecule.

        :param limits: A list of floats containing the limits.
        """
        self.limits = limits

    def reset_features_targets_and_limits(self) -> None:
        """Resets the features, targets, and limits to their raw values."""
        self.features, self.targets, self.limits = self.raw_features, self.raw_targets, self.raw_limits


class MoleculeDataset(Dataset):
    r"""A :class:`MoleculeDataset` contains a list of :class:`MoleculeDatapoint`\ s with access to their attributes."""

    def __init__(self, data: List[MoleculeDatapoint]):
        r"""
        :param data: A list of :class:`MoleculeDatapoint`\ s.
        """
        self._data = data
        self._random = Random()

    def smiles(self, flatten: bool = False) -> Union[List[str], List[List[str]]]:
        """
        Returns a list containing the SMILES list associated with each :class:`MoleculeDatapoint`.

        :param flatten: Whether to flatten the returned SMILES to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of SMILES, depending on :code:`flatten`.
        """
        if flatten:
            return [smiles for d in self._data for smiles in d.smiles]

        return [d.smiles for d in self._data]

    def mols(self, flatten: bool = False) -> Union[List[Chem.Mol], List[List[Chem.Mol]], List[Tuple[Chem.Mol, Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]]]:
        """
        Returns a list of the RDKit molecules associated with each :class:`MoleculeDatapoint`.

        :param flatten: Whether to flatten the returned RDKit molecules to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of RDKit molecules, depending on :code:`flatten`.
        """
        if flatten:
            return [mol for d in self._data for mol in d.mol]

        return [d.mol for d in self._data]

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in each :class:`MoleculeDatapoint`.

        :return: The number of molecules.
        """
        return self._data[0].number_of_molecules if len(self._data) > 0 else None

    def extra_features(self) -> List[np.ndarray]:
        """
        Returns the extra features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the extra features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].extra_features is None:
            return None

        return [d.extra_features for d in self._data]

    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        return [d.features for d in self._data]

    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        return [d.targets for d in self._data]

    def limits(self) -> List[List[Optional[float]]]:
        """
        Returns the limits associated with each molecule.

        :return: A list of lists of floats (or None) containing the limits.
        """
        return [d.limits for d in self._data]

    def features_size(self) -> int:
        """
        Returns the size of the additional features vector associated with the molecules.

        :return: The size of the additional features vector.
        """
        return len(self._data[0].features) if len(self._data) > 0 and self._data[0].features is not None else None

    def normalize_targets(self) -> StandardScaler:
        """
        Normalizes the targets of the dataset using a :class:`~chemprop.data.StandardScaler`.
        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each task independently.
        This should only be used for regression datasets.
        :return: A :class:`~chemprop.data.StandardScaler` fitted to the targets.
        """
        targets = [d.raw_targets for d in self._data]
        scaler = StandardScaler().fit(targets)
        scaled_targets = scaler.transform(targets).tolist()
        self.set_targets(scaled_targets)

        for i, limit in enumerate(self.limits()):
            if limit is not None:
                scaled_limits = scaler.transform(limit).tolist()
                self._data[i].set_limits(scaled_limits)

        return scaler

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats (or None) containing targets for each molecule. This must be the
                        same length as the underlying dataset.
        """
        if not len(self._data) == len(targets):
            raise ValueError(
                "number of molecules and targets must be of same length! "
                f"num molecules: {len(self._data)}, num targets: {len(targets)}"
            )
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def reset_features_targets_and_limits(self) -> None:
        """Resets the features (atom, bond, and molecule), targets, and limits to their raw values."""
        for d in self._data:
            d.reset_features_targets_and_limits()

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).

        :return: The length of the dataset.
        """
        return len(self._data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A :class:`MoleculeDatapoint` if an int is provided or a list of :class:`MoleculeDatapoint`\ s
                 if a slice is provided.
        """
        return self._data[item]

def construct_molecule_batch(data: List[MoleculeDatapoint]) -> MoleculeDataset:
    r"""
    Constructs a :class:`MoleculeDataset` from a list of :class:`MoleculeDatapoint`\ s.

    :param data: A list of :class:`MoleculeDatapoint`\ s.
    :return: A :class:`MoleculeDataset` containing all the :class:`MoleculeDatapoint`\ s.
    """
    data = MoleculeDataset(data)

    return data


class MoleculeSampler(Sampler):
    """A :class:`MoleculeSampler` samples data from a :class:`MoleculeDataset` for a :class:`MoleculeDataLoader`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if :code:`shuffle` is True.
        """
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.shuffle = shuffle
        self._random = Random(seed)

        self.positive_indices = self.negative_indices = None
        self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator over indices to sample."""
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of indices that will be sampled."""
        return self.length


class MoleculeDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 batch_size: int = 50,
                 num_workers: int = 8,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param dataset: The :class:`MoleculeDataset` containing the molecules to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        if self._shuffle:
            raise ValueError('Cannot safely extract targets when shuffle is enabled.')

        return [self._dataset[index].targets for index in self._sampler]

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(MoleculeDataLoader, self).__iter__()

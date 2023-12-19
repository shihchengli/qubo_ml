from typing import Callable, List, Union

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint


Molecule = Union[str, Chem.Mol]
FeaturesGenerator = Callable[[Molecule], np.ndarray]

FEATURES_GENERATOR_REGISTRY = {}
NUM_BITS = None

def register_features_generator(features_generator_name: str) -> Callable[[FeaturesGenerator], FeaturesGenerator]:
    """
    Creates a decorator which registers a features generator in a global dictionary to enable access by name.

    :param features_generator_name: The name to use to access the features generator.
    :return: A decorator which will add a features generator to the registry using the specified name.
    """
    def decorator(features_generator: FeaturesGenerator) -> FeaturesGenerator:
        FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator

    return decorator

def get_features_generator(features_generator_name: str,
                           num_bits: int) -> FeaturesGenerator:
    """
    Gets a registered features generator by name.

    :param features_generator_name: The name of the features generator.
    :param num_bits: Number of bits in the fingerprint.
    :return: The desired features generator.
    """
    if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found. '
                         f'If this generator relies on rdkit features, you may need to install descriptastorus.')
    
    # Set global variable
    global NUM_BITS
    NUM_BITS = num_bits

    return FEATURES_GENERATOR_REGISTRY[features_generator_name]


def get_available_features_generators() -> List[str]:
    """Returns a list of names of available features generators."""
    return list(FEATURES_GENERATOR_REGISTRY.keys())

@register_features_generator('MACCS')
def MACCS(mol: Molecule) -> np.ndarray:
    """
    Generates a binary MACCS fingerprint for a molecule.
    In order to allow the original numbering of the MACCS keys to be used, the
    fingerprints need to be 167 bits long. Bit 0 should always be zero. Thus,
    we ignore the first bit to make it 166 bits.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :return: A 1D numpy array containing the binary MACCS fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    fp = GetMACCSKeysFingerprint(mol)
    features = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, features)
    return features[1:]

@register_features_generator('Avalon')
def Avalon(mol: Molecule,
           num_bits: int = NUM_BITS) -> np.ndarray:
    """
    Generates a binary Avalon fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param num_bits: Number of bits in Avalon fingerprint.
    :return: A 1D numpy array containing the binary Avalon fingerprint.
    """
    if num_bits is None:
        global NUM_BITS
        num_bits = NUM_BITS
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    fp = GetAvalonFP(mol, nBits=num_bits)
    features = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, features)
    return features

@register_features_generator('ECFP4')
def ECFP4(mol: Molecule,
          num_bits: int = None) -> np.ndarray:
    """
    Generates a binary ECFP4 fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param num_bits: Number of bits in ECFP4 fingerprint.
    :return: A 1D numpy array containing the binary ECFP4 fingerprint.
    """
    if num_bits is None:
        global NUM_BITS
        num_bits = NUM_BITS
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=num_bits)
    features = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, features)
    return features

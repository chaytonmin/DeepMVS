from .dtu import DTUDataset
from .tanks import TanksDataset
from .blendedmvs import BlendedMVSDataset
from .normal_dtu import NormalDataset
#from .normal_blended import NormalDataset

dataset_dict = {'dtu': DTUDataset,
                'tanks': TanksDataset,
                'blendedmvs': BlendedMVSDataset,
		'normal':NormalDataset}

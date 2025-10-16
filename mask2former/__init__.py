from . import data  # register all new datasets
from . import modeling

# config
from .config import (
    add_maskformer2_config,
    add_vl_maskformer2_config,
    add_semi_sup_config,
)

from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_semi_sup_dataset_mapper import (
    MaskFormerPanopticSemiSupDatasetMapper,
)

from .data.dataset_mappers.mask_former_instance_dataset_mapper_sa1b import (
    MaskFormerInstanceDatasetMapperSA1B,
)

# models
from .vision_language_maskformer_model import VisionLanguageMaskFormer
from .vision_language_maskformer_student import VisionLanguageMaskFormerStudent

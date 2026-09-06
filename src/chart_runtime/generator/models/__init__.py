"""Native latest-model architecture modules."""

from .anchor import AnchorRelationModel
from .density_calibrator import DensityCalibrator, DensityCalibratorConfig
from .factor import FactorConfig, FactorEventModel, FactorHead
from .joint_plan import JointEventPlanModel
from .motion import MOTION_STATE_DIM, motion_state_from_representations, motion_state_sequence
from .planner import (
    FullSongBudgetPlanner,
    PersistentPlannerConfig,
    PersistentResidualSectionPlanner,
    PersistentSectionPlanner,
    PlannerConfig,
)
from .relational import V4RelationalRenderer, geometry_candidate_index
from .structure import ChartTransformerV2, V2Config
from .structured import StructuredFactorHead, V3StructuredRenderer
from .style import StylePrior, StylePriorConfig

__all__ = [
    "AnchorRelationModel",
    "ChartTransformerV2",
    "DensityCalibrator",
    "DensityCalibratorConfig",
    "FactorConfig",
    "FactorEventModel",
    "FactorHead",
    "FullSongBudgetPlanner",
    "JointEventPlanModel",
    "MOTION_STATE_DIM",
    "PersistentPlannerConfig",
    "PersistentResidualSectionPlanner",
    "PersistentSectionPlanner",
    "PlannerConfig",
    "StructuredFactorHead",
    "StylePrior",
    "StylePriorConfig",
    "V2Config",
    "V4RelationalRenderer",
    "V3StructuredRenderer",
    "geometry_candidate_index",
    "motion_state_from_representations",
    "motion_state_sequence",
]

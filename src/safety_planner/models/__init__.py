__all__ = [
    "VectorSceneEncoder",
    "ILHead",
    "BSplineCandidateHead",
    "SafetyPredictionHeads",
    "LightweightRanker",
    "GBDTRanker",
    "SmallMLPRanker",
    "SafetyConsistentPlannerModel",
]


def __getattr__(name: str):
    if name == "VectorSceneEncoder":
        from .scene_encoder import VectorSceneEncoder

        return VectorSceneEncoder
    if name == "ILHead":
        from .il_head import ILHead

        return ILHead
    if name == "BSplineCandidateHead":
        from .bspline_candidate_head import BSplineCandidateHead

        return BSplineCandidateHead
    if name in {"SafetyPredictionHeads", "LightweightRanker"}:
        from .safety_heads import LightweightRanker, SafetyPredictionHeads

        return {"SafetyPredictionHeads": SafetyPredictionHeads, "LightweightRanker": LightweightRanker}[name]
    if name in {"GBDTRanker", "SmallMLPRanker"}:
        from .lightweight_ranker import GBDTRanker, SmallMLPRanker

        return {"GBDTRanker": GBDTRanker, "SmallMLPRanker": SmallMLPRanker}[name]
    if name == "SafetyConsistentPlannerModel":
        from .planner_model import SafetyConsistentPlannerModel

        return SafetyConsistentPlannerModel
    raise AttributeError(name)

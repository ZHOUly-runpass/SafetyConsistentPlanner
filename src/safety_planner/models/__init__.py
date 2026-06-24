__all__ = [
    "VectorSceneEncoder",
    "ILHead",
    "BSplineCandidateHead",
    "SafetyPredictionHeads",
    "LightweightRanker",
    "GBDTRanker",
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
    if name == "GBDTRanker":
        from .lightweight_ranker import GBDTRanker

        return GBDTRanker
    raise AttributeError(name)

from .model.ffn import FFN, FFN2, FFNResidual, FFNKai, FFNSiLU, OverfitFFN

mapper = {
    "ffn": FFN,
    "ffn2": FFN2,
    "ffn_residual": FFNResidual,
    "ffn_kai": FFNKai,
    "ffn_silu": FFNSiLU,
    "overfit_ffn": OverfitFFN,
}
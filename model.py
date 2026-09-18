"""
model.py — the REAL Late Fusion model, extracted from the team's actual
baseline2_latefusion.py (not from predict.py — predict.py loads
DNABERT2Baseline, the stage1 DNA-only model; the checkpoint this
project needs is stage2 Late Fusion, which takes sequence + epi_signal
+ tissue_id).

Only the pieces needed for a single-variant forward pass are kept here
(architecture + a load/predict helper) — training-specific code
(DataLoader, optimizer, scheduler) is deliberately left out.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig


class Config:
    MODEL_NAME     = "zhihan1996/DNABERT-2-117M"
    MAX_LENGTH     = 256
    HIDDEN_DIM     = 768
    EPI_N_CHANNELS = 2
    EPI_HIDDEN     = 128
    EPI_USE_LEN    = 1024
    CNN_KERNEL     = 7
    TISSUE_DIM     = 768
    N_TISSUES      = 3
    FUSION_HIDDEN  = 512
    DROPOUT        = 0.1


class EpigenomicEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden=128, kernel=7):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel, padding=kernel // 2),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2),
            nn.BatchNorm1d(hidden), nn.GELU(),
        )

    def forward(self, x):
        return self.cnn(x).mean(dim=2)  # GlobalAvgPool -> [B, epi_hidden]


class LateFusionModel(nn.Module):
    """Verbatim architecture from baseline2_latefusion.py — do not
    change shapes here without re-checking the real checkpoint loads."""

    def __init__(self, cfg):
        super().__init__()
        bert_cfg = BertConfig.from_pretrained(cfg.MODEL_NAME)
        self.dna_enc = AutoModel.from_pretrained(
            cfg.MODEL_NAME, trust_remote_code=True, config=bert_cfg
        )
        self.epi_enc = EpigenomicEncoder(cfg.EPI_N_CHANNELS, cfg.EPI_HIDDEN, cfg.CNN_KERNEL)
        self.tissue_emb = nn.Embedding(cfg.N_TISSUES, cfg.TISSUE_DIM)
        self.tissue_norm = nn.LayerNorm(cfg.TISSUE_DIM)

        fusion_in = cfg.HIDDEN_DIM + cfg.EPI_HIDDEN + cfg.TISSUE_DIM
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_in, cfg.FUSION_HIDDEN),
            nn.LayerNorm(cfg.FUSION_HIDDEN), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.FUSION_HIDDEN, 256), nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask, epi_signal, tissue_id):
        dna_out = self.dna_enc(input_ids=input_ids, attention_mask=attention_mask)
        hidden = dna_out[0] if isinstance(dna_out, tuple) else dna_out.last_hidden_state
        cls_vec = hidden[:, 0, :]
        epi_vec = self.epi_enc(epi_signal)
        tis_vec = self.tissue_norm(self.tissue_emb(tissue_id))
        fused = self.fusion_proj(torch.cat([cls_vec, epi_vec, tis_vec], dim=1))
        return self.classifier(fused).squeeze(-1)


_MODEL_CACHE: dict = {}  # avoid reloading the model on every tool call


def _get_model_and_tokenizer(checkpoint_path: str, device: str):
    """Loads once per checkpoint path and caches — DNABERT-2 is too
    large to reload on every predict_pathogenicity() call."""
    if checkpoint_path in _MODEL_CACHE:
        return _MODEL_CACHE[checkpoint_path]

    cfg = Config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
    model = LateFusionModel(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])  # matches baseline2's save key
    model.to(device)
    model.eval()

    _MODEL_CACHE[checkpoint_path] = (model, tokenizer, cfg)
    return model, tokenizer, cfg


def predict(
    sequence: str,
    h3k27ac,
    dnase,
    tissue_id: int,
    checkpoint_path: str,
    device: str = "cuda",
):
    """Single-variant inference against the REAL Late Fusion checkpoint.

    h3k27ac / dnase: length-1024 normalized signal arrays (from
    get_epigenomic_signal, already train-only z-scored).
    tissue_id: 0=liver, 1=heart, 2=brain (matches Config.TISSUE_TO_ID
    in tools.py).
    Returns (label, confidence) — confidence is the model's raw
    pathogenic probability (0-1), same as predict.py's convention.
    """
    model, tokenizer, cfg = _get_model_and_tokenizer(checkpoint_path, device)

    sequence = sequence.upper().strip()
    inputs = tokenizer(
        sequence, max_length=cfg.MAX_LENGTH, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    epi_signal = torch.tensor([h3k27ac[: cfg.EPI_USE_LEN], dnase[: cfg.EPI_USE_LEN]], dtype=torch.float).unsqueeze(0)
    tissue_tensor = torch.tensor([tissue_id], dtype=torch.long)

    with torch.no_grad():
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        epi_signal = epi_signal.to(device)
        tissue_tensor = tissue_tensor.to(device)
        logits = model(ids, mask, epi_signal, tissue_tensor)
        confidence = torch.sigmoid(logits).item()

    label = "pathogenic" if confidence >= 0.5 else "benign"
    return label, confidence

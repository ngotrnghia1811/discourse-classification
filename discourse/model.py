from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .config import BIO_LABELS, DISCOURSE_TYPES, DiscourseConfig

try:
    from torchcrf import CRF
    _CRF_AVAILABLE = True
except ImportError:
    _CRF_AVAILABLE = False


def _num_labels(config: DiscourseConfig) -> int:
    if config.model.label_scheme == "bio":
        return len(BIO_LABELS)
    return len(DISCOURSE_TYPES)


class LinearCRFHead(nn.Module):
    """
    Shared classification head used by both baseline and HDE models.
    Optionally wraps a linear layer with a CRF decoder.
    """

    def __init__(self, hidden_dim: int, num_labels: int, use_crf: bool, dropout: float):
        super().__init__()
        self.num_labels = num_labels
        self.use_crf = use_crf
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

        if use_crf:
            if not _CRF_AVAILABLE:
                raise ImportError(
                    "pytorch-crf is required for CRF decoding. "
                    "Install with: pip install pytorch-crf"
                )
            self.crf = CRF(num_labels, batch_first=True)

    def forward(
        self,
        hidden: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            hidden: [B, L, H]
            labels: [B, L] or None
            mask:   [B, L] bool/long — 1 for valid positions
        Returns:
            (loss, logits) where logits are [B, L, num_labels].
            When using CRF and no labels, logits contain raw emission scores
            (decoded sequences are returned via crf.decode separately).
        """
        h = self.dropout(hidden)
        logits = self.classifier(h)

        if self.use_crf:
            crf_mask = mask.bool() if mask is not None else None
            if labels is not None:
                crf_labels = labels.clone()
                crf_labels[crf_labels == -100] = 0
                loss = -self.crf(logits, crf_labels, mask=crf_mask, reduction="mean")
                return loss, logits
            else:
                decoded = self.crf.decode(logits, mask=crf_mask)
                return None, decoded

        return None, logits


class BERTForADP(nn.Module):
    """
    BERT (or BigBIRD) encoder with a token-level classification head.
    Supports optional CRF decoding for sequence labeling.

    Input:  tokenized essay (up to max_seq_len tokens)
    Output: per-token discourse label (BIO scheme)
    """

    def __init__(self, config: DiscourseConfig, tokenizer_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.model.encoder)

        if tokenizer_size is not None:
            self.encoder.resize_token_embeddings(tokenizer_size)

        enc_hidden = AutoConfig.from_pretrained(config.model.encoder).hidden_size
        num_labels = _num_labels(config)

        self.head = LinearCRFHead(
            hidden_dim=enc_hidden,
            num_labels=num_labels,
            use_crf=config.model.use_crf,
            dropout=config.model.dropout,
        )
        self.num_labels = num_labels
        self._loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state  # [B, L, H]

        loss, logits = self.head(seq_out, labels=labels, mask=attention_mask)

        if labels is not None and loss is None:
            active = attention_mask.view(-1) == 1
            act_logits = logits.view(-1, self.num_labels)
            act_labels = torch.where(
                active, labels.view(-1), torch.full_like(labels.view(-1), -100)
            )
            loss = self._loss_fn(act_logits, act_labels)

        return loss, logits


class SentenceEncoder(nn.Module):
    """
    Lightweight transformer encoder applied on top of sentence embeddings
    extracted by the segment-level BERT encoder.

    Captures cross-sentence context before final classification.
    """

    def __init__(
        self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float
    ):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos_embed = nn.Embedding(512, hidden_dim)

    def forward(
        self,
        sent_embeds: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = sent_embeds.shape
        pos = torch.arange(S, device=sent_embeds.device).unsqueeze(0).expand(B, -1)
        x = sent_embeds + self.pos_embed(pos)
        return self.transformer(x, src_key_padding_mask=key_padding_mask)


class HDEForADP(nn.Module):
    """
    Hierarchical Document Encoder (HDE) for sentence-level Argumentative
    Discourse Parsing.

    Architecture:
      1. Split essay into `num_segments` segments of `segment_len` tokens.
         Each segment contains [SENT] marker tokens at sentence boundaries.
      2. BERT encodes each segment independently (shared weights).
      3. [SENT] token hidden states are gathered as sentence embeddings.
      4. Optional sentence encoder (transformer) refines sentence representations.
      5. Linear + optional CRF head classifies each sentence.

    Supports label_scheme = "category" (7 classes) or "bio" (15 classes).
    """

    def __init__(self, config: DiscourseConfig, tokenizer_size: Optional[int] = None):
        super().__init__()
        self.config = config

        self.encoder = AutoModel.from_pretrained(config.model.encoder)
        if tokenizer_size is not None:
            self.encoder.resize_token_embeddings(tokenizer_size)

        enc_hidden = AutoConfig.from_pretrained(config.model.encoder).hidden_size
        num_labels = _num_labels(config)

        self.sentence_encoder: Optional[SentenceEncoder] = None
        if config.model.use_sentence_encoder:
            self.sentence_encoder = SentenceEncoder(
                hidden_dim=enc_hidden,
                num_layers=config.model.sentence_encoder_layers,
                num_heads=config.model.sentence_encoder_heads,
                dropout=config.model.dropout,
            )

        self.head = LinearCRFHead(
            hidden_dim=enc_hidden,
            num_labels=num_labels,
            use_crf=config.model.use_crf,
            dropout=config.model.dropout,
        )
        self.num_labels = num_labels
        self._loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def _extract_sentence_embeds(
        self,
        segment_input_ids: torch.Tensor,
        segment_attention_mask: torch.Tensor,
        sent_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run BERT on all segments and gather [SENT] token embeddings.

        Args:
            segment_input_ids:      [B, num_segments, seg_len]
            segment_attention_mask: [B, num_segments, seg_len]
            sent_positions:         [B, max_sentences, 2] — (seg_idx, tok_idx),
                                    -1 for padding

        Returns:
            sentence_embeds: [B, max_sentences, H]
        """
        B, num_segs, seg_len = segment_input_ids.shape

        flat_ids = segment_input_ids.view(B * num_segs, seg_len)
        flat_mask = segment_attention_mask.view(B * num_segs, seg_len)

        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        hidden = outputs.last_hidden_state  # [B*num_segs, seg_len, H]
        H = hidden.shape[-1]
        hidden = hidden.view(B, num_segs, seg_len, H)

        max_sent = sent_positions.shape[1]

        seg_idx = sent_positions[..., 0].clamp(min=0)  # [B, max_sent]
        tok_idx = sent_positions[..., 1].clamp(min=0)  # [B, max_sent]

        B_idx = torch.arange(B, device=hidden.device).unsqueeze(1).expand(B, max_sent)
        sentence_embeds = hidden[B_idx, seg_idx, tok_idx]  # [B, max_sent, H]

        padding = sent_positions[..., 0] < 0  # [B, max_sent]
        sentence_embeds = sentence_embeds.masked_fill(padding.unsqueeze(-1), 0.0)

        return sentence_embeds

    def forward(
        self,
        segment_input_ids: torch.Tensor,
        segment_attention_mask: torch.Tensor,
        sent_positions: torch.Tensor,
        sentence_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        sent_embeds = self._extract_sentence_embeds(
            segment_input_ids, segment_attention_mask, sent_positions
        )

        if self.sentence_encoder is not None:
            pad_mask = ~sentence_mask.bool()
            sent_embeds = self.sentence_encoder(sent_embeds, key_padding_mask=pad_mask)

        loss, logits = self.head(sent_embeds, labels=labels, mask=sentence_mask)

        if labels is not None and loss is None:
            active = sentence_mask.view(-1) == 1
            act_logits = logits.view(-1, self.num_labels)
            act_labels = torch.where(
                active, labels.view(-1), torch.full_like(labels.view(-1), -100)
            )
            loss = self._loss_fn(act_logits, act_labels)

        return loss, logits


def build_model(config: DiscourseConfig, tokenizer=None) -> nn.Module:
    """Factory: return the appropriate model for the given config."""
    tok_size = len(tokenizer) if tokenizer is not None else None
    if config.model.use_hde:
        return HDEForADP(config, tokenizer_size=tok_size)
    return BERTForADP(config, tokenizer_size=tok_size)

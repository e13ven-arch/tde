"""Build (tokenizer wrapper, model) from a backbone name and readout mode."""
from __future__ import annotations

import torch
from torch import nn

from tde.model.encoding import DecisionTokenizer
from tde.model.joint import JointDecisionModel
from tde.model.branch import BranchDecisionModel
from tde.model.biencoder import BiEncoderDecisionModel

READOUTS = {"joint": JointDecisionModel, "branch": BranchDecisionModel, "biencoder": BiEncoderDecisionModel}


def load_backbone(name_or_path: str, tiny: bool = False):
    """Return (tokenizer, backbone, hidden). `tiny=True` builds a random small BERT for tests."""
    from transformers import AutoModel, AutoTokenizer

    if tiny:
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
        from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
        words = ("a an the is are of on in at to and or not box sits table what which how big color colour size yes no "
                 "red green blue yellow black white tiny small medium large huge dark dim bright blinding sky today it rained "
                 "wet wrong right true false like sea premise hypothesis answer question").split()
        vocab_list = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words + [chr(c) for c in range(33, 127)] + ["##" + chr(c) for c in range(33, 127)]
        wp = Tokenizer(models.WordPiece({t: i for i, t in enumerate(vocab_list)}, unk_token="[UNK]"))
        wp.normalizer = normalizers.Lowercase()
        wp.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tok = PreTrainedTokenizerFast(tokenizer_object=wp, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                      sep_token="[SEP]", mask_token="[MASK]")
        cfg = BertConfig(vocab_size=len(vocab_list), hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                         intermediate_size=128, max_position_embeddings=1024)
        return tok, BertModel(cfg, add_pooling_layer=False), cfg.hidden_size
    tok = AutoTokenizer.from_pretrained(name_or_path)
    backbone = AutoModel.from_pretrained(name_or_path)
    hidden = backbone.config.hidden_size
    return tok, backbone, hidden


def build_model(backbone_name: str, readout: str, *, tiny: bool = False, use_confidence_head: bool = False,
                branch_layers: int = 3, max_state_tokens: int = 448) -> tuple[DecisionTokenizer, nn.Module]:
    tok, backbone, hidden = load_backbone(backbone_name, tiny=tiny)
    dtok = DecisionTokenizer(tok, max_state_tokens=max_state_tokens)
    if dtok.added_tokens:
        backbone.resize_token_embeddings(len(tok))
    cls = READOUTS[readout]
    if readout == "branch":
        heads = max(1, hidden // 64)
        model = cls(backbone, hidden, n_layers=branch_layers, n_heads=heads, use_confidence_head=use_confidence_head)
    else:
        model = cls(backbone, hidden, use_confidence_head=use_confidence_head) if readout == "joint" else cls(backbone, hidden)
    return dtok, model


def apply_finetune_mode(model: nn.Module, mode: str, lora_r: int = 16) -> dict:
    """full | frozen80 | lora. Returns a summary of trainable parameters."""
    backbone = model.backbone
    if mode == "full":
        pass
    elif mode == "frozen80":
        params = list(backbone.named_parameters())
        cut = int(len(params) * 0.8)
        for i, (_, p) in enumerate(params):
            p.requires_grad = i >= cut
    elif mode == "lora":
        from peft import LoraConfig, get_peft_model
        targets = [n.split(".")[-1] for n, m in backbone.named_modules() if isinstance(m, nn.Linear)]
        targets = sorted({t for t in targets if any(k in t for k in ("q", "k", "v", "query", "key", "value", "Wqkv", "dense", "Wo"))})
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, target_modules=targets)
        model.backbone = get_peft_model(backbone, cfg)
    else:
        raise ValueError(mode)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"mode": mode, "total_params": total, "trainable_params": trainable}


def pick_device(pref: str | None = None) -> torch.device:
    if pref and pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

"""Build (tokenizer wrapper, model) from a backbone name and readout mode."""
from __future__ import annotations

import torch
from torch import nn

from tde.model.encoding import DecisionTokenizer
from tde.model.joint import JointDecisionModel
from tde.model.branch import BranchDecisionModel
from tde.model.biencoder import BiEncoderDecisionModel
from tde.model.decoder import DecoderDecisionModel

READOUTS = {"joint": JointDecisionModel, "branch": BranchDecisionModel, "biencoder": BiEncoderDecisionModel, "decoder": DecoderDecisionModel}


def load_backbone(name_or_path: str, tiny: bool = False):
    """Return (tokenizer, backbone, hidden). `tiny=True` builds a random small BERT for tests."""
    from transformers import AutoModel, AutoTokenizer

    if tiny == "modernbert":
        from transformers import ModernBertConfig, ModernBertModel
        tok = _tiny_tokenizer()
        ids = {f"{t}_token_id": tok.convert_tokens_to_ids(f"[{t.upper()}]") for t in ("pad", "cls", "sep")}
        # initializer_range 0.2 gives peaked, position-sensitive attention, so layout bugs show up in tests
        cfg = ModernBertConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                               local_attention=8, global_attn_every_n_layers=3, max_position_embeddings=1024, initializer_range=0.2,
                               bos_token_id=ids["cls_token_id"], eos_token_id=ids["sep_token_id"], **ids)
        return tok, ModernBertModel(cfg), cfg.hidden_size
    if tiny == "causal" or (tiny and name_or_path == "tiny-causal"):
        from transformers import LlamaConfig, LlamaModel
        tok = _tiny_tokenizer()
        tok.eos_token = "[SEP]"
        cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                          intermediate_size=128, max_position_embeddings=1024)
        return tok, LlamaModel(cfg), cfg.hidden_size
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
    from transformers import AutoConfig
    tok = AutoTokenizer.from_pretrained(name_or_path)
    config = AutoConfig.from_pretrained(name_or_path)
    if hasattr(config, "reference_compile"):
        config.reference_compile = False  # ModernBERT's Triton path needs a C compiler; SDPA fallback is fine
    kwargs = {}
    if getattr(config, "is_decoder", False) or config.model_type in ("qwen3", "qwen2", "llama", "gemma3_text", "gemma3"):
        kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32  # decoders: bf16 weights to fit 8 GB
    backbone = AutoModel.from_pretrained(name_or_path, config=config, **kwargs)
    hidden = backbone.config.hidden_size
    return tok, backbone, hidden


def _tiny_tokenizer():
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    words = ("a an the is are of on in at to and or not box sits table what which how big color colour size yes no "
             "red green blue yellow black white tiny small medium large huge dark dim bright blinding sky today it rained "
             "wet wrong right true false like sea premise hypothesis answer question").split()
    vocab_list = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words + [chr(c) for c in range(33, 127)] + ["##" + chr(c) for c in range(33, 127)]
    wp = Tokenizer(models.WordPiece({t: i for i, t in enumerate(vocab_list)}, unk_token="[UNK]"))
    wp.normalizer = normalizers.Lowercase()
    wp.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    return PreTrainedTokenizerFast(tokenizer_object=wp, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                  sep_token="[SEP]", mask_token="[MASK]")


def build_model(backbone_name: str, readout: str, *, tiny: bool = False, use_confidence_head: bool = False,
                branch_layers: int = 3, max_state_tokens: int = 448, branch_through_backbone: bool = True,
                marker: str = "mask", pool: str = "marker+span", topology: str = "seq", max_total: int = 1024) -> tuple[DecisionTokenizer, nn.Module]:
    if topology != "seq" and readout != "joint":
        raise ValueError(f"topology={topology!r} is only defined for the joint readout")
    tok, backbone, hidden = load_backbone(backbone_name, tiny=tiny)
    dtok = DecisionTokenizer(tok, max_state_tokens=max_state_tokens, marker=marker, max_total=max_total)
    if dtok.added_tokens:
        backbone.resize_token_embeddings(len(tok))
    if readout == "decoder" and not tiny and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    cls = READOUTS[readout]
    if readout == "decoder":
        model = cls(backbone, hidden, use_confidence_head=use_confidence_head)
    elif readout == "branch":
        heads = max(1, hidden // 64)
        model = cls(backbone, hidden, n_layers=branch_layers, n_heads=heads, use_confidence_head=use_confidence_head,
                    branch_through_backbone=branch_through_backbone, pool=pool)
    else:
        model = cls(backbone, hidden, use_confidence_head=use_confidence_head, pool=pool, topology=topology) if readout == "joint" else cls(backbone, hidden)
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
        targets = sorted({t for t in targets if any(k in t for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                                                                       "query", "key", "value", "Wqkv", "dense", "Wo"))})
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

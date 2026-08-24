"""
models.py

One factory for every backbone in the study, exposing a uniform interface:
    net = build_model(name)          # returns a ClsNet on DEVICE
    net.set_trainable("full" | "encoder" | "head")
    logits = net(x)                  # always [B, num_classes]

Backbones (friendly name -> source):
    resnet50        timm  resnet50                    (BatchNorm)
    vit             timm  vit_base_patch16_224        (LayerNorm)
    convnext_tiny   timm  convnext_tiny               (LayerNorm)
    convnext_large  timm  convnext_large              (LayerNorm)
    segformer_v0    HF    nvidia/mit-b0   (SegFormer encoder as classifier)
    segformer_v5    HF    nvidia/mit-b5

All use 224x224 ImageNet-normalized input, so the existing rice_data.py
transforms work unchanged and preprocessing is held constant across
architectures — only the model varies.

NOTE on AdaBN: only resnet50 uses BatchNorm. For the LayerNorm backbones
(ViT / ConvNeXt / MiT) AdaBN is a no-op; the runner detects this via has_bn()
and skips it.

Requires: timm, transformers (only if a segformer_* model is used).
"""

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ALL pinned to ImageNet-1k so the architecture comparison is fair
# (timm defaults would mix in21k/in22k/in12k pretraining -> unfair confound).
TIMM_NAMES = {
    "resnet50":       "resnet50.a1_in1k",
    "vit":            "vit_base_patch16_224.augreg_in1k",
    "convnext_tiny":  "convnext_tiny.fb_in1k",
    "convnext_large": "convnext_large.fb_in1k",
}
HF_NAMES = {
    "segformer_v0": "nvidia/mit-b0",   # SegFormer encoder (MiT-B0), ImageNet-1k
    "segformer_v5": "nvidia/mit-b5",   # MiT-B5, ImageNet-1k
}
PRETRAIN_DATA = {k: "ImageNet-1k" for k in list(TIMM_NAMES) + list(HF_NAMES)}
ALL_MODELS = list(TIMM_NAMES) + list(HF_NAMES)

# MiT architecture configs (so we can build the graph WITHOUT hitting the Hub;
# pretrained weights are still fetched from HF when pretrained=True).
HF_CONFIGS = {
    "segformer_v0": dict(depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256],
                         decoder_hidden_size=256, num_attention_heads=[1, 2, 5, 8]),
    "segformer_v5": dict(depths=[3, 6, 40, 3], hidden_sizes=[64, 128, 320, 512],
                         decoder_hidden_size=768, num_attention_heads=[1, 2, 5, 8]),
}


class ClsNet(nn.Module):
    """Thin wrapper giving every backbone the same forward + freeze API."""

    def __init__(self, net, kind, head_module):
        super().__init__()
        self.net = net
        self.kind = kind                       # "timm" or "hf"
        self._head_ids = {id(p) for p in head_module.parameters()}

    def forward(self, x):
        if self.kind == "hf":
            return self.net(pixel_values=x).logits
        return self.net(x)

    def set_trainable(self, mode):
        head = [p for p in self.net.parameters() if id(p) in self._head_ids]
        rest = [p for p in self.net.parameters() if id(p) not in self._head_ids]
        if mode == "full":
            for p in self.net.parameters():
                p.requires_grad_(True)
        elif mode == "head":
            for p in rest:
                p.requires_grad_(False)
            for p in head:
                p.requires_grad_(True)
        elif mode == "encoder":
            for p in self.net.parameters():
                p.requires_grad_(True)
            for p in head:
                p.requires_grad_(False)
        else:
            raise ValueError(f"unknown freeze mode: {mode}")
        return self


def build_model(name, num_classes=4, pretrained=True):
    """
    pretrained=True  -> load ImageNet weights (for from-scratch / source training)
    pretrained=False -> fast random build, meant to be overwritten by load_state_dict
                        (timm only; HF always loads pretrained then you overwrite)
    """
    name = name.lower()

    if name in TIMM_NAMES:
        import timm
        net = timm.create_model(TIMM_NAMES[name], pretrained=pretrained,
                                num_classes=num_classes)
        return ClsNet(net, "timm", net.get_classifier()).to(DEVICE)

    if name in HF_NAMES:
        from transformers import SegformerForImageClassification, SegformerConfig
        if pretrained:
            net = SegformerForImageClassification.from_pretrained(
                HF_NAMES[name], num_labels=num_classes, ignore_mismatched_sizes=True)
        else:
            # architecture only, no Hub access (weights get loaded via load_state_dict)
            net = SegformerForImageClassification(
                SegformerConfig(num_labels=num_classes, **HF_CONFIGS[name]))
        return ClsNet(net, "hf", net.classifier).to(DEVICE)

    raise ValueError(f"unknown model '{name}'. options: {ALL_MODELS}")


def has_bn(model):
    return any(isinstance(m, nn.BatchNorm2d) for m in model.modules())


def model_info(name):
    """Params, fp32 size, and the exact pretrained checkpoint. No download needed."""
    m = build_model(name, pretrained=False)
    p = sum(x.numel() for x in m.parameters())
    src = TIMM_NAMES.get(name.lower()) or HF_NAMES.get(name.lower())
    return {
        "model": name,
        "params": int(p),
        "params_M": round(p / 1e6, 2),
        "size_mb_fp32": round(p * 4 / 1e6, 1),
        "pretrained_checkpoint": src,
        "pretrain_data": PRETRAIN_DATA.get(name.lower(), "unknown"),
    }


def n_params(model, trainable_only=False):
    ps = model.parameters()
    return sum(p.numel() for p in ps if (p.requires_grad or not trainable_only))


if __name__ == "__main__":
    # smoke test: build each, run a forward, check freeze modes
    x = torch.randn(2, 3, 224, 224).to(DEVICE)
    for name in ALL_MODELS:
        try:
            m = build_model(name, pretrained=False)
        except Exception as e:
            print(f"{name:16s} build FAILED: {e}")
            continue
        out = m(x)
        m.set_trainable("head");    h = n_params(m, True)
        m.set_trainable("encoder"); e = n_params(m, True)
        m.set_trainable("full");    f = n_params(m, True)
        print(f"{name:16s} logits={tuple(out.shape)} bn={has_bn(m)} "
              f"trainable head={h/1e6:.2f}M enc={e/1e6:.1f}M full={f/1e6:.1f}M")

from typing import Union, Type, List, Tuple

import torch
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
#from dynamic_network_architectures.building_blocks.unet_residual_decoder import UNetResDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class PlainConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


class ResidualEncoderUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


# ============================================================
# DeepConcat：每個 encoder/decoder stage 都 concat SynthSeg one-hot 再 1×1 conv fuse
# ============================================================

class ResidualEncoderUNet_DeepConcat(nn.Module):
    """Anatomy-aware ResidualEncoderUNet：每 stage 注入 SynthSeg one-hot。

    輸入 x: [B, image_channels + 1, ...]
        前 image_channels 是 DWI/ADC 等真實影像（被 encoder 處理）
        最後一個 channel 是 SynthSeg raw label (0..mask_classes-1) — 進 forward 後做 one-hot
    每個 encoder stage 輸出 feature [B, C_i, ...] 後：
        mask_oh 下採樣到該 stage 解析度（nearest），concat 變 [B, C_i + mask_classes, ...]，
        過 1×1 conv 降回 C_i。再丟入下一 stage / 當 skip。
    Decoder 同理：每 stage 的輸出做 concat-fuse 再傳下去。

    depth-agnostic：fuse 模組數量自動跟 n_stages 對齊，d5/d6 通用。
    """

    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 mask_classes: int = 10,
                 ):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        features_per_stage = list(features_per_stage)

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, \
            f"n_blocks_per_stage must have {n_stages} entries, got {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), \
            f"n_conv_per_stage_decoder must have {n_stages - 1} entries, got {n_conv_per_stage_decoder}"

        self.mask_classes = mask_classes
        # encoder 只看影像；mask 是最後一個 channel，自己處理
        self.image_channels = input_channels - 1
        assert self.image_channels >= 1, "input_channels 至少要 2（image + mask）"

        self.encoder = ResidualEncoder(
            self.image_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
            return_skips=True, disable_default_stem=False, stem_channels=stem_channels
        )
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        # 每個 encoder stage 後一個 1×1 conv fuse
        self.enc_fuse = nn.ModuleList([
            conv_op(features_per_stage[i] + mask_classes, features_per_stage[i],
                    kernel_size=1, bias=conv_bias)
            for i in range(n_stages)
        ])
        # 每個 decoder stage 後一個 1×1 conv fuse；decoder 有 n_stages-1 個 stage，
        # 第 s 個 stage 輸出對應 encoder stage[n_stages-2-s] 的解析度與通道
        self.dec_fuse = nn.ModuleList([
            conv_op(features_per_stage[n_stages - 2 - s] + mask_classes,
                    features_per_stage[n_stages - 2 - s],
                    kernel_size=1, bias=conv_bias)
            for s in range(n_stages - 1)
        ])

        self._spatial_dim = convert_conv_op_to_dim(conv_op)

    def _onehot_mask(self, mask_raw: torch.Tensor) -> torch.Tensor:
        """mask_raw: [B, 1, D, H, W]（或 2D）→ [B, mask_classes, D, H, W] float。

        Resampling 可能把 integer label 變浮點，所以 round + clamp 再 long。
        """
        m = mask_raw.squeeze(1).round().long().clamp(0, self.mask_classes - 1)
        oh = F.one_hot(m, num_classes=self.mask_classes)  # [B, ..., mask_classes]
        # 將最後一維 (one-hot class) 移到 channel 位
        dims = list(range(oh.dim()))
        new_order = [0, dims[-1]] + dims[1:-1]
        return oh.permute(*new_order).contiguous().to(mask_raw.dtype if mask_raw.is_floating_point()
                                                      else torch.float32)

    @staticmethod
    def _resample_to(mask_oh: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if mask_oh.shape[2:] == target.shape[2:]:
            return mask_oh
        return F.interpolate(mask_oh, size=target.shape[2:], mode='nearest')

    def forward(self, x):
        image = x[:, :self.image_channels]
        mask_raw = x[:, self.image_channels:self.image_channels + 1]
        mask_oh = self._onehot_mask(mask_raw)

        # ----- Encoder（手動展開，每 stage 後做 mask fuse）-----
        feat = image
        if self.encoder.stem is not None:
            feat = self.encoder.stem(feat)
        skips = []
        for i, stage in enumerate(self.encoder.stages):
            feat = stage(feat)
            mask_at = self._resample_to(mask_oh, feat)
            feat = self.enc_fuse[i](torch.cat([feat, mask_at], dim=1))
            skips.append(feat)

        # ----- Decoder（手動展開，每 stage 後做 mask fuse）-----
        lres = skips[-1]
        seg_outputs = []
        deep_sup = self.decoder.deep_supervision
        n_dec = len(self.decoder.stages)
        for s in range(n_dec):
            up = self.decoder.transpconvs[s](lres)
            cat = torch.cat([up, skips[-(s + 2)]], dim=1)
            out = self.decoder.stages[s](cat)
            mask_at = self._resample_to(mask_oh, out)
            out = self.dec_fuse[s](torch.cat([out, mask_at], dim=1))
            if deep_sup:
                seg_outputs.append(self.decoder.seg_layers[s](out))
            elif s == (n_dec - 1):
                seg_outputs.append(self.decoder.seg_layers[-1](out))
            lres = out
        seg_outputs = seg_outputs[::-1]
        return seg_outputs if deep_sup else seg_outputs[0]

    def compute_conv_feature_map_size(self, input_size):
        # fuse 是 1×1 conv，相對於 encoder/decoder 的特徵圖開銷可忽略 → 用 inner 模組估算
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return (self.encoder.compute_conv_feature_map_size(input_size)
                + self.decoder.compute_conv_feature_map_size(input_size))

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


# ============================================================
# SPADE — Spatially-Adaptive (De)Normalization
# Park et al. CVPR 2019, https://arxiv.org/abs/1903.07291
# ============================================================

class SPADEBlock(nn.Module):
    """SPADE modulation：用 anatomy mask 預測 spatial-varying (γ, β) 來調制 feature。

    形式（沿用官方 SPADE code 的 residual-style 寫法，γ 隨機初始化 ~ 0 時退化為 identity）：

        x_norm = InstanceNorm(x, no affine)
        h = ReLU(shared_conv(mask_oh))
        γ, β = conv_γ(h), conv_β(h)

        若 use_alpha=False (S3 vanilla SPADE):
            out = (1 + γ) × x_norm + β
        若 use_alpha=True  (S4/S5 gated SPADE, AdaLN-Zero 啟發):
            out = x_norm + α × (γ × x_norm + β)
            其中 α 是 per-channel learnable parameter、初始化 = 0
            → 第 0 epoch 等同 baseline InstanceNorm，mask 訊號 0 影響
            → model 自己決定要不要漸漸學進來

    Args:
        feature_channels: 待調制的 feature map channel 數
        mask_classes: mask one-hot 的 class 數（如 SynthSeg_merged 10）
        conv_op: nn.Conv2d / nn.Conv3d
        norm_op: nn.InstanceNorm2d / nn.InstanceNorm3d
        hidden_channels: shared conv 的中間 channel（預設 128）
        use_alpha: False = S3 vanilla, True = S4/S5 gated（zero-init α）
    """

    def __init__(
        self,
        feature_channels: int,
        mask_classes: int,
        conv_op: Type[_ConvNd],
        norm_op: Type[nn.Module],
        hidden_channels: int = 128,
        use_alpha: bool = False,
        alpha_init: float = 0.0,
    ):
        super().__init__()
        self.use_alpha = use_alpha
        self.alpha_init = alpha_init
        self.feature_channels = feature_channels
        self.norm = norm_op(feature_channels, affine=False)
        self.shared = conv_op(mask_classes, hidden_channels, kernel_size=3, padding=1)
        self.conv_gamma = conv_op(hidden_channels, feature_channels, kernel_size=3, padding=1)
        self.conv_beta = conv_op(hidden_channels, feature_channels, kernel_size=3, padding=1)
        if use_alpha:
            # per-channel α，init=0 → 起點 = baseline；init=0.3 → 有 mask 起點影響
            spatial_dim = convert_conv_op_to_dim(conv_op)
            shape = [1, feature_channels] + [1] * spatial_dim
            self.alpha = nn.Parameter(torch.full(shape, float(alpha_init)))

    def forward(self, x: torch.Tensor, mask_oh: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        h = F.relu(self.shared(mask_oh))
        gamma = self.conv_gamma(h)
        beta = self.conv_beta(h)
        if self.use_alpha:
            return x_norm + self.alpha * (gamma * x_norm + beta)
        else:
            return (1 + gamma) * x_norm + beta


class ResidualEncoderUNet_SPADE(nn.Module):
    """SPADE-based ResidualEncoderUNet（S3/S4/S5 共用）。

    Flags:
        inject_in_decoder: 每個 decoder stage 後接 SPADE（S3/S4/S5 都 True）
        inject_in_encoder: 每個 encoder stage 後接 SPADE（S5 才 True）
        use_alpha:         SPADE 是否套 α gating（S4/S5 True，S3 False）

    輸入跟 deepconcat 一樣：[B, image_channels + 1, ...]，最後 1 個 channel 是 SynthSeg label。
    forward 內 one-hot 成 [B, mask_classes, ...]、interp 到每 stage 解析度餵給 SPADE。
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
        mask_classes: int = 10,
        inject_in_decoder: bool = True,
        inject_in_encoder: bool = False,
        use_alpha: bool = False,
        spade_hidden_channels: int = 128,
        spade_alpha_init: float = 0.0,
    ):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        features_per_stage = list(features_per_stage)

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        self.mask_classes = mask_classes
        self.image_channels = input_channels - 1
        self.inject_in_decoder = inject_in_decoder
        self.inject_in_encoder = inject_in_encoder
        self.use_alpha = use_alpha
        assert self.image_channels >= 1, "input_channels 至少 2 (image + mask)"

        # encoder 只看影像
        self.encoder = ResidualEncoder(
            self.image_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
            return_skips=True, disable_default_stem=False, stem_channels=stem_channels
        )
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        # 用 nn.InstanceNormXd（不管 recipe 給什麼 norm_op）— SPADE 需要 affine=False 才能由 γ/β 接管
        spatial_dim = convert_conv_op_to_dim(conv_op)
        spade_norm_op = {2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[spatial_dim]

        def _spade(c):
            return SPADEBlock(
                feature_channels=c,
                mask_classes=mask_classes,
                conv_op=conv_op,
                norm_op=spade_norm_op,
                hidden_channels=spade_hidden_channels,
                use_alpha=use_alpha,
                alpha_init=spade_alpha_init,
            )

        if inject_in_encoder:
            self.enc_spade = nn.ModuleList([_spade(features_per_stage[i]) for i in range(n_stages)])
        else:
            self.enc_spade = None
        if inject_in_decoder:
            # decoder 有 n_stages-1 stage；stage s 輸出對應 features_per_stage[n_stages-2-s]
            self.dec_spade = nn.ModuleList([
                _spade(features_per_stage[n_stages - 2 - s]) for s in range(n_stages - 1)
            ])
        else:
            self.dec_spade = None

    def _onehot_mask(self, mask_raw: torch.Tensor) -> torch.Tensor:
        m = mask_raw.squeeze(1).round().long().clamp(0, self.mask_classes - 1)
        oh = F.one_hot(m, num_classes=self.mask_classes)
        dims = list(range(oh.dim()))
        new_order = [0, dims[-1]] + dims[1:-1]
        return oh.permute(*new_order).contiguous().to(
            mask_raw.dtype if mask_raw.is_floating_point() else torch.float32
        )

    @staticmethod
    def _resample(mask_oh: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if mask_oh.shape[2:] == target.shape[2:]:
            return mask_oh
        return F.interpolate(mask_oh, size=target.shape[2:], mode='nearest')

    def forward(self, x):
        image = x[:, :self.image_channels]
        mask_raw = x[:, self.image_channels:self.image_channels + 1]
        mask_oh = self._onehot_mask(mask_raw)

        # ----- Encoder -----
        feat = image
        if self.encoder.stem is not None:
            feat = self.encoder.stem(feat)
        skips = []
        for i, stage in enumerate(self.encoder.stages):
            feat = stage(feat)
            if self.inject_in_encoder:
                m = self._resample(mask_oh, feat)
                feat = self.enc_spade[i](feat, m)
            skips.append(feat)

        # ----- Decoder -----
        lres = skips[-1]
        seg_outputs = []
        deep_sup = self.decoder.deep_supervision
        n_dec = len(self.decoder.stages)
        for s in range(n_dec):
            up = self.decoder.transpconvs[s](lres)
            cat = torch.cat([up, skips[-(s + 2)]], dim=1)
            out = self.decoder.stages[s](cat)
            if self.inject_in_decoder:
                m = self._resample(mask_oh, out)
                out = self.dec_spade[s](out, m)
            if deep_sup:
                seg_outputs.append(self.decoder.seg_layers[s](out))
            elif s == (n_dec - 1):
                seg_outputs.append(self.decoder.seg_layers[-1](out))
            lres = out
        seg_outputs = seg_outputs[::-1]
        return seg_outputs if deep_sup else seg_outputs[0]

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return (self.encoder.compute_conv_feature_map_size(input_size)
                + self.decoder.compute_conv_feature_map_size(input_size))

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


# ============================================================
# 3 個 thin wrapper：S3 / S4 / S5（差別只在 inject_in_encoder + use_alpha）
# ============================================================

class ResidualEncoderUNet_SPADEDecoder(ResidualEncoderUNet_SPADE):
    """S3：encoder 不動、decoder 每 stage 套 vanilla SPADE（無 α gating）。"""
    def __init__(self, *args, **kwargs):
        kwargs["inject_in_decoder"] = True
        kwargs["inject_in_encoder"] = False
        kwargs["use_alpha"] = False
        super().__init__(*args, **kwargs)


class ResidualEncoderUNet_SPADEDecoderAlpha(ResidualEncoderUNet_SPADE):
    """S4：encoder 不動、decoder 每 stage 套 SPADE + α gating（zero-init，訓練起點 = baseline）。"""
    def __init__(self, *args, **kwargs):
        kwargs["inject_in_decoder"] = True
        kwargs["inject_in_encoder"] = False
        kwargs["use_alpha"] = True
        super().__init__(*args, **kwargs)


class ResidualEncoderUNet_SPADEFull(ResidualEncoderUNet_SPADE):
    """S5：encoder + decoder 每 stage 都套 SPADE + α gating（最完整、參數最多）。"""
    def __init__(self, *args, **kwargs):
        kwargs["inject_in_decoder"] = True
        kwargs["inject_in_encoder"] = True
        kwargs["use_alpha"] = True
        super().__init__(*args, **kwargs)


# ============================================================
# Dual-Seg-Head：joint multi-task (主 head infarct + 輔 head SynthSeg region)
# ============================================================

class ResidualEncoderUNet_DualSegHead(nn.Module):
    """雙 segmentation head — 共用 encoder + decoder，最後接兩個獨立 head。

    主 head: infarct 分割（num_classes 個 class，通常 2 = bg/infarct）
    輔 head: SynthSeg region 分割（num_classes_aux 個 class，例：SynthSeg_merged = 10，0=bg + 1-9 regions）

    Forward 回傳 tuple (main_logits, aux_logits)：
      - deep_supervision=False: (main, aux)，各 shape [B, C, ...]
      - deep_supervision=True : (main_list, aux_list)，各為 list of [B, C_i, ...]

    Loss 由 trainer 算：L = L_main(main, gt_main) + λ * L_aux(aux, gt_aux)。
    Inference 時用 main head 結果評估 infarct（同 baseline）。
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
        num_classes_aux: int = 11,
    ):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        features_per_stage = list(features_per_stage)
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        self.num_classes_aux = num_classes_aux

        # encoder + decoder 跟 vanilla 一樣
        self.encoder = ResidualEncoder(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
            return_skips=True, disable_default_stem=False, stem_channels=stem_channels,
        )
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        # 輔助 head：跟 decoder.seg_layers 平行，channels 跟著 decoder stage 的輸出
        # decoder 有 (n_stages - 1) 個 stage；stage s 輸出 features_per_stage[n_stages-2-s]
        self.aux_seg_layers = nn.ModuleList([
            conv_op(features_per_stage[n_stages - 2 - s], num_classes_aux, kernel_size=1, bias=True)
            for s in range(n_stages - 1)
        ])

    def forward(self, x):
        skips = self.encoder(x)

        lres = skips[-1]
        main_outputs = []
        aux_outputs = []
        deep_sup = self.decoder.deep_supervision
        n_dec = len(self.decoder.stages)
        for s in range(n_dec):
            up = self.decoder.transpconvs[s](lres)
            cat = torch.cat([up, skips[-(s + 2)]], dim=1)
            out = self.decoder.stages[s](cat)
            if deep_sup:
                main_outputs.append(self.decoder.seg_layers[s](out))
                aux_outputs.append(self.aux_seg_layers[s](out))
            elif s == (n_dec - 1):
                main_outputs.append(self.decoder.seg_layers[-1](out))
                aux_outputs.append(self.aux_seg_layers[-1](out))
            lres = out

        main_outputs = main_outputs[::-1]
        aux_outputs = aux_outputs[::-1]
        if deep_sup:
            return main_outputs, aux_outputs
        return main_outputs[0], aux_outputs[0]

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return (self.encoder.compute_conv_feature_map_size(input_size)
                + self.decoder.compute_conv_feature_map_size(input_size))

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


"""
class ResidualUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)
"""

class EasyResidualBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_ratio=4, dropout=0.0):
        super().__init__()
        hidden_channels = in_channels // bottleneck_ratio

        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(hidden_channels)
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(hidden_channels)
        self.conv3 = nn.Conv3d(hidden_channels, in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(in_channels)

        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.dropout(out)
        return self.relu(out + identity)


class Classifier(nn.Module):
    def __init__(self, encoder, features_per_stage, num_classes, num_blocks=2, bottleneck_ratio=4, dropout=0.1, num_denses=3, min_dim=32):
        super().__init__()
        self.bottleneck_channels = features_per_stage[-1]

        self.res_blocks = nn.Sequential(
            *[EasyResidualBlock(self.bottleneck_channels, bottleneck_ratio, dropout)
              for _ in range(num_blocks)]
        )

        #3次，每次/2 => 512 => 256 => 128 => 64
        #3次，每次/4 => 512 => 128 => 32 => 8
        dense_layers = []
        in_dim = self.bottleneck_channels
        for _ in range(num_denses):
            out_dim = max(in_dim // 2, min_dim)
            dense_layers.append(nn.Linear(in_dim, out_dim))
            dense_layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
            dense_layers.append(nn.Dropout(p=dropout))
            in_dim = out_dim

        dense_layers.append(nn.Linear(in_dim, num_classes))  # 最終輸出

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            *dense_layers
        )

    def forward(self, skips):
        x = skips[-1]
        x = self.res_blocks(x)
        return self.classifier(x)


class ResidualEncoderUNetClassifier(nn.Module):
    def __init__(self,        
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 classifier_num_classes: int = 5,
                 classifier_num_blocks: int = 2,
                 classifier_num_denses: int = 3,
                 classifier_min_dim: int = 32,
                 classifier_dropout: float = 0.1,
                 classifier_bottleneck_ratio: int = 4,
                 ):

        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        #這邊新增classifier head網路
        #輸出是 raw logits (5類: 0=無動脈瘤, 1-4=四個location)
        self.classifier_head = Classifier(encoder=self.encoder,
                                          features_per_stage=features_per_stage,  # <--- 傳進來
                                          num_classes=classifier_num_classes,
                                          num_blocks=classifier_num_blocks,
                                          bottleneck_ratio=classifier_bottleneck_ratio,
                                          dropout=classifier_dropout,
                                          num_denses=classifier_num_denses,
                                          min_dim=classifier_min_dim)

    def forward(self, x):
        skips = self.encoder(x)
        seg_output = self.decoder(skips)
        cls_output = self.classifier_head(skips)
        return seg_output, cls_output


    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class EasyResidualBlock2D(nn.Module):
    def __init__(self, in_channels, bottleneck_ratio=4, dropout=0.0):
        super().__init__()
        hidden_channels = in_channels // bottleneck_ratio

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)

        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.dropout(out)
        return self.relu(out + identity)


class Classifier2D(nn.Module):
    def __init__(self, encoder, features_per_stage, num_classes, num_blocks=2, bottleneck_ratio=4, dropout=0.1, num_denses=3, min_dim=32):
        super().__init__()
        self.bottleneck_channels = features_per_stage[-1]

        self.res_blocks = nn.Sequential(
            *[EasyResidualBlock2D(self.bottleneck_channels, bottleneck_ratio, dropout)
              for _ in range(num_blocks)]
        )

        #3次，每次/2 => 512 => 256 => 128 => 64
        #3次，每次/4 => 512 => 128 => 32 => 8
        dense_layers = []
        in_dim = self.bottleneck_channels
        for _ in range(num_denses):
            out_dim = max(in_dim // 2, min_dim)
            dense_layers.append(nn.Linear(in_dim, out_dim))
            dense_layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
            dense_layers.append(nn.Dropout(p=dropout))
            in_dim = out_dim

        dense_layers.append(nn.Linear(in_dim, num_classes))  # 最終輸出

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            *dense_layers
        )

    def forward(self, skips):
        x = skips[-1]
        x = self.res_blocks(x)
        return self.classifier(x)

class ResidualEncoderUNetClassifier2D(nn.Module):
    def __init__(self,        
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 classifier_num_classes: int = 5,
                 classifier_num_blocks: int = 2,
                 classifier_num_denses: int = 3,
                 classifier_min_dim: int = 32,
                 classifier_dropout: float = 0.1,
                 classifier_bottleneck_ratio: int = 4,
                 ):

        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        #這邊新增classifier head網路
        #輸出是 raw logits (5類: 0=無動脈瘤, 1-4=四個location)
        self.classifier_head = Classifier2D(encoder=self.encoder,
                                          features_per_stage=features_per_stage,  # <--- 傳進來
                                          num_classes=classifier_num_classes,
                                          num_blocks=classifier_num_blocks,
                                          bottleneck_ratio=classifier_bottleneck_ratio,
                                          dropout=classifier_dropout,
                                          num_denses=classifier_num_denses,
                                          min_dim=classifier_min_dim)

    def forward(self, x):
        skips = self.encoder(x)
        seg_output = self.decoder(skips)
        cls_output = self.classifier_head(skips)
        return seg_output, cls_output


    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class CrossAttentionPooling3D(nn.Module):
    """RSNA 風格的交叉注意力池化：用可學習的查詢向量對特徵圖做 cross-attention，
    再將多個查詢的輸出展平後經線性層分類。"""
    def __init__(self, embed_dim, query_num, num_classes, num_heads=4, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.query_num = query_num

        self.class_query = nn.Parameter(torch.randn(query_num, embed_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(query_num * embed_dim, num_classes)

        nn.init.xavier_uniform_(self.class_query)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        if x.dim() == 5:
            x = x.flatten(2)  # [B, D, H*W*L]
        x = x.permute(2, 0, 1)  # [S, B, D]
        query = self.class_query.unsqueeze(1).expand(-1, batch_size, -1)  # [Q, B, D]
        attended, _ = self.cross_attention(query=query, key=x, value=x)
        attended = self.norm(attended)
        attended = self.dropout(attended)
        attended = attended.permute(1, 0, 2).flatten(1)  # [B, Q*D]
        return self.classifier(attended)


class CrossAttentionPooling2D(nn.Module):
    """2D 版本的交叉注意力池化。"""
    def __init__(self, embed_dim, query_num, num_classes, num_heads=4, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.query_num = query_num

        self.class_query = nn.Parameter(torch.randn(query_num, embed_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(query_num * embed_dim, num_classes)

        nn.init.xavier_uniform_(self.class_query)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        if x.dim() == 4:
            x = x.flatten(2)  # [B, D, H*W]
        x = x.permute(2, 0, 1)  # [S, B, D]
        query = self.class_query.unsqueeze(1).expand(-1, batch_size, -1)
        attended, _ = self.cross_attention(query=query, key=x, value=x)
        attended = self.norm(attended)
        attended = self.dropout(attended)
        attended = attended.permute(1, 0, 2).flatten(1)
        return self.classifier(attended)


class AttentionClassifier(nn.Module):
    """RSNA 風格的注意力分類頭（3D），使用 CrossAttentionPooling 從 encoder bottleneck 特徵做分類。"""
    def __init__(self, features_per_stage, num_classes, query_num=4, num_heads=4, dropout=0.0):
        super().__init__()
        self.pooling = CrossAttentionPooling3D(
            embed_dim=features_per_stage[-1],
            query_num=query_num,
            num_classes=num_classes,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, skips):
        return self.pooling(skips[-1])


class AttentionClassifier2D(nn.Module):
    """RSNA 風格的注意力分類頭（2D），使用 CrossAttentionPooling 從 encoder bottleneck 特徵做分類。"""
    def __init__(self, features_per_stage, num_classes, query_num=4, num_heads=4, dropout=0.0):
        super().__init__()
        self.pooling = CrossAttentionPooling2D(
            embed_dim=features_per_stage[-1],
            query_num=query_num,
            num_classes=num_classes,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, skips):
        return self.pooling(skips[-1])


class ResidualEncoderUNetAttentionClassifier(nn.Module):
    """ResidualEncoder UNet + RSNA 風格 Cross-Attention 分類頭（3D）。
    與 ResidualEncoderUNetClassifier 的差別在於分類頭使用注意力機制而非 CNN。"""
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 classifier_num_classes: int = 5,
                 classifier_query_num: int = 4,
                 classifier_num_heads: int = 4,
                 classifier_dropout: float = 0.0,
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        self.classifier_head = AttentionClassifier(
            features_per_stage=features_per_stage,
            num_classes=classifier_num_classes,
            query_num=classifier_query_num,
            num_heads=classifier_num_heads,
            dropout=classifier_dropout
        )

    def forward(self, x):
        skips = self.encoder(x)
        seg_output = self.decoder(skips)
        cls_output = self.classifier_head(skips)
        return seg_output, cls_output

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class ResidualEncoderUNetAttentionClassifier2D(nn.Module):
    """ResidualEncoder UNet + RSNA 風格 Cross-Attention 分類頭（2D）。"""
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 classifier_num_classes: int = 5,
                 classifier_query_num: int = 4,
                 classifier_num_heads: int = 4,
                 classifier_dropout: float = 0.0,
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        self.classifier_head = AttentionClassifier2D(
            features_per_stage=features_per_stage,
            num_classes=classifier_num_classes,
            query_num=classifier_query_num,
            num_heads=classifier_num_heads,
            dropout=classifier_dropout
        )

    def forward(self, x):
        skips = self.encoder(x)
        seg_output = self.decoder(skips)
        cls_output = self.classifier_head(skips)
        return seg_output, cls_output

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


# ============================================================
# Guided 版本：分類特徵透過 FiLM 注入 decoder，幫助降低 seg FP
# ============================================================

class CrossAttentionPoolingWithFeatures3D(nn.Module):
    """與 CrossAttentionPooling3D 相同，但同時回傳中間特徵供 FiLM 使用。
    回傳 (cls_logits, cls_features)。"""
    def __init__(self, embed_dim, query_num, num_classes, num_heads=4, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_num = query_num

        self.class_query = nn.Parameter(torch.randn(query_num, embed_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(query_num * embed_dim, num_classes)

        nn.init.xavier_uniform_(self.class_query)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        if x.dim() == 5:
            x = x.flatten(2)
        x = x.permute(2, 0, 1)
        query = self.class_query.unsqueeze(1).expand(-1, batch_size, -1)
        attended, _ = self.cross_attention(query=query, key=x, value=x)
        attended = self.norm(attended)
        attended = self.dropout(attended)
        cls_features = attended.permute(1, 0, 2).flatten(1)  # [B, Q*D]
        cls_logits = self.classifier(cls_features)            # [B, num_classes]
        return cls_logits, cls_features


class CrossAttentionPoolingWithFeatures2D(nn.Module):
    """2D 版本，同時回傳 (cls_logits, cls_features)。"""
    def __init__(self, embed_dim, query_num, num_classes, num_heads=4, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_num = query_num

        self.class_query = nn.Parameter(torch.randn(query_num, embed_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(query_num * embed_dim, num_classes)

        nn.init.xavier_uniform_(self.class_query)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        if x.dim() == 4:
            x = x.flatten(2)
        x = x.permute(2, 0, 1)
        query = self.class_query.unsqueeze(1).expand(-1, batch_size, -1)
        attended, _ = self.cross_attention(query=query, key=x, value=x)
        attended = self.norm(attended)
        attended = self.dropout(attended)
        cls_features = attended.permute(1, 0, 2).flatten(1)
        cls_logits = self.classifier(cls_features)
        return cls_logits, cls_features


class FiLMConditioner(nn.Module):
    """FiLM (Feature-wise Linear Modulation)：
    用分類特徵向量對每層 skip connection 做 scale + shift，
    讓 decoder 帶著「這是哪種血管」的資訊去分割。

    對每層 skip[i] (shape [B, C_i, ...])：
        gamma_i, beta_i = Linear(cls_features)   # [B, C_i] each
        skip[i] = skip[i] * (1 + gamma_i) + beta_i

    (1 + gamma) 確保初始化時 gamma≈0 → 退化為恆等映射，不破壞原有行為。
    """
    def __init__(self, cls_feature_dim: int, features_per_stage: List[int]):
        super().__init__()
        self.film_layers = nn.ModuleList()
        for c in features_per_stage:
            layer = nn.Linear(cls_feature_dim, c * 2)
            # 初始化接近 0，讓訓練初期不干擾原有分割行為
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.film_layers.append(layer)

    def forward(self, cls_features, skips):
        """
        cls_features: [B, cls_feature_dim]
        skips: list of tensors [B, C_i, ...] (從淺到深)
        returns: conditioned skips (同樣順序)
        """
        conditioned = []
        for skip, film in zip(skips, self.film_layers):
            gb = film(cls_features)        # [B, C_i * 2]
            C = skip.shape[1]
            gamma = gb[:, :C]              # [B, C]
            beta = gb[:, C:]               # [B, C]
            # reshape for spatial broadcasting: [B, C] → [B, C, 1, 1, 1] 或 [B, C, 1, 1]
            shape = [skip.shape[0], C] + [1] * (skip.dim() - 2)
            gamma = gamma.view(*shape)
            beta = beta.view(*shape)
            conditioned.append(skip * (1 + gamma) + beta)
        return conditioned


class ResidualEncoderUNetGuidedClassifier(nn.Module):
    """ResidualEncoder UNet + Cross-Attention 分類頭 + FiLM 注入 decoder（3D）。
    分類特徵透過 FiLM 調制所有 skip connections，讓 decoder 知道血管類型，降低 seg FP。

    架構：
        encoder → skips
                    ↓
        CrossAttention(skips[-1]) → cls_logits, cls_features
                    ↓
        FiLM(cls_features) 調制每層 skip
                    ↓
        decoder(conditioned_skips) → seg_output
    """
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 classifier_num_classes: int = 5,
                 classifier_query_num: int = 4,
                 classifier_num_heads: int = 4,
                 classifier_dropout: float = 0.0,
                 ):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        self.features_per_stage = list(features_per_stage)

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        # 分類頭：回傳 (cls_logits, cls_features)
        cls_feature_dim = classifier_query_num * self.features_per_stage[-1]
        self.classifier_head = CrossAttentionPoolingWithFeatures3D(
            embed_dim=self.features_per_stage[-1],
            query_num=classifier_query_num,
            num_classes=classifier_num_classes,
            num_heads=classifier_num_heads,
            dropout=classifier_dropout
        )

        # FiLM：用 cls_features 調制每層 skip
        self.film = FiLMConditioner(
            cls_feature_dim=cls_feature_dim,
            features_per_stage=self.features_per_stage
        )

    def forward(self, x):
        skips = self.encoder(x)

        # 1. 分類：從 bottleneck 取得分類結果 + 中間特徵
        cls_logits, cls_features = self.classifier_head(skips[-1])

        # 2. FiLM 調制：用分類特徵調整所有 skip connections
        conditioned_skips = self.film(cls_features, skips)

        # 3. Decoder：帶著血管類型資訊解碼
        seg_output = self.decoder(conditioned_skips)

        return seg_output, cls_logits

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class ResidualEncoderUNetGuidedClassifier2D(nn.Module):
    """2D 版本的 Guided Classifier。"""
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 classifier_num_classes: int = 5,
                 classifier_query_num: int = 4,
                 classifier_num_heads: int = 4,
                 classifier_dropout: float = 0.0,
                 ):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        self.features_per_stage = list(features_per_stage)

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        cls_feature_dim = classifier_query_num * self.features_per_stage[-1]
        self.classifier_head = CrossAttentionPoolingWithFeatures2D(
            embed_dim=self.features_per_stage[-1],
            query_num=classifier_query_num,
            num_classes=classifier_num_classes,
            num_heads=classifier_num_heads,
            dropout=classifier_dropout
        )

        self.film = FiLMConditioner(
            cls_feature_dim=cls_feature_dim,
            features_per_stage=self.features_per_stage
        )

    def forward(self, x):
        skips = self.encoder(x)
        cls_logits, cls_features = self.classifier_head(skips[-1])
        conditioned_skips = self.film(cls_features, skips)
        seg_output = self.decoder(conditioned_skips)
        return seg_output, cls_logits

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


# ============================================================
# S7-S10 hybrids: mask-fusion backbone (DeepConcat/SPADEDecoder) + classifier head
# 輸入 x: [B, image_channels + 1, ...]  最後 channel = raw label (vessel8 for aneurysm)
# forward: 回傳 (seg_output, cls_logits) — 讓 trainer 走 has_cls_head 路徑
# ============================================================


class ResidualEncoderUNet_DeepConcat_AttentionClassifier(nn.Module):
    """S7: DeepConcat backbone + Cross-Attention classifier head。
    encoder/decoder 每 stage 後 concat mask one-hot → 1×1 conv fuse。
    classifier 讀 encoder 最深 skip（已經帶 mask 資訊）。
    """
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 mask_classes: int = 10,
                 classifier_num_classes: int = 5,
                 classifier_query_num: int = 4,
                 classifier_num_heads: int = 4,
                 classifier_dropout: float = 0.0,
                 ):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        features_per_stage = list(features_per_stage)
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        self.mask_classes = mask_classes
        self.image_channels = input_channels - 1
        assert self.image_channels >= 1

        self.encoder = ResidualEncoder(
            self.image_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
            return_skips=True, disable_default_stem=False, stem_channels=stem_channels
        )
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        self.enc_fuse = nn.ModuleList([
            conv_op(features_per_stage[i] + mask_classes, features_per_stage[i],
                    kernel_size=1, bias=conv_bias)
            for i in range(n_stages)
        ])
        self.dec_fuse = nn.ModuleList([
            conv_op(features_per_stage[n_stages - 2 - s] + mask_classes,
                    features_per_stage[n_stages - 2 - s],
                    kernel_size=1, bias=conv_bias)
            for s in range(n_stages - 1)
        ])

        self.classifier_head = AttentionClassifier(
            features_per_stage=features_per_stage,
            num_classes=classifier_num_classes,
            query_num=classifier_query_num,
            num_heads=classifier_num_heads,
            dropout=classifier_dropout,
        )

    def _onehot_mask(self, mask_raw: torch.Tensor) -> torch.Tensor:
        m = mask_raw.squeeze(1).round().long().clamp(0, self.mask_classes - 1)
        oh = F.one_hot(m, num_classes=self.mask_classes)
        dims = list(range(oh.dim()))
        new_order = [0, dims[-1]] + dims[1:-1]
        return oh.permute(*new_order).contiguous().to(
            mask_raw.dtype if mask_raw.is_floating_point() else torch.float32
        )

    @staticmethod
    def _resample_to(mask_oh: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if mask_oh.shape[2:] == target.shape[2:]:
            return mask_oh
        return F.interpolate(mask_oh, size=target.shape[2:], mode='nearest')

    def forward(self, x):
        image = x[:, :self.image_channels]
        mask_raw = x[:, self.image_channels:self.image_channels + 1]
        mask_oh = self._onehot_mask(mask_raw)

        feat = image
        if self.encoder.stem is not None:
            feat = self.encoder.stem(feat)
        skips = []
        for i, stage in enumerate(self.encoder.stages):
            feat = stage(feat)
            mask_at = self._resample_to(mask_oh, feat)
            feat = self.enc_fuse[i](torch.cat([feat, mask_at], dim=1))
            skips.append(feat)

        cls_output = self.classifier_head(skips)

        lres = skips[-1]
        seg_outputs = []
        deep_sup = self.decoder.deep_supervision
        n_dec = len(self.decoder.stages)
        for s in range(n_dec):
            up = self.decoder.transpconvs[s](lres)
            cat = torch.cat([up, skips[-(s + 2)]], dim=1)
            out = self.decoder.stages[s](cat)
            mask_at = self._resample_to(mask_oh, out)
            out = self.dec_fuse[s](torch.cat([out, mask_at], dim=1))
            if deep_sup:
                seg_outputs.append(self.decoder.seg_layers[s](out))
            elif s == (n_dec - 1):
                seg_outputs.append(self.decoder.seg_layers[-1](out))
            lres = out
        seg_outputs = seg_outputs[::-1]
        seg_out = seg_outputs if deep_sup else seg_outputs[0]
        return seg_out, cls_output

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return (self.encoder.compute_conv_feature_map_size(input_size)
                + self.decoder.compute_conv_feature_map_size(input_size))

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class ResidualEncoderUNet_DeepConcat_GuidedClassifier(nn.Module):
    """S8: DeepConcat backbone + Guided (Cross-Attention + FiLM) classifier。
    分類特徵透過 FiLM 調制所有 skip → 影響 decoder seg 分佈（降低 FP）。
    """
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 mask_classes: int = 10,
                 classifier_num_classes: int = 5,
                 classifier_query_num: int = 4,
                 classifier_num_heads: int = 4,
                 classifier_dropout: float = 0.0,
                 ):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        features_per_stage = list(features_per_stage)
        self.features_per_stage = features_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        self.mask_classes = mask_classes
        self.image_channels = input_channels - 1
        assert self.image_channels >= 1

        self.encoder = ResidualEncoder(
            self.image_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
            return_skips=True, disable_default_stem=False, stem_channels=stem_channels
        )
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        self.enc_fuse = nn.ModuleList([
            conv_op(features_per_stage[i] + mask_classes, features_per_stage[i],
                    kernel_size=1, bias=conv_bias)
            for i in range(n_stages)
        ])
        self.dec_fuse = nn.ModuleList([
            conv_op(features_per_stage[n_stages - 2 - s] + mask_classes,
                    features_per_stage[n_stages - 2 - s],
                    kernel_size=1, bias=conv_bias)
            for s in range(n_stages - 1)
        ])

        cls_feature_dim = classifier_query_num * features_per_stage[-1]
        self.classifier_head = CrossAttentionPoolingWithFeatures3D(
            embed_dim=features_per_stage[-1],
            query_num=classifier_query_num,
            num_classes=classifier_num_classes,
            num_heads=classifier_num_heads,
            dropout=classifier_dropout,
        )
        self.film = FiLMConditioner(
            cls_feature_dim=cls_feature_dim,
            features_per_stage=features_per_stage,
        )

    def _onehot_mask(self, mask_raw: torch.Tensor) -> torch.Tensor:
        m = mask_raw.squeeze(1).round().long().clamp(0, self.mask_classes - 1)
        oh = F.one_hot(m, num_classes=self.mask_classes)
        dims = list(range(oh.dim()))
        new_order = [0, dims[-1]] + dims[1:-1]
        return oh.permute(*new_order).contiguous().to(
            mask_raw.dtype if mask_raw.is_floating_point() else torch.float32
        )

    @staticmethod
    def _resample_to(mask_oh: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if mask_oh.shape[2:] == target.shape[2:]:
            return mask_oh
        return F.interpolate(mask_oh, size=target.shape[2:], mode='nearest')

    def forward(self, x):
        image = x[:, :self.image_channels]
        mask_raw = x[:, self.image_channels:self.image_channels + 1]
        mask_oh = self._onehot_mask(mask_raw)

        feat = image
        if self.encoder.stem is not None:
            feat = self.encoder.stem(feat)
        skips = []
        for i, stage in enumerate(self.encoder.stages):
            feat = stage(feat)
            mask_at = self._resample_to(mask_oh, feat)
            feat = self.enc_fuse[i](torch.cat([feat, mask_at], dim=1))
            skips.append(feat)

        cls_logits, cls_features = self.classifier_head(skips[-1])
        conditioned_skips = self.film(cls_features, skips)

        lres = conditioned_skips[-1]
        seg_outputs = []
        deep_sup = self.decoder.deep_supervision
        n_dec = len(self.decoder.stages)
        for s in range(n_dec):
            up = self.decoder.transpconvs[s](lres)
            cat = torch.cat([up, conditioned_skips[-(s + 2)]], dim=1)
            out = self.decoder.stages[s](cat)
            mask_at = self._resample_to(mask_oh, out)
            out = self.dec_fuse[s](torch.cat([out, mask_at], dim=1))
            if deep_sup:
                seg_outputs.append(self.decoder.seg_layers[s](out))
            elif s == (n_dec - 1):
                seg_outputs.append(self.decoder.seg_layers[-1](out))
            lres = out
        seg_outputs = seg_outputs[::-1]
        seg_out = seg_outputs if deep_sup else seg_outputs[0]
        return seg_out, cls_logits

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return (self.encoder.compute_conv_feature_map_size(input_size)
                + self.decoder.compute_conv_feature_map_size(input_size))

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class ResidualEncoderUNet_SPADEDecoder_AttentionClassifier(nn.Module):
    """S9: SPADEDecoder backbone + Cross-Attention classifier。
    encoder 純影像；decoder 每 stage 後 SPADE 用 mask one-hot 產生 (γ, β) 調制 feature。
    classifier 讀 encoder 最深 skip（純影像特徵，不含 mask）。
    """
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 mask_classes: int = 10,
                 spade_hidden_channels: int = 128,
                 classifier_num_classes: int = 5,
                 classifier_query_num: int = 4,
                 classifier_num_heads: int = 4,
                 classifier_dropout: float = 0.0,
                 ):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        features_per_stage = list(features_per_stage)
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        self.mask_classes = mask_classes
        self.image_channels = input_channels - 1
        assert self.image_channels >= 1

        self.encoder = ResidualEncoder(
            self.image_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
            return_skips=True, disable_default_stem=False, stem_channels=stem_channels
        )
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        spatial_dim = convert_conv_op_to_dim(conv_op)
        spade_norm_op = {2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[spatial_dim]
        self.dec_spade = nn.ModuleList([
            SPADEBlock(
                feature_channels=features_per_stage[n_stages - 2 - s],
                mask_classes=mask_classes,
                conv_op=conv_op,
                norm_op=spade_norm_op,
                hidden_channels=spade_hidden_channels,
                use_alpha=False,
                alpha_init=0.0,
            )
            for s in range(n_stages - 1)
        ])

        self.classifier_head = AttentionClassifier(
            features_per_stage=features_per_stage,
            num_classes=classifier_num_classes,
            query_num=classifier_query_num,
            num_heads=classifier_num_heads,
            dropout=classifier_dropout,
        )

    def _onehot_mask(self, mask_raw: torch.Tensor) -> torch.Tensor:
        m = mask_raw.squeeze(1).round().long().clamp(0, self.mask_classes - 1)
        oh = F.one_hot(m, num_classes=self.mask_classes)
        dims = list(range(oh.dim()))
        new_order = [0, dims[-1]] + dims[1:-1]
        return oh.permute(*new_order).contiguous().to(
            mask_raw.dtype if mask_raw.is_floating_point() else torch.float32
        )

    @staticmethod
    def _resample(mask_oh: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if mask_oh.shape[2:] == target.shape[2:]:
            return mask_oh
        return F.interpolate(mask_oh, size=target.shape[2:], mode='nearest')

    def forward(self, x):
        image = x[:, :self.image_channels]
        mask_raw = x[:, self.image_channels:self.image_channels + 1]
        mask_oh = self._onehot_mask(mask_raw)

        skips = self.encoder(image)
        cls_output = self.classifier_head(skips)

        lres = skips[-1]
        seg_outputs = []
        deep_sup = self.decoder.deep_supervision
        n_dec = len(self.decoder.stages)
        for s in range(n_dec):
            up = self.decoder.transpconvs[s](lres)
            cat = torch.cat([up, skips[-(s + 2)]], dim=1)
            out = self.decoder.stages[s](cat)
            m = self._resample(mask_oh, out)
            out = self.dec_spade[s](out, m)
            if deep_sup:
                seg_outputs.append(self.decoder.seg_layers[s](out))
            elif s == (n_dec - 1):
                seg_outputs.append(self.decoder.seg_layers[-1](out))
            lres = out
        seg_outputs = seg_outputs[::-1]
        seg_out = seg_outputs if deep_sup else seg_outputs[0]
        return seg_out, cls_output

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return (self.encoder.compute_conv_feature_map_size(input_size)
                + self.decoder.compute_conv_feature_map_size(input_size))

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class ResidualEncoderUNet_SPADEDecoder_GuidedClassifier(nn.Module):
    """S10: SPADEDecoder backbone + Guided (Cross-Attention + FiLM) classifier。
    分類特徵先 FiLM 調制 skip，再進 decoder 做 SPADE 調制。
    """
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 mask_classes: int = 10,
                 spade_hidden_channels: int = 128,
                 classifier_num_classes: int = 5,
                 classifier_query_num: int = 4,
                 classifier_num_heads: int = 4,
                 classifier_dropout: float = 0.0,
                 ):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        features_per_stage = list(features_per_stage)
        self.features_per_stage = features_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        self.mask_classes = mask_classes
        self.image_channels = input_channels - 1
        assert self.image_channels >= 1

        self.encoder = ResidualEncoder(
            self.image_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
            return_skips=True, disable_default_stem=False, stem_channels=stem_channels
        )
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        spatial_dim = convert_conv_op_to_dim(conv_op)
        spade_norm_op = {2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[spatial_dim]
        self.dec_spade = nn.ModuleList([
            SPADEBlock(
                feature_channels=features_per_stage[n_stages - 2 - s],
                mask_classes=mask_classes,
                conv_op=conv_op,
                norm_op=spade_norm_op,
                hidden_channels=spade_hidden_channels,
                use_alpha=False,
                alpha_init=0.0,
            )
            for s in range(n_stages - 1)
        ])

        cls_feature_dim = classifier_query_num * features_per_stage[-1]
        self.classifier_head = CrossAttentionPoolingWithFeatures3D(
            embed_dim=features_per_stage[-1],
            query_num=classifier_query_num,
            num_classes=classifier_num_classes,
            num_heads=classifier_num_heads,
            dropout=classifier_dropout,
        )
        self.film = FiLMConditioner(
            cls_feature_dim=cls_feature_dim,
            features_per_stage=features_per_stage,
        )

    def _onehot_mask(self, mask_raw: torch.Tensor) -> torch.Tensor:
        m = mask_raw.squeeze(1).round().long().clamp(0, self.mask_classes - 1)
        oh = F.one_hot(m, num_classes=self.mask_classes)
        dims = list(range(oh.dim()))
        new_order = [0, dims[-1]] + dims[1:-1]
        return oh.permute(*new_order).contiguous().to(
            mask_raw.dtype if mask_raw.is_floating_point() else torch.float32
        )

    @staticmethod
    def _resample(mask_oh: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if mask_oh.shape[2:] == target.shape[2:]:
            return mask_oh
        return F.interpolate(mask_oh, size=target.shape[2:], mode='nearest')

    def forward(self, x):
        image = x[:, :self.image_channels]
        mask_raw = x[:, self.image_channels:self.image_channels + 1]
        mask_oh = self._onehot_mask(mask_raw)

        skips = self.encoder(image)
        cls_logits, cls_features = self.classifier_head(skips[-1])
        conditioned_skips = self.film(cls_features, skips)

        lres = conditioned_skips[-1]
        seg_outputs = []
        deep_sup = self.decoder.deep_supervision
        n_dec = len(self.decoder.stages)
        for s in range(n_dec):
            up = self.decoder.transpconvs[s](lres)
            cat = torch.cat([up, conditioned_skips[-(s + 2)]], dim=1)
            out = self.decoder.stages[s](cat)
            m = self._resample(mask_oh, out)
            out = self.dec_spade[s](out, m)
            if deep_sup:
                seg_outputs.append(self.decoder.seg_layers[s](out))
            elif s == (n_dec - 1):
                seg_outputs.append(self.decoder.seg_layers[-1](out))
            lres = out
        seg_outputs = seg_outputs[::-1]
        seg_out = seg_outputs if deep_sup else seg_outputs[0]
        return seg_out, cls_logits

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return (self.encoder.compute_conv_feature_map_size(input_size)
                + self.decoder.compute_conv_feature_map_size(input_size))

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


if __name__ == '__main__':
    data = torch.rand((1, 4, 128, 128, 128))

    model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4,
                                (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True)

    if False:
        import hiddenlayer as hl

        g = hl.build_graph(model, data,
                           transforms=None)
        g.save("network_architecture.pdf")
        del g

    print(model.compute_conv_feature_map_size(data.shape[2:]))

    data = torch.rand((1, 4, 512, 512))

    model = PlainConvUNet(4, 8, (32, 64, 125, 256, 512, 512, 512, 512), nn.Conv2d, 3, (1, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2, 2), 4,
                                (2, 2, 2, 2, 2, 2, 2), False, nn.BatchNorm2d, None, None, None, nn.ReLU, deep_supervision=True)

    if False:
        import hiddenlayer as hl

        g = hl.build_graph(model, data,
                           transforms=None)
        g.save("network_architecture.pdf")
        del g

    print(model.compute_conv_feature_map_size(data.shape[2:]))


# ============================================================
# AMAP-style domain prompting —— case/patch 特徵條件化
# 設計依據：docs/mra_aneurysm_sota_methods.md §E.8–E.13
#           docs/plan_feature_conditioning_v4.md
# ============================================================
#
# ⚠️ 與設計文件的兩處刻意偏離，都是為了「step 0 逐位元等於 baseline」這個性質：
#
# 1. cross-attention 的 η 預設 **0.0**（文件依 AMAP 原文寫 0.1）。
#    η=0 時整個 block 退化為 identity；η 自己的梯度不為零（dL/dη = <attn_out, dL/dh>），
#    一步之後就會離開 0，attention 內部參數隨即開始學。這是 ReZero / AdaLN-Zero 的標準做法。
#    另外 FFN 的最後一層也 zero-init，否則 η=0 仍會被 FFN 改動 feature。
#
# 2. FiLM 寫成 **殘差式** `feat + γ·norm(feat) + β`（文件寫 `norm(feat)*(1+γ)+β`）。
#    後者在 γ=β=0 時等於 `InstanceNorm(feat)`，**不是 identity** —— 原本的 decoder
#    每個 stage 後並沒有這個 norm，會平白改變輸出。殘差式在 γ=β=0 時才真的是 identity。
#
# 這兩點讓 s1_prompt 在 epoch 0 的驗證指標必須等於 Dataset136 fold_0，可作為載入正確性的檢查點。


class CasePromptEncoder(nn.Module):
    """case-level 11 維 → K 個 prompt token。"""

    def __init__(self, in_dim: int = 11, out_tokens: int = 5, dim: int = 128,
                 input_norm: bool = False):
        super().__init__()
        self.out_tokens, self.dim = out_tokens, dim
        # input_norm 只給 patch 分支用，case 11 維已在外部套 TRAIN 的 mu/sigma
        # （那是 README 的部署契約：新醫院一律沿用訓練集參數，不得重新擬合）。
        self.in_norm = nn.BatchNorm1d(in_dim) if input_norm else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, dim), nn.GELU(),
            nn.Linear(dim, out_tokens * dim),
        )

    def forward(self, v):                       # v: [B, in_dim]
        return self.mlp(self.in_norm(v)).view(v.shape[0], self.out_tokens, self.dim)


class PatchPromptEncoder(CasePromptEncoder):
    """patch-level 13 維 → K 個 prompt token。

    輸入端有 BatchNorm1d：patch 特徵是動態算的原始值，各維尺度差到 64 倍
    （實測 patch_coord std 0.44、neighbor_MCA/ACA/VB 只有 0.008~0.012），
    而第一層是裸 Linear，各維貢獻與梯度都正比於其 std —— 小尺度那幾維會慢
    40~55 倍才學得動。用 BN 而非外部 normstats CSV，是因為統計量存在
    checkpoint 裡跟著權重走，訓練與推論結構上不可能不同步。
    """

    def __init__(self, in_dim: int = 13, out_tokens: int = 3, dim: int = 128):
        super().__init__(in_dim=in_dim, out_tokens=out_tokens, dim=dim, input_norm=True)


class PromptCrossAttentionBottleneck(nn.Module):
    """Bottleneck 的一層 cross-attention：feature token 對 [feature ‖ prompt] 做 attend。

    η 與 FFN 末層 zero-init → 初始為 identity（見本節開頭說明）。
    """

    def __init__(self, feat_channels: int, prompt_dim: int = 128,
                 num_heads: int = 8, eta_init: float = 0.0):
        super().__init__()
        self.prompt_proj = (nn.Linear(prompt_dim, feat_channels)
                            if prompt_dim != feat_channels else nn.Identity())
        self.norm_q = nn.LayerNorm(feat_channels)
        self.norm_kv = nn.LayerNorm(feat_channels)
        heads = num_heads if feat_channels % num_heads == 0 else 1
        self.attn = nn.MultiheadAttention(feat_channels, heads, batch_first=True)
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        self.ffn = nn.Sequential(
            nn.Linear(feat_channels, feat_channels * 2), nn.GELU(),
            nn.Linear(feat_channels * 2, feat_channels),
        )
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, feat, prompts):
        # feat: [B, C, *spatial]；prompts: [B, K, prompt_dim]
        b, c = feat.shape[:2]
        spatial = feat.shape[2:]
        tok = feat.flatten(2).transpose(1, 2)                    # [B, N, C]
        kv = torch.cat([tok, self.prompt_proj(prompts)], dim=1)  # [B, N+K, C]
        out, _ = self.attn(self.norm_q(tok), self.norm_kv(kv), self.norm_kv(kv))
        h = tok + self.eta * out
        h = h + self.ffn(h)
        return h.transpose(1, 2).view(b, c, *spatial)


class FiLMFromPrompt(nn.Module):
    """global prompt → per-channel (γ, β)，殘差式套用，zero-init 時為 identity。"""

    def __init__(self, prompt_dim: int, feat_channels: int, conv_dim: int = 3):
        super().__init__()
        self.conv_dim = conv_dim
        self.norm = {2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[conv_dim](
            feat_channels, affine=False)
        self.to_film = nn.Linear(prompt_dim, 2 * feat_channels)
        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_film.bias)

    def forward(self, feat, prompts):
        g, b = self.to_film(prompts.mean(dim=1)).chunk(2, dim=-1)
        shape = (-1, feat.shape[1]) + (1,) * self.conv_dim
        return feat + g.view(*shape) * self.norm(feat) + b.view(*shape)


class _PromptMixin(nn.Module):
    """共用的 prompt 產生邏輯：shared tokens + case tokens + patch tokens。"""

    def _build_prompt_modules(self, prompt_dim, n_shared, case_dim, patch_dim,
                              n_case_tokens=5, n_patch_tokens=3, use_patch=True):
        self.prompt_dim = prompt_dim
        self.use_patch = use_patch
        self.shared_prompts = nn.Parameter(torch.randn(n_shared, prompt_dim) * 0.02)
        self.case_mlp = CasePromptEncoder(case_dim, n_case_tokens, prompt_dim)
        self.patch_mlp = (PatchPromptEncoder(patch_dim, n_patch_tokens, prompt_dim)
                          if use_patch else None)

    def _make_prompts(self, b, case_vec, patch_vec, device, dtype):
        toks = [self.shared_prompts.unsqueeze(0).expand(b, -1, -1).to(dtype)]
        if case_vec is None:
            case_vec = torch.zeros(b, self.case_mlp.mlp[0].in_features,
                                   device=device, dtype=dtype)
        toks.append(self.case_mlp(case_vec.to(dtype)))
        if self.use_patch:
            if patch_vec is None:
                patch_vec = torch.zeros(b, self.patch_mlp.mlp[0].in_features,
                                        device=device, dtype=dtype)
            toks.append(self.patch_mlp(patch_vec.to(dtype)))
        return torch.cat(toks, dim=1)


class ResidualEncoderUNet_Prompt(ResidualEncoderUNet, _PromptMixin):
    """S1 + prompt：encoder/decoder 結構與 ResidualEncoderUNet 完全相同（1ch 輸入），
    只在 bottleneck 加 cross-attention、每個 decoder stage 後加 FiLM。

    因為結構同構且 prompt 模組 zero-init，載入 Dataset136 fold_0 後
    **step 0 的輸出與 baseline 逐位元相同**。
    """

    def __init__(self, *args, prompt_dim: int = 128, n_shared_tokens: int = 8,
                 case_feature_dim: int = 11, patch_feature_dim: int = 13,
                 use_patch_prompt: bool = True, eta_init: float = 0.0,
                 network_in_channels: int = None, **kwargs):
        # network_in_channels：覆寫 plans 的輸入通道數。
        # s1 用 2ch 資料集（ch1 供 dataloader 算 patch 特徵），但網路只吃 ch0（MRA），
        # 必須維持 1ch 才能載入 Dataset136 的 1ch stem —— 那是「step 0 等於 baseline」的前提。
        if network_in_channels is not None:
            if args:
                args = (network_in_channels,) + tuple(args[1:])
            else:
                kwargs["input_channels"] = network_in_channels
        super().__init__(*args, **kwargs)
        self._build_prompt_modules(prompt_dim, n_shared_tokens,
                                   case_feature_dim, patch_feature_dim,
                                   use_patch=use_patch_prompt)
        feats = [s.output_channels for s in self.encoder.stages]
        conv_dim = convert_conv_op_to_dim(self.encoder.conv_op)
        self.bottleneck_attn = PromptCrossAttentionBottleneck(
            feats[-1], prompt_dim, eta_init=eta_init)
        n_dec = len(self.decoder.stages)
        self.dec_film = nn.ModuleList([
            FiLMFromPrompt(prompt_dim, feats[len(feats) - 2 - s], conv_dim)
            for s in range(n_dec)
        ])

    def forward(self, x, case_vec=None, patch_vec=None):
        prompts = self._make_prompts(x.shape[0], case_vec, patch_vec, x.device, x.dtype)
        skips = self.encoder(x)
        skips = list(skips)
        skips[-1] = self.bottleneck_attn(skips[-1], prompts)

        lres, seg_outputs = skips[-1], []
        deep_sup = self.decoder.deep_supervision
        n_dec = len(self.decoder.stages)
        for s in range(n_dec):
            up = self.decoder.transpconvs[s](lres)
            out = self.decoder.stages[s](torch.cat([up, skips[-(s + 2)]], dim=1))
            out = self.dec_film[s](out, prompts)
            if deep_sup:
                seg_outputs.append(self.decoder.seg_layers[s](out))
            elif s == (n_dec - 1):
                seg_outputs.append(self.decoder.seg_layers[-1](out))
            lres = out
        seg_outputs = seg_outputs[::-1]
        return seg_outputs if deep_sup else seg_outputs[0]


# ------------------------------------------------------------
# S2 / S3 的 prompt 版本
#
# ⚠️ 第三處刻意偏離設計文件：文件的 `SPADEWithPrompt` 把 prompt bias 併進 SPADE 的 γ/β
#    一起算；這裡改成**串接** —— 先跑既有的 SPADE / DeepConcat fuse，再套一層殘差式 FiLM。
#    兩者在數學上都是「spatial 調制 + global per-channel affine」，但串接版：
#      (1) 完全不動已訓練好的 SPADEBlock / fuse conv，載入既有權重時零風險
#      (2) FiLM zero-init → **加上 prompt 後與原架構逐位元相同**，可驗證
#    代價是多一次 InstanceNorm 的計算，可忽略。
#
# ⚠️ 注意：s2/s3 從 Dataset136 fold_0 載入時，enc_fuse / dec_fuse / dec_spade 是全新模組，
#    必然隨機初始化 —— 所以 s2/s3 **沒有** s1 那種「step 0 等於 Dataset136」的性質。
#    它們的恆等性是相對於「同權重的非 prompt 版本」而言。
# ------------------------------------------------------------


def _attach_prompt_blocks(self, prompt_dim, n_shared, case_dim, patch_dim,
                          use_patch, eta_init, feats, conv_dim, n_dec):
    """s2/s3 共用：建 prompt encoder + bottleneck attention + 每 decoder stage 的 FiLM。"""
    self._build_prompt_modules(prompt_dim, n_shared, case_dim, patch_dim,
                               use_patch=use_patch)
    self.bottleneck_attn = PromptCrossAttentionBottleneck(
        feats[-1], prompt_dim, eta_init=eta_init)
    self.dec_film = nn.ModuleList([
        FiLMFromPrompt(prompt_dim, feats[len(feats) - 2 - s], conv_dim)
        for s in range(n_dec)
    ])


class ResidualEncoderUNet_DeepConcatPrompt(ResidualEncoderUNet_DeepConcat, _PromptMixin):
    """S2 + prompt：vessel4 走 input concat + 每 stage fuse（原樣），另加 prompt 條件化。"""

    def __init__(self, *args, prompt_dim: int = 128, n_shared_tokens: int = 8,
                 case_feature_dim: int = 11, patch_feature_dim: int = 13,
                 use_patch_prompt: bool = True, eta_init: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        feats = [s.output_channels for s in self.encoder.stages]
        _attach_prompt_blocks(self, prompt_dim, n_shared_tokens, case_feature_dim,
                              patch_feature_dim, use_patch_prompt, eta_init, feats,
                              convert_conv_op_to_dim(self.encoder.conv_op),
                              len(self.decoder.stages))

    def forward(self, x, case_vec=None, patch_vec=None):
        prompts = self._make_prompts(x.shape[0], case_vec, patch_vec, x.device, x.dtype)
        image = x[:, :self.image_channels]
        mask_oh = self._onehot_mask(x[:, self.image_channels:self.image_channels + 1])

        feat = image
        if self.encoder.stem is not None:
            feat = self.encoder.stem(feat)
        skips = []
        for i, stage in enumerate(self.encoder.stages):
            feat = stage(feat)
            feat = self.enc_fuse[i](torch.cat([feat, self._resample_to(mask_oh, feat)], dim=1))
            skips.append(feat)
        skips[-1] = self.bottleneck_attn(skips[-1], prompts)

        lres, seg_outputs = skips[-1], []
        deep_sup = self.decoder.deep_supervision
        n_dec = len(self.decoder.stages)
        for s in range(n_dec):
            up = self.decoder.transpconvs[s](lres)
            out = self.decoder.stages[s](torch.cat([up, skips[-(s + 2)]], dim=1))
            out = self.dec_fuse[s](torch.cat([out, self._resample_to(mask_oh, out)], dim=1))
            out = self.dec_film[s](out, prompts)
            if deep_sup:
                seg_outputs.append(self.decoder.seg_layers[s](out))
            elif s == (n_dec - 1):
                seg_outputs.append(self.decoder.seg_layers[-1](out))
            lres = out
        seg_outputs = seg_outputs[::-1]
        return seg_outputs if deep_sup else seg_outputs[0]


class ResidualEncoderUNet_SPADEDecoderPrompt(ResidualEncoderUNet_SPADE, _PromptMixin):
    """S3 + prompt：decoder 每 stage 先 vanilla SPADE（vessel4 空間條件），再套 prompt FiLM。"""

    def __init__(self, *args, prompt_dim: int = 128, n_shared_tokens: int = 8,
                 case_feature_dim: int = 11, patch_feature_dim: int = 13,
                 use_patch_prompt: bool = True, eta_init: float = 0.0, **kwargs):
        kwargs.setdefault('inject_in_decoder', True)
        kwargs.setdefault('inject_in_encoder', False)
        kwargs.setdefault('use_alpha', False)
        super().__init__(*args, **kwargs)
        feats = [s.output_channels for s in self.encoder.stages]
        _attach_prompt_blocks(self, prompt_dim, n_shared_tokens, case_feature_dim,
                              patch_feature_dim, use_patch_prompt, eta_init, feats,
                              convert_conv_op_to_dim(self.encoder.conv_op),
                              len(self.decoder.stages))

    def forward(self, x, case_vec=None, patch_vec=None):
        prompts = self._make_prompts(x.shape[0], case_vec, patch_vec, x.device, x.dtype)
        image = x[:, :self.image_channels]
        mask_oh = self._onehot_mask(x[:, self.image_channels:self.image_channels + 1])

        feat = image
        if self.encoder.stem is not None:
            feat = self.encoder.stem(feat)
        skips = []
        for i, stage in enumerate(self.encoder.stages):
            feat = stage(feat)
            if self.inject_in_encoder:
                feat = self.enc_spade[i](feat, self._resample(mask_oh, feat))
            skips.append(feat)
        skips[-1] = self.bottleneck_attn(skips[-1], prompts)

        lres, seg_outputs = skips[-1], []
        deep_sup = self.decoder.deep_supervision
        n_dec = len(self.decoder.stages)
        for s in range(n_dec):
            up = self.decoder.transpconvs[s](lres)
            out = self.decoder.stages[s](torch.cat([up, skips[-(s + 2)]], dim=1))
            if self.inject_in_decoder:
                out = self.dec_spade[s](out, self._resample(mask_oh, out))
            out = self.dec_film[s](out, prompts)
            if deep_sup:
                seg_outputs.append(self.decoder.seg_layers[s](out))
            elif s == (n_dec - 1):
                seg_outputs.append(self.decoder.seg_layers[-1](out))
            lres = out
        seg_outputs = seg_outputs[::-1]
        return seg_outputs if deep_sup else seg_outputs[0]

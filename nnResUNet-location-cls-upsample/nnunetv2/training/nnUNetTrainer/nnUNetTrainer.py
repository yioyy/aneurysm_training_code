import inspect
import multiprocessing
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List
import mlflow
from torchinfo import summary

import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose


class ProtectedChannelsWrapper(AbstractTransform):
    """把一個 transform 包起來，套用前 snapshot 受保護的 channel，套完 restore。

    用途：SynthSeg label channel、ADC 物理量 channel 等，不該被亮度/對比/雜訊/gamma 擾動的 channel。
    空間變換（rotation/scaling/mirror）跟整批的 spatial geometry 對齊，不該受此保護。

    Args:
        inner: 內層 transform（會被套到 data dict 全部 channel）
        protected_channels: 要保護的 channel index list（in data shape [B, C, ...]）
        data_key: data dict 裡 image batch 的 key（預設 'data'）
    """

    def __init__(self, inner: AbstractTransform, protected_channels, data_key: str = 'data'):
        self.inner = inner
        self.protected_channels = sorted(set(int(c) for c in protected_channels))
        self.data_key = data_key

    def __call__(self, **data_dict):
        if not self.protected_channels:
            return self.inner(**data_dict)
        data = data_dict.get(self.data_key)
        if data is None:
            return self.inner(**data_dict)
        # data shape: [B, C, ...]; snapshot 受保護 channel
        snapshot = data[:, self.protected_channels].copy()
        data_dict = self.inner(**data_dict)
        out = data_dict.get(self.data_key)
        # transform 可能 in-place 或回傳新 array；都要 restore
        out[:, self.protected_channels] = snapshot
        data_dict[self.data_key] = out
        return data_dict

    def __repr__(self):
        return f"ProtectedChannelsWrapper(protected={self.protected_channels}, inner={self.inner.__class__.__name__})"


class LabelPreservingSpatialTransform(AbstractTransform):
    """SpatialTransform 包裝：對指定 label channel 用 nearest interp（避免 cubic overshoot 破壞 categorical label）。

    Cubic spline 內插（SpatialTransform 預設 order_data=3）對連續值影像沒問題，
    但對 categorical anatomy label（如 SynthSeg）會在邊界產生 OVERSHOOT：
        label 7 voxel 旁邊是 label 0 → cubic 內插出 8.5 → clamp 變 label 9（小腦）→ 完全錯位

    解法：把 label channel 暫時從 'data' 移到 'seg' key，SpatialTransform 對 seg 用 order_seg
    （我們設 0 = nearest），然後把結果搬回 data。SAME 旋轉/縮放/形變參數 → image 跟 label 空間對齊。

    Args:
        spatial_transform: 真實的 SpatialTransform 實體（必須已配置好參數）
        label_channels: data 裡的哪些 channel 是 label channel（要走 nearest interp）
    """

    def __init__(self, spatial_transform, label_channels, data_key: str = 'data', seg_key: str = 'seg'):
        self.spatial_transform = spatial_transform
        self.label_channels = sorted(set(int(c) for c in label_channels))
        self.data_key = data_key
        self.seg_key = seg_key

    def __call__(self, **data_dict):
        if not self.label_channels:
            return self.spatial_transform(**data_dict)
        data = data_dict.get(self.data_key)
        if data is None:
            return self.spatial_transform(**data_dict)

        # 1. 切出 image / label channels
        n_ch = data.shape[1]
        image_ch_idx = [c for c in range(n_ch) if c not in self.label_channels]
        # 保持 channel 順序記憶（restore 用）
        image_data = data[:, image_ch_idx]
        label_data = data[:, self.label_channels]

        # 2. 把 label 暫時併到 seg 裡（seg 在 SpatialTransform 用 order_seg 處理，預設 1=linear；
        # 我們透過 setattr 改成 0=nearest 才能保 integer label）
        original_seg = data_dict.get(self.seg_key, None)
        if original_seg is not None:
            combined_seg = np.concatenate([original_seg, label_data], axis=1)
            n_orig_seg = original_seg.shape[1]
        else:
            combined_seg = label_data
            n_orig_seg = 0

        new_dict = dict(data_dict)
        new_dict[self.data_key] = image_data
        new_dict[self.seg_key] = combined_seg

        # 3. 強制 order_seg=0 (nearest) + border_cval_seg=0（旋轉外圍當「腦外背景 label 0」）
        #    記原值、套完還原
        orig_order_seg = getattr(self.spatial_transform, 'order_seg', 1)
        orig_border_cval_seg = getattr(self.spatial_transform, 'border_cval_seg', -1)
        try:
            self.spatial_transform.order_seg = 0
            self.spatial_transform.border_cval_seg = 0  # background_outside_brain
            out = self.spatial_transform(**new_dict)
        finally:
            self.spatial_transform.order_seg = orig_order_seg
            self.spatial_transform.border_cval_seg = orig_border_cval_seg

        # 4. 重組
        new_image = out[self.data_key]
        new_combined_seg = out[self.seg_key]
        if n_orig_seg > 0:
            new_seg = new_combined_seg[:, :n_orig_seg]
            new_label = new_combined_seg[:, n_orig_seg:]
        else:
            new_seg = None
            new_label = new_combined_seg

        # 5. 按原 channel 順序 reconstruct data
        out_data = np.empty(
            (new_image.shape[0], n_ch) + new_image.shape[2:],
            dtype=new_image.dtype,
        )
        out_data[:, image_ch_idx] = new_image
        out_data[:, self.label_channels] = new_label

        result = dict(out)
        result[self.data_key] = out_data
        if new_seg is not None:
            result[self.seg_key] = new_seg
        elif self.seg_key in result:
            # 沒有 original seg，但 SpatialTransform 把 label 拿去當 seg，不該留在 seg 裡
            del result[self.seg_key]
        return result

    def __repr__(self):
        return (f"LabelPreservingSpatialTransform(label_channels={self.label_channels}, "
                f"inner={self.spatial_transform.__class__.__name__})")
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_softmax, resample_and_save
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, predict_sliding_window_return_logits
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset, build_sampling_probabilities
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss, Log_DC_loss, CE_loss, DC_loss, Tversky_and_CE_loss, Compound_loss
from nnunetv2.training.loss.boundary_loss import BoundaryLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss, MemoryEfficientLogDiceLoss, MemoryEfficientNewSoftDiceLoss, NewSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import should_i_save_to_file, check_workers_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from sklearn.model_selection import KFold
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class nnUNetTrainer(object):
    # 依類別加權抽樣：類別 1~4 的權重，比例 2:1:1:1。可於子類覆寫以改變比例。
    SAMPLING_CATEGORY_WEIGHTS = {1: 2, 2: 1, 3: 1, 4: 1}
    # 抽樣權重解讀模式：
    # - "multiplier": 權重直接套用到每個 case（類別內所有 case 權重相同）
    # - "target_proportion": 權重解讀為目標類別抽樣比例（會自動除以該 fold 類別 case 數量）
    SAMPLING_CATEGORY_WEIGHT_MODE = "target_proportion"

    # normal upsample 模式：normal patch 按類別加權取樣
    # None → 非 upsample 模式，所有座標合併後隨機取樣（傳統行為）
    # dict → upsample 模式，e.g. {1:1, 2:1, 3:1, 4:1} 表示 4 類各 25% 機率
    NORMAL_CLASS_WEIGHTS = None

    # 顯式開關（由 CLI --enable_sampling_weights / --enable_normal_upsample 控制）
    # False → 即使 sampling_categories 存在也不套用加權取樣
    # True  → 套用 SAMPLING_CATEGORY_WEIGHTS 加權取樣
    ENABLE_SAMPLING_WEIGHTS = False
    # False → 所有座標合併隨機取樣（傳統行為）
    # True  → 按 NORMAL_CLASS_WEIGHTS 比例取樣
    ENABLE_NORMAL_UPSAMPLE = False

    # 是否記錄每層 deep supervision 的 dice（ce_l, dice_l, individual_dice_losses, dc1/dc2/...）
    # False → 只計算 loss + dc0（最高解析度），省掉 ~2x loss 計算 + N 層 dice 計算
    # True  → 完整記錄每層 dice（用於 debug 各層表現）
    ENABLE_DEEP_SUPERVISION_LOGGING = False

    # EMA (Exponential Moving Average) model
    # False → 不使用 EMA
    # True  → 維護一份 EMA 參數副本，驗證時額外跑 EMA model 並記錄/存檔
    ENABLE_EMA = False
    EMA_DECAY = 0.999

    # 當由 MUTP 呼叫時，關閉 Trainer 內建的 MLflow（由 MUTP 的 watcher thread 獨立追蹤）
    # False → Trainer 自行管理 MLflow run（獨立使用時）
    # True  → 跳過所有 mlflow.start_run / log_metric / log_artifact / autolog
    DISABLE_BUILTIN_MLFLOW = False

    # 分類頭判斷 positive 時只看哪些 seg label
    # None → 所有 > 0 的 label 都算前景（原始行為，適用 2 分類）
    # [1] → 只有 label==1 算前景（multi-label 時只看動脈瘤）
    # [1, 2] → label==1 或 2 算前景
    CLS_FOREGROUND_LABELS = None

    # Best checkpoint 根據哪幾個 class 的 dice 來判斷
    # None → 所有 foreground class 的平均（原始行為）
    # [4] → 只看第 5 個 class（0-indexed，例如 Aneurysm）
    # [0, 4] → 只看第 1 和第 5 個 class 的平均
    BEST_VAL_CLASSES = None

    # 資料增強設定（由 MUTP recipe 控制，None=使用 nnU-Net 預設值）
    AUGMENTATION_CONFIG = None

    # 模型額外建構參數（由 MUTP recipe.model.extra 透過 run_training 傳入）
    # 會被 merge 進 get_network_from_plans 的 kwargs[UNet_class_name]
    # 例：{"spade_alpha_init": 0.3}
    MODEL_EXTRA_KWARGS = None

    # 梯度累積（batch_size 大時降低 data loader 壓力）
    # False ��� 每步都 optimizer.step()（預設行為）
    # True → 累積 GRADIENT_ACCUMULATION_STEPS 步才 optimizer.step()
    ENABLE_GRADIENT_ACCUMULATION = False
    GRADIENT_ACCUMULATION_STEPS = 1

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'),
                 initial_lr: float = 1e-4,
                 oversample_foreground_percent: float = 0.5,
                 oversample_foreground_percent_val: float = 0.2,
                 num_iterations_per_epoch: int = 500,
                 num_epochs: int = 1000,
                 optimizer_type: str = 'AdamW',
                 lr_scheduler_type: str = 'CosineAnnealingLR',
                 enable_early_stopping: bool = False,
                 early_stopping_patience: int = 50,
                 early_stopping_min_delta: float = 0.0001):
        # From https://grugbrain.dev/. Worth a read ya big brains ;-)

        # apex predator of grug is complexity
        # complexity bad
        # say again:
        # complexity very bad
        # you say now:
        # complexity very, very bad
        # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
        # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
        # one day code base understandable and grug can get work done, everything good!
        # next day impossible: complexity demon spirit has entered code and very dangerous situation!

        # OK OK I am guilty. But I tried. http://tiny.cc/gzgwuz

        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()

        self.device = device

        # print what device we are using
        if self.is_ddp:  # implicitly it's clear that we use cuda in this case
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
                  f"{dist.get_world_size()}."
                  f"Setting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
                self.device = torch.device(type='cuda', index=0)
            print(f"Using device: {self.device}")

        # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
        # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
        # need. So let's save the init args
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]

        ###  Saving all the init args into class variables for later access
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.configuration_name = configuration
        self.dataset_json = dataset_json
        self.fold = fold
        self.unpack_dataset = unpack_dataset

        ### Setting all the folder names. We need to make sure things don't crash in case we are just running
        # inference and some of the folders may not be defined!
        self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
            if nnUNet_preprocessed is not None else None
        self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                       self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
            if nnUNet_results is not None else None
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.configuration_manager.data_identifier)
        # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
        # be a different configuration in the same plans
        # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
        # "previous_stage" and "next_stage"). Otherwise it won't work!
        self.is_cascaded = self.configuration_manager.previous_stage_name is not None
        self.folder_with_segs_from_previous_stage = \
            join(nnUNet_results, self.plans_manager.dataset_name,
                 self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
                 self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
                if self.is_cascaded else None
        # Dual-head multi-task: 餵 aux seg 進 seg_from_prev_stage 機制 → nnUNetDataset 自動疊成 2nd channel
        # 由 mutp 在啟動訓練前設定：
        #   MUTP_AUX_SEG_INLINE=true  → 新版扁平佈局：{case}_aux_seg.npz 跟主檔同層（dataset 端用 sentinel 處理）
        #   MUTP_AUX_SEG_FOLDER=<dir> → 舊版獨立資料夾：<dir>/{case}.npz
        if not self.is_cascaded:
            _aux_inline = os.environ.get("MUTP_AUX_SEG_INLINE", "").lower() in ("1", "true", "yes")
            _aux_folder = os.environ.get("MUTP_AUX_SEG_FOLDER")
            if _aux_inline:
                self.folder_with_segs_from_previous_stage = "MUTP_INLINE_AUX_SEG"
                print(f"[MUTP] dual-head aux seg: inline ({{case}}_aux_seg.npz alongside data)")
            elif _aux_folder:
                self.folder_with_segs_from_previous_stage = _aux_folder
                print(f"[MUTP] dual-head aux seg folder: {_aux_folder}")

        ### Some hyperparameters for you to fiddle with
        self.initial_lr = initial_lr
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = oversample_foreground_percent
        self.oversample_foreground_percent_val = oversample_foreground_percent_val
        self.num_iterations_per_epoch = num_iterations_per_epoch
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = num_epochs
        self.current_epoch = 0
        
        ### Optimizer and learning rate scheduler configuration
        self.optimizer_type = optimizer_type
        self.lr_scheduler_type = lr_scheduler_type
        
        ### Early stopping configuration
        self.enable_early_stopping = enable_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_counter = 0
        self.early_stopping_best_metric = None
        self.should_stop_training = False

        ### Dealing with labels/regions
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
        # needed for predictions. We do sigmoid in case of (overlapping) regions

        self.num_input_channels = None  # -> self.initialize()
        self.network = None  # -> self._get_network()
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        self.loss = None  # -> self.initialize

        ### Simple logging. Don't take that away from me!
        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging
        timestamp = datetime.now()
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        # Initialize logger with dynamic deep supervision levels based on network architecture
        # has_cls_head 在 initialize() 時才確定，初始先用 True（initialize() 會重建 logger）
        initial_num_deep_supervision_levels = len(self._get_deep_supervision_scales())
        self.logger = nnUNetLogger(verbose=False, num_deep_supervision_levels=initial_num_deep_supervision_levels, has_cls_head=True)

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start

        ### initializing stuff for remembering things and such
        self._best_ema = None
        self._best_ema_model_dice = None  # EMA model 的 best val dice
        self.ema_model = None  # EMA 參數副本，在 initialize() 時建立

        ### inference things
        self.inference_allowed_mirroring_axes = None  # this variable is set in
        # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints

        ### checkpoint saving stuff
        self.save_every = 50
        self.disable_checkpointing = False

        ## DDP batch size and oversampling can differ between workers and needs adaptation
        # we need to change the batch size in DDP because we don't use any of those distributed samplers
        self._set_batch_size_and_oversample()

        self.was_initialized = False

        self.print_to_log_file("\n#######################################################################\n"
                               "Please cite the following paper when using nnU-Net:\n"
                               "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
                               "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
                               "Nature methods, 18(2), 203-211.\n"
                               "#######################################################################\n",
                               also_print_to_console=True, add_timestamp=False)

    # 有分類頭的模型名稱集合，用來自動偵測是否啟用分類相關邏輯
    _CLS_NETWORK_NAMES = {
        'ResidualEncoderUNetClassifier', 'ResidualEncoderUNetClassifier2D',
        'ResidualEncoderUNetAttentionClassifier', 'ResidualEncoderUNetAttentionClassifier2D',
        'ResidualEncoderUNetGuidedClassifier', 'ResidualEncoderUNetGuidedClassifier2D',
        # S7-S10 hybrids: mask-fusion backbone + classifier
        'ResidualEncoderUNet_DeepConcat_AttentionClassifier',
        'ResidualEncoderUNet_DeepConcat_GuidedClassifier',
        'ResidualEncoderUNet_SPADEDecoder_AttentionClassifier',
        'ResidualEncoderUNet_SPADEDecoder_GuidedClassifier',
    }

    # 雙 seg head 模型名稱集合 — multi-task: main = infarct, aux = SynthSeg region
    # Forward 回傳 (main_logits, aux_logits) tuple；target 必須是 2-channel seg
    _DUAL_SEG_HEAD_NAMES = {
        'ResidualEncoderUNet_DualSegHead',
    }

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                           self.configuration_manager,
                                                           self.num_input_channels,
                                                           enable_deep_supervision=True).to(self.device)

            # 自動偵測是否有分類頭
            self.has_cls_head = self.configuration_manager.UNet_class_name in self._CLS_NETWORK_NAMES
            self.print_to_log_file(f'has_cls_head: {self.has_cls_head} (UNet_class_name: {self.configuration_manager.UNet_class_name})')

            # 自動偵測是否有雙 seg head (multi-task: main = infarct, aux = SynthSeg region)
            # Dual-head model forward 回傳 (main, aux) tuple；target 必須是 2-channel (infarct + region)
            self.has_aux_seg_head = self.configuration_manager.UNet_class_name in self._DUAL_SEG_HEAD_NAMES
            # aux_loss_weight 優先順序: MUTP_MULTI_HEAD_CONFIG json > class attr AUX_LOSS_WEIGHT > 預設 1.0
            self.aux_loss_weight = getattr(self, "AUX_LOSS_WEIGHT", 1.0)
            self.multi_head_config = None
            _mh_path = os.environ.get("MUTP_MULTI_HEAD_CONFIG")
            if self.has_aux_seg_head and _mh_path and os.path.isfile(_mh_path):
                import json as _json
                with open(_mh_path) as _f:
                    self.multi_head_config = _json.load(_f)
                if self.multi_head_config.get("enabled") and len(self.multi_head_config.get("heads", [])) >= 2:
                    _aux = self.multi_head_config["heads"][1]
                    self.aux_loss_weight = float(_aux.get("loss", {}).get("weight", 1.0))
                    self.print_to_log_file(
                        f'[multi-head] config loaded from {_mh_path}: '
                        f'main={self.multi_head_config["heads"][0]["name"]}/'
                        f'{self.multi_head_config["heads"][0]["num_classes"]}cls, '
                        f'aux={_aux["name"]}/{_aux["num_classes"]}cls, '
                        f'aux_loss_weight={self.aux_loss_weight}'
                    )
            self.print_to_log_file(
                f'has_aux_seg_head: {self.has_aux_seg_head}'
                + (f' (aux_loss_weight={self.aux_loss_weight})' if self.has_aux_seg_head else '')
            )

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            if self.has_cls_head:
                self.cls_loss = self._build_cls_loss()
            if self.has_aux_seg_head:
                # aux head 用同一個 loss class，但 num_classes 不同 → 直接重用 _build_loss 即可
                # (DC_and_CE_loss 是 generic for any num_classes)
                self.aux_loss = self._build_loss()

            self.num_deep_supervision_levels = len(self._get_deep_supervision_scales())
            self.enable_deep_supervision_logging = getattr(self, "ENABLE_DEEP_SUPERVISION_LOGGING", False)
            self.print_to_log_file(
                f"ENABLE_DEEP_SUPERVISION_LOGGING={self.enable_deep_supervision_logging}",
                also_print_to_console=True
            )

            if self.enable_deep_supervision_logging:
                self.ce_loss = self._build_ce_loss()
                self.dice_loss = self._build_dice_loss()
                self.individual_dice_losses = {}
                for i in range(self.num_deep_supervision_levels):
                    dice_loss_name = f'dice_loss{i}'
                    self.individual_dice_losses[dice_loss_name] = self._build_individual_dice_loss(i)
                    setattr(self, dice_loss_name, self.individual_dice_losses[dice_loss_name])
            
            # Reinitialize logger with correct number of deep supervision levels and cls head flag
            self.logger = nnUNetLogger(verbose=self.logger.verbose, num_deep_supervision_levels=self.num_deep_supervision_levels, has_cls_head=self.has_cls_head)
            # 把 loss 描述傳給 logger，給 progress.png suptitle 用
            self.logger.loss_str = getattr(self, 'loss_str', None)

            # 大字 banner 印 loss 組合（log + console），方便回頭翻 log 一眼看出實驗
            if getattr(self, 'loss_str', None):
                _banner = "=" * 70
                self.print_to_log_file(_banner, also_print_to_console=True)
                self.print_to_log_file(f"  Loss config: {self.loss_str}", also_print_to_console=True)
                self.print_to_log_file(_banner, also_print_to_console=True)
            
            # EMA model：建立一份 network 的深拷貝作為 EMA 參數副本
            if self.ENABLE_EMA:
                self.ema_model = deepcopy(self.network.module if self.is_ddp else self.network)
                self.ema_model.requires_grad_(False)  # EMA 不需要梯度
                self.ema_model.eval()
                self.print_to_log_file(
                    f"EMA model initialized (decay={self.EMA_DECAY})",
                    also_print_to_console=True
                )

            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def _save_debug_information(self):
        # saving some debug information
        if self.local_rank == 0:
            dct = {}
            for k in self.__dir__():
                if not k.startswith("__"):
                    if not callable(getattr(self, k)) or k in ['loss', ]:
                        dct[k] = str(getattr(self, k))
                    elif k in ['network', ]:
                        dct[k] = str(getattr(self, k).__class__.__name__)
                    else:
                        # print(k)
                        pass
                if k in ['dataloader_train', 'dataloader_val']:
                    if hasattr(getattr(self, k), 'generator'):
                        dct[k + '.generator'] = str(getattr(self, k).generator)
                    if hasattr(getattr(self, k), 'num_processes'):
                        dct[k + '.num_processes'] = str(getattr(self, k).num_processes)
                    if hasattr(getattr(self, k), 'transform'):
                        dct[k + '.transform'] = str(getattr(self, k).transform)
            import subprocess
            hostname = subprocess.getoutput(['hostname'])
            dct['hostname'] = hostname
            torch_version = torch.__version__
            if self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name()
                dct['gpu_name'] = gpu_name
                cudnn_version = torch.backends.cudnn.version()
            else:
                cudnn_version = 'None'
            dct['device'] = str(self.device)
            dct['torch_version'] = torch_version
            dct['cudnn_version'] = cudnn_version
            save_json(dct, join(self.output_folder, "debug.json"))

    @torch.no_grad()
    def _update_ema(self):
        """更新 EMA model 參數：ema_param = decay * ema_param + (1 - decay) * param"""
        if self.ema_model is None:
            return
        decay = self.EMA_DECAY
        source = self.network.module if self.is_ddp else self.network
        for ema_p, src_p in zip(self.ema_model.parameters(), source.parameters()):
            ema_p.data.mul_(decay).add_(src_p.data, alpha=1.0 - decay)

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        his is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        but don't bug us with your bugs arising from fiddling with this :-P
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)

        """
        # 從 class attr 讀額外建構參數（mutp 透過 MODEL_EXTRA_KWARGS class attr 傳）
        # 例：S4/S5 的 spade_alpha_init=0.3
        extra = getattr(nnUNetTrainer, 'MODEL_EXTRA_KWARGS', None)
        return get_network_from_plans(plans_manager, dataset_json, configuration_manager,
                                      num_input_channels, deep_supervision=enable_deep_supervision,
                                      extra_kwargs=extra)

    def _get_deep_supervision_scales(self):
        deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]
        return deep_supervision_scales

    def _set_batch_size_and_oversample(self):
        if not self.is_ddp:
            # set batch size to what the plan says, leave oversample untouched
            self.batch_size = self.configuration_manager.batch_size
        else:
            # batch size is distributed over DDP workers and we need to change oversample_percent for each worker
            batch_sizes = []
            oversample_percents = []

            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            global_batch_size = self.configuration_manager.batch_size
            assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of ' \
                                                    'GPUs... Duh.'

            batch_size_per_GPU = np.ceil(global_batch_size / world_size).astype(int)

            for rank in range(world_size):
                if (rank + 1) * batch_size_per_GPU > global_batch_size:
                    batch_size = batch_size_per_GPU - ((rank + 1) * batch_size_per_GPU - global_batch_size)
                else:
                    batch_size = batch_size_per_GPU

                batch_sizes.append(batch_size)

                sample_id_low = 0 if len(batch_sizes) == 0 else np.sum(batch_sizes[:-1])
                sample_id_high = np.sum(batch_sizes)

                if sample_id_high / global_batch_size < (1 - self.oversample_foreground_percent):
                    oversample_percents.append(0.0)
                elif sample_id_low / global_batch_size > (1 - self.oversample_foreground_percent):
                    oversample_percents.append(1.0)
                else:
                    percent_covered_by_this_rank = sample_id_high / global_batch_size - sample_id_low / global_batch_size
                    oversample_percent_here = 1 - (((1 - self.oversample_foreground_percent) -
                                                    sample_id_low / global_batch_size) / percent_covered_by_this_rank)
                    oversample_percents.append(oversample_percent_here)

            print("worker", my_rank, "oversample", oversample_percents[my_rank])
            print("worker", my_rank, "batch_size", batch_sizes[my_rank])
            # self.print_to_log_file("worker", my_rank, "oversample", oversample_percents[my_rank])
            # self.print_to_log_file("worker", my_rank, "batch_size", batch_sizes[my_rank])

            self.batch_size = batch_sizes[my_rank]
            self.oversample_foreground_percent = oversample_percents[my_rank]

    def _build_cls_loss(self):
        """Classification head loss: 5-class CE (0=no aneurysm, 1-4=location)"""
        return nn.CrossEntropyLoss()

    def _build_loss(self):
        if self.label_manager.has_regions:
            # 從 dataset.json 讀取可選的 per-channel loss 權重
            channel_weights = self.dataset_json.get('region_loss_weights', None)
            if channel_weights is not None:
                self.print_to_log_file(
                    f"enable_region_loss_weights=True → per-channel loss 權重: {channel_weights}",
                    also_print_to_console=True
                )
            else:
                self.print_to_log_file(
                    "enable_region_loss_weights=False → 所有 channel 等權重",
                    also_print_to_console=True
                )

            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientNewSoftDiceLoss,
                                   channel_weights=channel_weights)
            self.loss_str = "DC+BCE"
            print('Loss: DC_and_BCE_loss')
        else:
            # 從 recipe 讀 loss 設定（mutp 透過 LOSS_CONFIG class attr 傳）
            # 兩種格式都接受：
            #   舊：{name: 'Tversky_and_CE_loss', params: {...}}                 (單 loss)
            #   新：{components: [{name, weight, params}, {name, weight, params}, ...]}  (複合)
            loss_cfg = getattr(self, 'LOSS_CONFIG', None) or {}
            components = loss_cfg.get('components', None)
            if components is None:
                # 舊格式 → 包成 1-element components
                single_name = loss_cfg.get('name', 'DC_and_CE_loss')
                single_params = loss_cfg.get('params', {}) or {}
                components = [{'name': single_name, 'weight': 1.0, 'params': single_params}]

            built = []
            built_weights = []
            built_names = []
            short_names = []   # 給 log / CSV / progress.png 用的緊湊描述
            for comp in components:
                cname = comp.get('name', 'DC_and_CE_loss')
                cweight = float(comp.get('weight', 1.0))
                cparams = comp.get('params', {}) or {}
                if cname == 'Tversky_and_CE_loss':
                    alpha = float(cparams.get('alpha', 0.3))
                    beta = float(cparams.get('beta', 0.7))
                    sub_loss = Tversky_and_CE_loss(
                        {'batch_dice': self.configuration_manager.batch_dice,
                         'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp,
                         'alpha': alpha, 'beta': beta},
                        {}, weight_ce=1, weight_tversky=1,
                        ignore_label=self.label_manager.ignore_label,
                    )
                    print(f'  + {cname} weight={cweight} (alpha={alpha} FP, beta={beta} FN)')
                    short_names.append(f"T(α{alpha},β{beta})+CE" if cweight == 1.0 else f"{cweight}*T(α{alpha},β{beta})+CE")
                elif cname == 'BoundaryLoss':
                    idc = cparams.get('idc', [1])
                    engine = cparams.get('engine', 'torch')
                    max_dist = int(cparams.get('max_dist', 10))
                    normalize = bool(cparams.get('normalize', True))
                    spacing = cparams.get('spacing', None)
                    if spacing is not None:
                        spacing = tuple(spacing)
                    sub_loss = BoundaryLoss(idc=idc, engine=engine, max_dist=max_dist,
                                            normalize=normalize, spacing=spacing)
                    print(f'  + {cname} weight={cweight} (engine={engine}, max_dist={max_dist}, idc={idc})')
                    short_names.append(f"{cweight}*B" if cweight != 1.0 else "B")
                else:
                    # default DC_and_CE_loss
                    sub_loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                               'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                              {}, weight_ce=1, weight_dice=1,
                                              ignore_label=self.label_manager.ignore_label,
                                              dice_class=NewSoftDiceLoss)
                    print(f'  + DC_and_CE_loss weight={cweight}')
                    short_names.append("DC+CE" if cweight == 1.0 else f"{cweight}*(DC+CE)")
                built.append(sub_loss)
                built_weights.append(cweight)
                built_names.append(cname)

            # 短描述字串：給 log / CSV / progress.png suptitle 用
            self.loss_str = " + ".join(short_names)

            if len(built) == 1:
                loss = built[0]
                print(f'Loss: single ({built_names[0]}) → {self.loss_str}')
            else:
                loss = Compound_loss(built, built_weights, built_names)
                print(f'Loss: Compound  [{" + ".join(f"{n}*{w}" for n, w in zip(built_names, built_weights))}]')
                print(f'Loss str: {self.loss_str}')
            print('batch_dice:', self.configuration_manager.batch_dice, 'ddp', self.is_ddp)
            # loss = Log_DC_loss({'batch_dice': self.configuration_manager.batch_dice,
            #                        'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, weight_dice=1,
            #                       ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientLogDiceLoss)  
            # print('Loss: DC_and_CE_loss => Log_DC_loss')          


        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        print('weights!!!:', weights)

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
    
    #建立拆分的loss去看為什麼loss curve那麼奇怪
    def _build_ce_loss(self):
        if self.label_manager.has_regions:
            # region-based: BCE 作為 CE 的替代
            ce_loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientNewSoftDiceLoss,
                                   weight_dice=0, weight_ce=1)
            print('CE Loss (region-based): DC_and_BCE_loss with weight_dice=0')
        else:
            ce_loss = CE_loss({}, weight_ce=1, ignore_label=self.label_manager.ignore_label)
            print('Loss: CE_loss')

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights = weights / weights.sum()
        ce_loss = DeepSupervisionWrapper(ce_loss, weights)
        return ce_loss

    def _build_dice_loss(self):
        if self.label_manager.has_regions:
            # region-based: 用 sigmoid dice
            dice_loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientNewSoftDiceLoss,
                                   weight_dice=1, weight_ce=0)
            print('Dice Loss (region-based): DC_and_BCE_loss with weight_ce=0')
        else:
            dice_loss = DC_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                  ignore_label=self.label_manager.ignore_label, dice_class=NewSoftDiceLoss)
            print('Loss: DC_loss')

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights = weights / weights.sum()
        dice_loss = DeepSupervisionWrapper(dice_loss, weights)
        return dice_loss
    
    def _build_individual_dice_loss(self, target_level):
        """
        Build dice loss for a specific deep supervision level
        Args:
            target_level: which level to focus on (0-based index)
        """
        if self.label_manager.has_regions:
            dice_loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientNewSoftDiceLoss,
                                   weight_dice=1, weight_ce=0)
        else:
            dice_loss = DC_loss({'batch_dice': self.configuration_manager.batch_dice,
                                    'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                    ignore_label=self.label_manager.ignore_label, dice_class=NewSoftDiceLoss)
        print(f'Loss: DC_loss{target_level}')
        
        # Get the number of deep supervision levels
        num_levels = len(self._get_deep_supervision_scales())
        
        # Create weights array with 1 at target_level and 0 elsewhere
        weights = np.zeros(num_levels)
        weights[target_level] = 1.0
        
        # Normalize weights (though it's just 1 at target position)
        weights = weights / weights.sum()
        
        # Wrap the loss
        dice_loss = DeepSupervisionWrapper(dice_loss, weights)             
        return dice_loss


    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = {
                    'x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                }
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnunet for now)
        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            *rotation_for_DA.values(),
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = ("%s:" % dt_object, *args)

            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)

    def print_plans(self):
        if self.local_rank == 0:
            dct = deepcopy(self.plans_manager.plans)
            del dct['configurations']
            self.print_to_log_file(f"\nThis is the configuration used by this "
                                   f"training:\nConfiguration name: {self.configuration_name}\n",
                                   self.configuration_manager, '\n', add_timestamp=False)
            self.print_to_log_file('These are the global plan.json settings:\n', dct, '\n', add_timestamp=False)
            
            # Print training hyperparameters
            self.print_to_log_file('\n', add_timestamp=False)
            self.print_to_log_file('Training Hyperparameters:', add_timestamp=False)
            self.print_to_log_file(f'  num_iterations_per_epoch: {self.num_iterations_per_epoch}', add_timestamp=False)
            self.print_to_log_file(f'  num_epochs: {self.num_epochs}', add_timestamp=False)
            self.print_to_log_file(f'  initial_lr: {self.initial_lr}', add_timestamp=False)
            self.print_to_log_file(f'  optimizer_type: {self.optimizer_type}', add_timestamp=False)
            self.print_to_log_file(f'  lr_scheduler_type: {self.lr_scheduler_type}', add_timestamp=False)
            self.print_to_log_file(f'  oversample_foreground_percent: {self.oversample_foreground_percent}', add_timestamp=False)
            self.print_to_log_file(f'  oversample_foreground_percent_val: {self.oversample_foreground_percent_val}', add_timestamp=False)
            self.print_to_log_file(f'  enable_early_stopping: {self.enable_early_stopping}', add_timestamp=False)
            self.print_to_log_file(f'  early_stopping_patience: {self.early_stopping_patience}', add_timestamp=False)
            self.print_to_log_file(f'  early_stopping_min_delta: {self.early_stopping_min_delta}', add_timestamp=False)
            self.print_to_log_file('\n', add_timestamp=False)

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler based on settings.
        
        Optimizer options:
        - 'SGD': Stochastic Gradient Descent with momentum and Nesterov
        - 'AdamW': AdamW optimizer
        
        LR Scheduler options:
        - 'PolyLRScheduler': Polynomial learning rate decay
        - 'CosineAnnealingLR': Cosine annealing learning rate
        """
        # Configure optimizer
        if self.optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(
                self.network.parameters(), 
                self.initial_lr, 
                weight_decay=self.weight_decay,
                momentum=0.99, 
                nesterov=True
            )
            print(f'Optimizer: SGD with lr={self.initial_lr}, weight_decay={self.weight_decay}, momentum=0.99, nesterov=True')
        elif self.optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.network.parameters(), 
                self.initial_lr, 
                weight_decay=self.weight_decay
            )
            print(f'Optimizer: AdamW with lr={self.initial_lr}, weight_decay={self.weight_decay}')
        else:
            raise ValueError(f'Unknown optimizer type: {self.optimizer_type}. Choose from ["SGD", "AdamW"]')
        
        # Configure learning rate scheduler
        if self.lr_scheduler_type == 'PolyLRScheduler':
            lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
            print(f'LR Scheduler: PolyLRScheduler with initial_lr={self.initial_lr}, num_epochs={self.num_epochs}')
        elif self.lr_scheduler_type == 'CosineAnnealingLR':
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
            print(f'LR Scheduler: CosineAnnealingLR with T_max={self.num_epochs}')
        else:
            raise ValueError(f'Unknown lr_scheduler type: {self.lr_scheduler_type}. Choose from ["PolyLRScheduler", "CosineAnnealingLR"]')
        
        return optimizer, lr_scheduler    

    def plot_network_architecture(self):
        if self.local_rank == 0:
            try:
                # raise NotImplementedError('hiddenlayer no longer works and we do not have a viable alternative :-(')
                # pip install git+https://github.com/saugatkandel/hiddenlayer.git

                # from torchviz import make_dot
                # # not viable.
                # make_dot(tuple(self.network(torch.rand((1, self.num_input_channels,
                #                                         *self.configuration_manager.patch_size),
                #                                        device=self.device)))).render(
                #     join(self.output_folder, "network_architecture.pdf"), format='pdf')
                # self.optimizer.zero_grad()

                # broken.

                #做mlflow的model圖，Log model summary.
                txt_path = os.path.join(self.output_folder, 'model_summary_' + str(len(self.configuration_manager.n_conv_per_stage_encoder)) + 'L.txt')
                with open(txt_path, "w") as f:
                    #f.write(str(summary(self.network, (self.configuration_manager['batch_size'], self.num_input_channels, self.configuration_manager['patch_size'][0], self.configuration_manager['patch_size'][1], self.configuration_manager['patch_size'][2]))))
                    f.write(str(summary(self.network, 
                                        (self.configuration_manager.batch_size, self.num_input_channels, self.configuration_manager.patch_size[0], self.configuration_manager.patch_size[1], self.configuration_manager.patch_size[2]),
                                        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                                        depth=4,
                                        verbose=1
                                        ))) #depth=float('inf') => 預設全展開

                if not self.DISABLE_BUILTIN_MLFLOW:
                    mlflow.log_artifact(txt_path)

                #下面這個有可能失敗，所以放到後面做....
                import hiddenlayer as hl
                g = hl.build_graph(self.network,
                                   torch.rand((1, self.num_input_channels,
                                               *self.configuration_manager.patch_size),
                                              device=self.device),
                                   transforms=None)
                g.save(join(self.output_folder, "network_architecture.pdf"))
                del g

            except Exception as e:
                self.print_to_log_file("Unable to plot network architecture:")
                self.print_to_log_file(e)

                # self.print_to_log_file("\nprinting the network instead:\n")
                # self.print_to_log_file(self.network)
                # self.print_to_log_file("\n")
            finally:
                empty_cache(self.device)

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                self.sampling_categories = None
                splits = []
                all_keys_sorted = np.sort(list(dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append({})
                    splits[-1]['train'] = list(train_keys)
                    splits[-1]['val'] = list(test_keys)
                save_json(splits, splits_file)
                self.print_to_log_file(
                    "splits_final.json 未包含 sampling_categories；將不會啟用依類別調整的抽樣分配。",
                    also_print_to_console=True
                )

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                split_data = load_json(splits_file)
                # 支援新格式：{"splits": [...], "sampling_categories": {case_id: 1~4}}
                if isinstance(split_data, dict):
                    splits = split_data["splits"]
                    self.sampling_categories = split_data.get("sampling_categories")
                    if self.sampling_categories is not None:
                        self.print_to_log_file(
                            "Loaded sampling_categories for %d cases." % len(self.sampling_categories),
                            also_print_to_console=True
                        )
                    else:
                        self.print_to_log_file(
                            "splits_final.json 未發現 sampling_categories；將不會啟用依類別調整的抽樣分配。",
                            also_print_to_console=True
                        )
                else:
                    splits = split_data
                    self.sampling_categories = None
                    self.print_to_log_file(
                        "splits_final.json 為舊格式（list of splits），未包含 sampling_categories；將不會啟用依類別調整的抽樣分配。",
                        also_print_to_console=True
                    )
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        # num_images_properties_loading_threshold: 超過此數量才不預載 pkl
        # 設為 10000 讓所有 properties 預載入 RAM（~8 GB for 3000 cases），
        # 避免 24 個 worker 反覆讀取磁碟造成 I/O 瓶頸
        dataset_tr = nnUNetDataset(self.preprocessed_dataset_folder, tr_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=10000)
        dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=10000)
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline — augmentation config log
        aug_config = getattr(self, 'AUGMENTATION_CONFIG', None) or {}

        # 解析 noise_aug_channels / intensity_aug_channels → 反推各自要保護的 channel
        # 拆 2 類：
        #   noise（加法）— gaussian_noise，物理上模擬 scanner 噪聲，ADC 收
        #   intensity（乘法/非線性）— gaussian_blur, brightness, contrast, gamma, simulate_low_resolution，ADC 保護
        # 兩個 key 各自獨立，None=不保護任何 channel
        if aug_config:
            aug_config = dict(aug_config)  # 不污染上層的 AUGMENTATION_CONFIG
            n_ch = self.num_input_channels
            for key_in, key_out, label in [
                ("noise_aug_channels", "_protected_noise_channels_resolved", "noise aug"),
                ("intensity_aug_channels", "_protected_intensity_channels_resolved", "intensity aug"),
            ]:
                if aug_config.get(key_in) is not None:
                    allow = set(int(c) for c in aug_config[key_in])
                    protected = [c for c in range(n_ch) if c not in allow]
                    aug_config[key_out] = protected
                    self.print_to_log_file(
                        f"[Channel-restricted {label}] {key_in}={sorted(allow)}, "
                        f"protected={protected}（snapshot+restore）",
                        also_print_to_console=True,
                    )
            # spatial_label_channels：SpatialTransform 對這些 channel 走 nearest interp
            # （categorical label channel 不能用 cubic，會 overshoot 跳 label）
            spatial_labels = aug_config.get("spatial_label_channels")
            if spatial_labels:
                aug_config["_spatial_label_channels_resolved"] = list(int(c) for c in spatial_labels)
                self.print_to_log_file(
                    f"[Label-preserving spatial] spatial_label_channels={sorted(spatial_labels)}"
                    f"（這些 channel 在 SpatialTransform 走 nearest interp，避免 cubic overshoot）",
                    also_print_to_console=True,
                )

        if aug_config:
            aug_lines = ["", "=" * 60, "Data Augmentation Config（from MUTP recipe）", "=" * 60]
            # 開關
            switches = {
                "rotation": ("隨機旋轉", True),
                "scaling": ("隨機縮放", True),
                "elastic_deformation": ("彈性形變", False),
                "gaussian_noise": ("高斯雜訊", True),
                "gaussian_blur": ("高斯模糊", True),
                "brightness_multiply": ("亮度乘法", True),
                "contrast": ("對比度增強", True),
                "gamma_transform": ("Gamma 轉換", True),
                "simulate_low_resolution": ("低解析度模擬", False),
                "mirror": ("鏡像翻轉", True),
            }
            for key, (desc, default) in switches.items():
                val = aug_config.get(key, default)
                status = "ON" if val else "OFF"
                changed = "" if val == default else " ← 已修改"
                aug_lines.append(f"  {desc:<16} ({key}): {status}{changed}")
            # 數值參數
            params = {
                "rotation_p": ("旋轉機率", 0.2),
                "scaling_range": ("縮放範圍", [0.7, 1.4]),
                "scaling_p": ("縮放機率", 0.2),
                "gaussian_noise_p": ("雜訊機率", 0.1),
                "gaussian_blur_p": ("模糊機率", 0.2),
                "brightness_p": ("亮度機率", 0.15),
                "contrast_p": ("對比度機率", 0.15),
                "simulate_low_resolution_p": ("低解析度機率", 0.25),
                "simulate_low_resolution_zoom": ("低解析度縮放", [0.5, 1.0]),
            }
            for key, (desc, default) in params.items():
                val = aug_config.get(key, default)
                changed = "" if val == default else " ← 已修改"
                aug_lines.append(f"  {desc:<16} ({key}): {val}{changed}")
            aug_lines.append("=" * 60)
            self.print_to_log_file("\n".join(aug_lines), also_print_to_console=True)
        else:
            self.print_to_log_file("Data Augmentation: 使用 nnU-Net 預設值（無 MUTP recipe 覆蓋）",
                                   also_print_to_console=True)

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data=3, order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            aug_config=aug_config,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size, dim)

        allowed_num_processes = get_allowed_n_proc_DA()
        allowed_num_processes = 24
        #num_cached = max(12, allowed_num_processes * 2)
        print('allowed_num_processes:', allowed_num_processes)

        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            print('used LimitedLenWrapper!!!')
            mt_gen_train = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr, transform=tr_transforms,
                                             num_processes=allowed_num_processes, num_cached=20, seeds=None,
                                             pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
                                           transform=val_transforms, num_processes=allowed_num_processes,
                                           num_cached=20, seeds=None, pin_memory=self.device.type == 'cuda',
                                           wait_time=0.02)
        return mt_gen_train, mt_gen_val

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        tr_keys = list(dataset_tr.keys())
        sampling_categories = getattr(self, "sampling_categories", None)
        enable_sampling_weights = getattr(self, "ENABLE_SAMPLING_WEIGHTS", False)

        if enable_sampling_weights:
            category_weights = getattr(self.__class__, "SAMPLING_CATEGORY_WEIGHTS", None)
            sampling_weight_mode = getattr(self.__class__, "SAMPLING_CATEGORY_WEIGHT_MODE", "multiplier")
            sampling_probabilities = build_sampling_probabilities(
                tr_keys,
                sampling_categories=sampling_categories,
                category_weights=category_weights,
                mode=sampling_weight_mode,
            )
            self.print_to_log_file(
                "ENABLE_SAMPLING_WEIGHTS=True → 啟用依類別加權取樣",
                also_print_to_console=True
            )
        else:
            category_weights = None
            sampling_weight_mode = "multiplier"
            sampling_probabilities = None
            self.print_to_log_file(
                "ENABLE_SAMPLING_WEIGHTS=False → 不啟用依類別加權取樣（uniform sampling）",
                also_print_to_console=True
            )

        # 記錄本次訓練使用的 sampling mode（不論是否成功啟用 sampling_probabilities）
        self.print_to_log_file(
            f"Sampling category weight mode: {sampling_weight_mode}",
            also_print_to_console=True
        )

        # 訓練開始前預覽前 5 筆 sampling_probabilities（含 key 與類別），方便確認抽樣機率是否有變化
        if sampling_probabilities is None:
            self.print_to_log_file(
                "Sampling probabilities preview: None",
                also_print_to_console=True
            )
        else:
            n_preview = min(5, len(tr_keys))
            preview_lines = ["Sampling probabilities preview (first %d):" % n_preview]
            for idx in range(n_preview):
                k = tr_keys[idx]
                c = sampling_categories.get(k, 0) if sampling_categories is not None else 0
                preview_lines.append("  [%d] %s | category=%s | p=%.8f" % (idx, k, str(c), float(sampling_probabilities[idx])))
            self.print_to_log_file("\n".join(preview_lines), add_timestamp=False, also_print_to_console=True)

        # 若有 sampling_categories，打印各類別總數並寫入 training_log_日期時間.txt
        if sampling_categories is not None:
            from collections import Counter
            counts = Counter(sampling_categories.get(k, 0) for k in tr_keys)
            # 從實際資料取得所有 category IDs（不再 hardcode 1-4）
            category_ids = sorted(counts.keys())
            fg_category_ids = [c for c in category_ids if c > 0]  # 排除 0（negative）
            lines = ["Sampling category counts (training set):", "=" * 40]

            # 打印採樣比例（只顯示實際存在的 category）
            if enable_sampling_weights and category_weights and fg_category_ids:
                weight_vals = [float(category_weights.get(c, 1.0)) for c in fg_category_ids]
                ratio_str = ":".join(str(int(w)) if abs(w - int(w)) < 1e-9 else ("%g" % w) for w in weight_vals)
                w_sum = sum(weight_vals)
                target_pct = ":".join("%.2f%%" % (w / w_sum * 100.0) for w in weight_vals) if w_sum > 0 else "N/A"
                cat_labels = ":".join(str(c) for c in fg_category_ids)
                lines.append("Configured sampling ratio (Category %s weights): %s" % (cat_labels, ratio_str))
                lines.append("Configured sampling target proportions (normalized): %s" % target_pct)
            else:
                lines.append("Sampling weights: 未啟用（uniform sampling）")
            lines.append("Sampling category weight mode: %s" % sampling_weight_mode)

            # 校正後的期望類別抽樣比例
            if sampling_probabilities is not None and enable_sampling_weights:
                expected_mass = {}
                for k, p in zip(tr_keys, sampling_probabilities):
                    c = sampling_categories.get(k, 0)
                    expected_mass[c] = expected_mass.get(c, 0.0) + float(p)
                expected_ids = sorted(expected_mass.keys())
                expected_pct = ":".join("%.2f%%" % (expected_mass.get(c, 0.0) * 100.0) for c in expected_ids)
                lines.append("Corrected expected sampling proportions (by category, from probabilities): %s" % expected_pct)
            lines.append("-" * 40)

            total = len(tr_keys)
            for c in category_ids:
                n = counts.get(c, 0)
                ratio = (n / total * 100.0) if total > 0 else 0.0
                w = category_weights.get(c, 1.0) if category_weights else 1.0
                lines.append("  Category %d: %d samples (%.2f%%) (weight %.1f)" % (c, n, ratio, w))
            lines.append("  Total: %d samples" % len(tr_keys))
            lines.append("")
            msg = "\n".join(lines)
            self.print_to_log_file(msg, add_timestamp=False)
            log_filename = "training_log_%s.txt" % datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(join(self.output_folder, log_filename), "w", encoding="utf-8") as f:
                f.write(msg)
            self.print_to_log_file("Category counts written to: %s" % log_filename, add_timestamp=False)
        else:
            self.print_to_log_file(
                "未使用 sampling_categories（splits_final.json 未提供或未載入）；抽樣分配不會依類別調整。",
                also_print_to_console=True
            )

        enable_normal_upsample = getattr(self, "ENABLE_NORMAL_UPSAMPLE", False)
        if enable_normal_upsample:
            normal_class_weights = getattr(self.__class__, "NORMAL_CLASS_WEIGHTS", None)
            self.print_to_log_file(
                f"ENABLE_NORMAL_UPSAMPLE=True → 啟用 normal upsample 模式, weights: {normal_class_weights}",
                also_print_to_console=True
            )
        else:
            normal_class_weights = None
            self.print_to_log_file(
                "ENABLE_NORMAL_UPSAMPLE=False → 不啟用 normal upsample（所有座標合併隨機取樣）",
                also_print_to_console=True
            )

        compute_positives = getattr(self, "has_cls_head", False)
        cls_foreground_labels = getattr(self, "CLS_FOREGROUND_LABELS", None)
        self.print_to_log_file(
            f"compute_positives={compute_positives} (僅 classifier 架構才計算 positives label)",
            also_print_to_console=True
        )

        best_val_classes = getattr(self, "BEST_VAL_CLASSES", None)
        if best_val_classes is not None:
            self.print_to_log_file(
                f"BEST_VAL_CLASSES={best_val_classes} → best checkpoint 只看 class {best_val_classes} 的 dice",
                also_print_to_console=True
            )
        else:
            self.print_to_log_file(
                "BEST_VAL_CLASSES=None → best checkpoint 看全部 foreground class 的 dice 平均",
                also_print_to_console=True
            )
        if cls_foreground_labels is not None:
            self.print_to_log_file(
                f"CLS_FOREGROUND_LABELS={cls_foreground_labels} (分類頭只看這些 label 判斷 positive)",
                also_print_to_console=True
            )

        # 梯度累積
        ga_enabled = getattr(self, "ENABLE_GRADIENT_ACCUMULATION", False)
        ga_steps = getattr(self, "GRADIENT_ACCUMULATION_STEPS", 1)
        if ga_enabled and ga_steps > 1:
            effective_batch = self.batch_size * ga_steps
            self.print_to_log_file(
                f"GRADIENT_ACCUMULATION=ON → 每 {ga_steps} 步累積一次（"
                f"實際 batch={self.batch_size}, 等效 batch={effective_batch}）",
                also_print_to_console=True
            )
        else:
            self.print_to_log_file(
                "GRADIENT_ACCUMULATION=OFF → 每步都更新（標準模式）",
                also_print_to_console=True
            )

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=sampling_probabilities, pad_sides=None,
                                       sampling_categories=sampling_categories,
                                       normal_class_weights=normal_class_weights,
                                       compute_positives=compute_positives,
                                       cls_foreground_labels=cls_foreground_labels)
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent_val,
                                        sampling_probabilities=None, pad_sides=None,
                                        sampling_categories=sampling_categories,
                                        normal_class_weights=normal_class_weights,
                                        compute_positives=compute_positives,
                                        cls_foreground_labels=cls_foreground_labels)
        else:
            dl_tr = nnUNetDataLoader3D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=sampling_probabilities, pad_sides=None,
                                       sampling_categories=sampling_categories,
                                       normal_class_weights=normal_class_weights,
                                       compute_positives=compute_positives,
                                       cls_foreground_labels=cls_foreground_labels)
            dl_val = nnUNetDataLoader3D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent_val,
                                        sampling_probabilities=None, pad_sides=None,
                                        sampling_categories=sampling_categories,
                                        normal_class_weights=normal_class_weights,
                                        compute_positives=compute_positives)
        return dl_tr, dl_val

    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None,
                                aug_config: dict = None) -> AbstractTransform:
        # aug_config: recipe 的 augmentation 設定（dict），None=使用預設值
        ac = aug_config or {}

        # 受保護 channel：套 transform 前 snapshot、套完 restore
        # 拆 2 類（由 nnUNetTrainer 在呼叫此 method 前先解析放進 aug_config）：
        #   _protected_noise_channels_resolved      — GaussianNoise 用（加法噪聲）
        #   _protected_intensity_channels_resolved  — Brightness/Contrast/Gamma/Blur/SimLowRes 用（乘法/非線性）
        protected_noise = list(ac.get("_protected_noise_channels_resolved", []) or [])
        protected_intensity = list(ac.get("_protected_intensity_channels_resolved", []) or [])

        def _protect_noise(transform):
            if protected_noise:
                return ProtectedChannelsWrapper(transform, protected_noise)
            return transform

        def _protect(transform):
            """乘法/非線性 transform 用的保護（通常比 noise 嚴格）。"""
            if protected_intensity:
                return ProtectedChannelsWrapper(transform, protected_intensity)
            return transform

        tr_transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        # 空間變換（旋轉 + 縮放 + 彈性形變）
        do_rotation = ac.get("rotation", True)
        do_scale = ac.get("scaling", True)
        do_elastic = ac.get("elastic_deformation", False)
        scale_range = tuple(ac.get("scaling_range", [0.7, 1.4]))
        rot_p = ac.get("rotation_p", 0.2)
        scale_p = ac.get("scaling_p", 0.2)

        _spatial = SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=None,
            do_elastic_deform=do_elastic, alpha=(0, 900), sigma=(9, 13),
            do_rotation=do_rotation, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
            p_rot_per_axis=1,
            do_scale=do_scale, scale=scale_range,
            border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
            random_crop=False,
            p_el_per_sample=0.2 if do_elastic else 0,
            p_scale_per_sample=scale_p,
            p_rot_per_sample=rot_p,
            independent_scale_for_each_axis=False
        )
        # 如果 recipe 標了 spatial_label_channels（如 SynthSeg ch2），用 LabelPreservingSpatialTransform
        # 包起來：對 label channel 走 nearest interp 避 cubic overshoot
        spatial_labels = ac.get("_spatial_label_channels_resolved", []) or []
        if spatial_labels:
            tr_transforms.append(LabelPreservingSpatialTransform(_spatial, spatial_labels))
        else:
            tr_transforms.append(_spatial)

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        # 高斯雜訊（加法 noise → 用 _protect_noise，預設可讓 ADC 收）
        if ac.get("gaussian_noise", True):
            tr_transforms.append(_protect_noise(
                GaussianNoiseTransform(p_per_sample=ac.get("gaussian_noise_p", 0.1))))

        # 高斯模糊
        if ac.get("gaussian_blur", True):
            tr_transforms.append(_protect(
                GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True,
                                      p_per_sample=ac.get("gaussian_blur_p", 0.2),
                                      p_per_channel=0.5)))

        # 亮度乘法
        if ac.get("brightness_multiply", True):
            tr_transforms.append(_protect(
                BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25),
                                                  p_per_sample=ac.get("brightness_p", 0.15))))

        # 對比度增強
        if ac.get("contrast", True):
            tr_transforms.append(_protect(
                ContrastAugmentationTransform(p_per_sample=ac.get("contrast_p", 0.15))))

        # 低解析度模擬（nearest 下採樣 + cubic 上採樣會破壞 integer label / 物理量）
        if ac.get("simulate_low_resolution", False):
            zoom = tuple(ac.get("simulate_low_resolution_zoom", [0.5, 1.0]))
            tr_transforms.append(_protect(SimulateLowResolutionTransform(
                zoom_range=zoom, per_channel=True, p_per_channel=0.5,
                order_downsample=0, order_upsample=3,
                p_per_sample=ac.get("simulate_low_resolution_p", 0.25),
                ignore_axes=ignore_axes)))

        # Gamma 轉換（兩組）
        if ac.get("gamma_transform", True):
            tr_transforms.append(_protect(
                GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1)))
            tr_transforms.append(_protect(
                GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3)))

        # 鏡像翻轉
        if ac.get("mirror", True) and mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                               mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            tr_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(foreground_labels), 0)),
                p_per_sample=0.4,
                key="data",
                strel_size=(1, 8),
                p_per_label=1))
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15))

        tr_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            tr_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                       if ignore_label is not None else regions,
                                                                       'target', 'target'))

        if deep_supervision_scales is not None:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                              output_key='target'))
        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms(deep_supervision_scales: Union[List, Tuple],
                                  is_cascaded: bool = False,
                                  foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                  regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                  ignore_label: int = None) -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            val_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))

        val_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            val_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                        if ignore_label is not None else regions,
                                                                        'target', 'target'))

        if deep_supervision_scales is not None:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))

        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            self.network.module.decoder.deep_supervision = enabled
        else:
            self.network.decoder.deep_supervision = enabled

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(True)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            #unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
            #               num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
            unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=64)
            self.print_to_log_file('unpacking done...')

        if self.is_ddp:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")

    def on_train_end(self):
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        # now we can delete latest
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        empty_cache(self.device)

    def on_train_epoch_start(self):
        self.network.train()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def train_step(self, batch: dict) -> dict:
        #這邊修改成同時計算dice，以方便畫train dice去比較有無overfitting
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        if self.has_cls_head:
            positive = batch['positives']
            if not isinstance(positive, torch.Tensor):
                positive = torch.from_numpy(positive)
            positive = positive.to(self.device, non_blocking=True)

        # 梯度累積：只在累積週期的第一步 zero_grad
        accum_steps = self.GRADIENT_ACCUMULATION_STEPS if self.ENABLE_GRADIENT_ACCUMULATION else 1
        if not hasattr(self, '_accum_counter'):
            self._accum_counter = 0
        is_first_accum = (self._accum_counter % accum_steps == 0)
        is_last_accum = ((self._accum_counter + 1) % accum_steps == 0)

        if is_first_accum:
            self.optimizer.zero_grad()

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            if self.has_cls_head:
                output_seg, output_cls = self.network(data)
            elif self.has_aux_seg_head:
                # Dual-head: network 回傳 (main_logits, aux_logits)
                # target 必須是 2-channel (infarct + region)，由 seg_from_prev_stage 機制疊起
                output_seg, output_aux = self.network(data)
                # split target → main_target (infarct, ch0), aux_target (region, ch1)
                if isinstance(target, list):
                    target_main = [t[:, 0:1] for t in target]
                    target_aux  = [t[:, 1:2] for t in target]
                else:
                    target_main = target[:, 0:1]
                    target_aux  = target[:, 1:2]
            else:
                output_seg = self.network(data)

            if self.has_aux_seg_head:
                # output_seg = main → loss_main 對 infarct；output_aux → loss_aux 對 region
                seg_l = self.loss(output_seg, target_main)
                aux_l = self.aux_loss(output_aux, target_aux)
            else:
                seg_l = self.loss(output_seg, target)

            if self.has_cls_head:
                cls_l = self.cls_loss(output_cls, positive.squeeze(1).long())
                l = seg_l + cls_l
            elif self.has_aux_seg_head:
                l = seg_l + self.aux_loss_weight * aux_l
            else:
                l = seg_l

            # 梯度累積：backward 用除過的 (等效大 batch 平均)，logging 保留 pre-GA 值
            # 讓 train_loss / train_seg_loss / train_loss_CE 三者的量級一致，跨 GA/non-GA 實驗可比對
            l_for_log = l.detach()   # 原始未除 accum 的總 loss（跟 CE+Dice 對得起來）
            if accum_steps > 1:
                l = l / accum_steps

            # 額外 loss 分項 + 每層 dice（僅 deep supervision logging 開啟時）
            if self.enable_deep_supervision_logging:
                _ds_target = target_main if self.has_aux_seg_head else target
                ce_l = self.ce_loss(output_seg, _ds_target)
                dice_l = self.dice_loss(output_seg, _ds_target)
                individual_dice_losses = {}
                for i in range(self.num_deep_supervision_levels):
                    individual_dice_losses[f'dice_l{i}'] = self.individual_dice_losses[f'dice_loss{i}'](output_seg, _ds_target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            if is_last_accum:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
        else:
            l.backward()
            if is_last_accum:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        self._accum_counter += 1

        # classification metrics (only when has_cls_head)
        if self.has_cls_head:
            preds = torch.argmax(output_cls, dim=1)
            positive_flat = positive.squeeze(1).long()
            pred_pos = (preds > 0)
            gt_pos = (positive_flat > 0)
            TP = (pred_pos & gt_pos).sum().item()
            TN = (~pred_pos & ~gt_pos).sum().item()
            FP = (pred_pos & ~gt_pos).sum().item()
            FN = (~pred_pos & gt_pos).sum().item()
            accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
            sensitivity = TP / (TP + FN + 1e-8)
            specificity = TN / (TN + FP + 1e-8)

        # ---- Pseudo dice metrics（只算 i==0 最高解析度，其餘層由開關控制）----
        output0 = output_seg[0]
        target0 = target[0]
        axes = [0] + list(range(2, len(output0.shape)))
        axes0 = list(range(2, len(output0.shape)))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot0 = (torch.sigmoid(output0) > 0.5).long()
        else:
            output_seg_argmax = output0.argmax(1)[:, None]
            predicted_segmentation_onehot0 = torch.zeros(output0.shape, device=output0.device, dtype=torch.float32)
            predicted_segmentation_onehot0.scatter_(1, output_seg_argmax, 1)
            del output_seg_argmax

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask0 = (target0 != self.label_manager.ignore_label).float()
                target0[target0 == self.label_manager.ignore_label] = 0
            else:
                mask0 = 1 - target0[:, -1:]
                target0 = target0[:, :-1]
        else:
            mask0 = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot0, target0, axes=axes, mask=mask0)
        tp0, fp0, fn0, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot0, target0, axes=axes0, mask=mask0)

        tp_hard = tp.detach()
        fp_hard = fp.detach()
        fn_hard = fn.detach()

        smooth = 1e-5
        tp_hard0 = tp0.detach()
        fp_hard0 = fp0.detach()
        fn_hard0 = fn0.detach()
        nominator = 2 * tp_hard0
        denominator = 2 * tp_hard0 + fp_hard0 + fn_hard0
        dc0_val = (nominator + smooth) / (torch.clip(denominator + smooth, 1e-8))

        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
            dc0_val = dc0_val[:, 1:].mean()
        else:
            dc0_val = dc0_val.mean()

        # 組裝 result
        # 'loss' 用 pre-GA 版本 (l_for_log)，不用 l.detach()（已被 GA 除過會偏低）
        result = {
            'loss': l_for_log.cpu().numpy(),
            'seg_loss': seg_l.detach().cpu().numpy(),
            'tp_hard': tp_hard.cpu().numpy(),
            'fp_hard': fp_hard.cpu().numpy(),
            'fn_hard': fn_hard.cpu().numpy(),
            'dc0': dc0_val.cpu().numpy(),
        }
        # 把每個 loss component 的數值放進 result（compound loss 才有）
        _lc = getattr(self.loss, 'last_components', None)
        if _lc:
            result['loss_components'] = dict(_lc)

        if self.enable_deep_supervision_logging:
            result['ce_loss'] = ce_l.detach().cpu().numpy()
            result['dice_loss'] = dice_l.detach().cpu().numpy()
            for i in range(self.num_deep_supervision_levels):
                result[f'dice_loss{i}'] = individual_dice_losses[f'dice_l{i}'].detach().cpu().numpy()

            # 計算每層 dc + fake_dcl
            metrics = {'dc0': dc0_val.cpu().numpy()}
            fake_dcl_all = [dc0_val.cpu().numpy()]
            weights = np.array([1 / (2 ** i) for i in range(len(output_seg))])
            weights = weights / weights.sum()

            for i in range(1, len(output_seg)):
                out_i = output_seg[i]
                tgt_i = target[i]
                axes_i = list(range(2, len(out_i.shape)))
                if self.label_manager.has_regions:
                    pred_i = (torch.sigmoid(out_i) > 0.5).long()
                else:
                    seg_argmax_i = out_i.argmax(1)[:, None]
                    pred_i = torch.zeros(out_i.shape, device=out_i.device, dtype=torch.float32)
                    pred_i.scatter_(1, seg_argmax_i, 1)
                if self.label_manager.has_ignore_label:
                    if not self.label_manager.has_regions:
                        mask_i = (tgt_i != self.label_manager.ignore_label).float()
                        tgt_i = tgt_i.clone()
                        tgt_i[tgt_i == self.label_manager.ignore_label] = 0
                    else:
                        mask_i = 1 - tgt_i[:, -1:]
                        tgt_i = tgt_i[:, :-1]
                else:
                    mask_i = None
                tp_i, fp_i, fn_i, _ = get_tp_fp_fn_tn(pred_i, tgt_i, axes=axes_i, mask=mask_i)
                dc_i = (2 * tp_i.detach() + smooth) / (torch.clip(2 * tp_i.detach() + fp_i.detach() + fn_i.detach() + smooth, 1e-8))
                if not self.label_manager.has_regions:
                    dc_i = dc_i[:, 1:].mean()
                else:
                    dc_i = dc_i.mean()
                metrics[f'dc{i}'] = dc_i.cpu().numpy()
                fake_dcl_all.append(dc_i.cpu().numpy())

            fake_dcl = 1 - np.sum(np.array(fake_dcl_all) * weights)
            result['fake_dcl'] = fake_dcl
            for i in range(self.num_deep_supervision_levels):
                result[f'dc{i}'] = metrics.get(f'dc{i}', 0)

        if self.has_cls_head:
            result['cls_loss'] = cls_l.detach().cpu().numpy()
            result['accuracy'] = accuracy
            result['sensitivity'] = sensitivity
            result['specificity'] = specificity
        return result

    def _ddp_gather_mean(self, outputs, key):
        """DDP helper: gather and mean a key from outputs."""
        buf = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(buf, outputs[key])
        return np.vstack(buf).mean()

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)
        tp = np.sum(outputs['tp_hard'], 0)
        fp = np.sum(outputs['fp_hard'], 0)
        fn = np.sum(outputs['fn_hard'], 0)

        if self.is_ddp:
            loss_here = self._ddp_gather_mean(outputs, 'loss')
            seg_loss_here = self._ddp_gather_mean(outputs, 'seg_loss')

            world_size = dist.get_world_size()
            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)
            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)
            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            if self.enable_deep_supervision_logging:
                ce_loss_here = self._ddp_gather_mean(outputs, 'ce_loss')
                dice_loss_here = self._ddp_gather_mean(outputs, 'dice_loss')
                fake_dice_loss_here = self._ddp_gather_mean(outputs, 'fake_dcl')
                individual_dice_loss_here = {}
                individual_dice_here = {}
                for i in range(self.num_deep_supervision_levels):
                    individual_dice_loss_here[f'dice_loss{i}_here'] = self._ddp_gather_mean(outputs, f'dice_loss{i}')
                    individual_dice_here[f'dice{i}_here'] = self._ddp_gather_mean(outputs, f'dc{i}')

            if self.has_cls_head:
                cls_loss_here = self._ddp_gather_mean(outputs, 'cls_loss')
                accuracys = [None for _ in range(world_size)]
                dist.all_gather_object(accuracys, outputs['accuracy'])
                accuracy_here = np.mean(np.concatenate(accuracys))
                sensitivitys = [None for _ in range(world_size)]
                dist.all_gather_object(sensitivitys, outputs['sensitivity'])
                sensitivity_here = np.mean(np.concatenate(sensitivitys))
                specificitys = [None for _ in range(world_size)]
                dist.all_gather_object(specificitys, outputs['specificity'])
                specificity_here = np.mean(np.concatenate(specificitys))
        else:
            loss_here = np.mean(outputs['loss'])
            seg_loss_here = np.mean(outputs['seg_loss'])

            if self.enable_deep_supervision_logging:
                ce_loss_here = np.mean(outputs['ce_loss'])
                dice_loss_here = np.mean(outputs['dice_loss'])
                fake_dice_loss_here = np.mean(outputs['fake_dcl'])
                individual_dice_loss_here = {}
                individual_dice_here = {}
                for i in range(self.num_deep_supervision_levels):
                    individual_dice_here[f'dice{i}_here'] = np.mean(outputs[f'dc{i}'])
                    individual_dice_loss_here[f'dice_loss{i}_here'] = np.mean(outputs[f'dice_loss{i}'])

            if self.has_cls_head:
                cls_loss_here = np.mean(outputs['cls_loss'])
                accuracy_here = np.mean(outputs['accuracy'])
                sensitivity_here = np.mean(outputs['sensitivity'])
                specificity_here = np.mean(outputs['specificity'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)

        self.logger.log('train_losses', loss_here, self.current_epoch)
        self.logger.log('train_seg_losses', seg_loss_here, self.current_epoch)
        self.logger.log('train_mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('train_dice_per_class_or_region', global_dc_per_class, self.current_epoch)

        # === per-component loss 加總平均（compound loss 才有）===
        if 'loss_components' in outputs and outputs['loss_components']:
            comp_dicts = outputs['loss_components']   # list of dict
            comp_keys = comp_dicts[0].keys() if comp_dicts else []
            for k in comp_keys:
                vals = [d.get(k, float('nan')) for d in comp_dicts]
                key_name = f'train_loss_{k}'
                if key_name not in self.logger.my_fantastic_logging:
                    self.logger.my_fantastic_logging[key_name] = list()
                self.logger.log(key_name, float(np.nanmean(vals)), self.current_epoch)

        if self.enable_deep_supervision_logging:
            self.logger.log('train_ce_losses', ce_loss_here, self.current_epoch)
            self.logger.log('train_dice_losses', dice_loss_here, self.current_epoch)
            self.logger.log('train_fake_dice_losses', fake_dice_loss_here, self.current_epoch)
            for i in range(self.num_deep_supervision_levels):
                self.logger.log(f'train_dice_loss{i}', individual_dice_loss_here[f'dice_loss{i}_here'], self.current_epoch)
                self.logger.log(f'train_supervision_dice{i}', individual_dice_here[f'dice{i}_here'], self.current_epoch)

        if self.has_cls_head:
            self.logger.log('train_cls_losses', cls_loss_here, self.current_epoch)
            self.logger.log('train_accuracys', accuracy_here, self.current_epoch)
            self.logger.log('train_sensitivitys', sensitivity_here, self.current_epoch)
            self.logger.log('train_specificitys', specificity_here, self.current_epoch)

    def on_validation_epoch_start(self):
        self.network.eval()

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        if self.has_cls_head:
            positive = batch['positives']
            if not isinstance(positive, torch.Tensor):
                positive = torch.from_numpy(positive)
            positive = positive.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            if self.has_cls_head:
                output_seg, output_cls = self.network(data)
            elif self.has_aux_seg_head:
                output_seg, output_aux = self.network(data)
                if isinstance(target, list):
                    target_main = [t[:, 0:1] for t in target]
                    target_aux  = [t[:, 1:2] for t in target]
                else:
                    target_main = target[:, 0:1]
                    target_aux  = target[:, 1:2]
            else:
                output_seg = self.network(data)
            del data

            if self.has_aux_seg_head:
                seg_l = self.loss(output_seg, target_main)
                aux_l = self.aux_loss(output_aux, target_aux)
            else:
                seg_l = self.loss(output_seg, target)

            if self.has_cls_head:
                cls_l = self.cls_loss(output_cls, positive.squeeze(1).long())
                l = seg_l + cls_l
            elif self.has_aux_seg_head:
                l = seg_l + self.aux_loss_weight * aux_l
            else:
                l = seg_l

            if self.enable_deep_supervision_logging:
                _ds_target = target_main if self.has_aux_seg_head else target
                ce_l = self.ce_loss(output_seg, _ds_target)
                dice_l = self.dice_loss(output_seg, _ds_target)
                individual_dice_losses = {}
                for i in range(self.num_deep_supervision_levels):
                    individual_dice_losses[f'dice_l{i}'] = self.individual_dice_losses[f'dice_loss{i}'](output_seg, _ds_target)

        # classification metrics (only when has_cls_head)
        if self.has_cls_head:
            preds = torch.argmax(output_cls, dim=1)
            positive_flat = positive.squeeze(1).long()
            pred_pos = (preds > 0)
            gt_pos = (positive_flat > 0)
            TP = (pred_pos & gt_pos).sum().item()
            TN = (~pred_pos & ~gt_pos).sum().item()
            FP = (pred_pos & ~gt_pos).sum().item()
            FN = (~pred_pos & gt_pos).sum().item()
            accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
            sensitivity = TP / (TP + FN + 1e-8)
            specificity = TN / (TN + FP + 1e-8)

        # ---- Pseudo dice metrics（只算 i==0 最高解析度；dual-head 只算 infarct 主 head）----
        output0 = output_seg[0]
        target0 = (target_main if self.has_aux_seg_head else target)[0]
        axes = [0] + list(range(2, len(output0.shape)))
        axes0 = list(range(2, len(output0.shape)))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot0 = (torch.sigmoid(output0) > 0.5).long()
        else:
            output_seg_argmax = output0.argmax(1)[:, None]
            predicted_segmentation_onehot0 = torch.zeros(output0.shape, device=output0.device, dtype=torch.float32)
            predicted_segmentation_onehot0.scatter_(1, output_seg_argmax, 1)
            del output_seg_argmax

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask0 = (target0 != self.label_manager.ignore_label).float()
                target0[target0 == self.label_manager.ignore_label] = 0
            else:
                mask0 = 1 - target0[:, -1:]
                target0 = target0[:, :-1]
        else:
            mask0 = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot0, target0, axes=axes, mask=mask0)
        tp0, fp0, fn0, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot0, target0, axes=axes0, mask=mask0)

        tp_hard = tp.detach()
        fp_hard = fp.detach()
        fn_hard = fn.detach()

        smooth = 1e-5
        tp_hard0 = tp0.detach()
        fp_hard0 = fp0.detach()
        fn_hard0 = fn0.detach()
        nominator = 2 * tp_hard0
        denominator = 2 * tp_hard0 + fp_hard0 + fn_hard0
        dc0_val = (nominator + smooth) / (torch.clip(denominator + smooth, 1e-8))

        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
            dc0_val = dc0_val[:, 1:].mean()
        else:
            dc0_val = dc0_val.mean()

        result = {
            'loss': l.detach().cpu().numpy(),
            'seg_loss': seg_l.detach().cpu().numpy(),
            'tp_hard': tp_hard.cpu().numpy(),
            'fp_hard': fp_hard.cpu().numpy(),
            'fn_hard': fn_hard.cpu().numpy(),
            'dc0': dc0_val.cpu().numpy(),
        }
        # 把每個 loss component 的數值放進 result（compound loss 才有）
        _lc = getattr(self.loss, 'last_components', None)
        if _lc:
            result['loss_components'] = dict(_lc)

        if self.enable_deep_supervision_logging:
            result['ce_loss'] = ce_l.detach().cpu().numpy()
            result['dice_loss'] = dice_l.detach().cpu().numpy()
            for i in range(self.num_deep_supervision_levels):
                result[f'dice_loss{i}'] = individual_dice_losses[f'dice_l{i}'].detach().cpu().numpy()

            metrics = {'dc0': dc0_val.cpu().numpy()}
            fake_dcl_all = [dc0_val.cpu().numpy()]
            weights = np.array([1 / (2 ** i) for i in range(len(output_seg))])
            weights = weights / weights.sum()

            for i in range(1, len(output_seg)):
                out_i = output_seg[i]
                tgt_i = target[i]
                axes_i = list(range(2, len(out_i.shape)))
                if self.label_manager.has_regions:
                    pred_i = (torch.sigmoid(out_i) > 0.5).long()
                else:
                    seg_argmax_i = out_i.argmax(1)[:, None]
                    pred_i = torch.zeros(out_i.shape, device=out_i.device, dtype=torch.float32)
                    pred_i.scatter_(1, seg_argmax_i, 1)
                if self.label_manager.has_ignore_label:
                    if not self.label_manager.has_regions:
                        mask_i = (tgt_i != self.label_manager.ignore_label).float()
                        tgt_i = tgt_i.clone()
                        tgt_i[tgt_i == self.label_manager.ignore_label] = 0
                    else:
                        mask_i = 1 - tgt_i[:, -1:]
                        tgt_i = tgt_i[:, :-1]
                else:
                    mask_i = None
                tp_i, fp_i, fn_i, _ = get_tp_fp_fn_tn(pred_i, tgt_i, axes=axes_i, mask=mask_i)
                dc_i = (2 * tp_i.detach() + smooth) / (torch.clip(2 * tp_i.detach() + fp_i.detach() + fn_i.detach() + smooth, 1e-8))
                if not self.label_manager.has_regions:
                    dc_i = dc_i[:, 1:].mean()
                else:
                    dc_i = dc_i.mean()
                metrics[f'dc{i}'] = dc_i.cpu().numpy()
                fake_dcl_all.append(dc_i.cpu().numpy())

            fake_dcl = 1 - np.sum(np.array(fake_dcl_all) * weights)
            result['fake_dcl'] = fake_dcl
            for i in range(self.num_deep_supervision_levels):
                result[f'dc{i}'] = metrics.get(f'dc{i}', 0)

        if self.has_cls_head:
            result['cls_loss'] = cls_l.detach().cpu().numpy()
            result['accuracy'] = accuracy
            result['sensitivity'] = sensitivity
            result['specificity'] = specificity
        return result

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            loss_here = self._ddp_gather_mean(outputs_collated, 'loss')
            seg_loss_here = self._ddp_gather_mean(outputs_collated, 'seg_loss')

            if self.enable_deep_supervision_logging:
                ce_loss_here = self._ddp_gather_mean(outputs_collated, 'ce_loss')
                dice_loss_here = self._ddp_gather_mean(outputs_collated, 'dice_loss')
                fake_dice_loss_here = self._ddp_gather_mean(outputs_collated, 'fake_dcl')
                val_individual_dice_loss_here = {}
                val_individual_dice_here = {}
                for i in range(self.num_deep_supervision_levels):
                    val_individual_dice_loss_here[f'dice_loss{i}_here'] = self._ddp_gather_mean(outputs_collated, f'dice_loss{i}')
                    val_individual_dice_here[f'dice{i}_here'] = self._ddp_gather_mean(outputs_collated, f'dc{i}')

            if self.has_cls_head:
                cls_loss_here = self._ddp_gather_mean(outputs_collated, 'cls_loss')
                accuracys = [None for _ in range(world_size)]
                dist.all_gather_object(accuracys, outputs_collated['accuracy'])
                accuracy_here = np.mean(np.concatenate(accuracys))
                sensitivitys = [None for _ in range(world_size)]
                dist.all_gather_object(sensitivitys, outputs_collated['sensitivity'])
                sensitivity_here = np.mean(np.concatenate(sensitivitys))
                specificitys = [None for _ in range(world_size)]
                dist.all_gather_object(specificitys, outputs_collated['specificity'])
                specificity_here = np.mean(np.concatenate(specificitys))
        else:
            loss_here = np.mean(outputs_collated['loss'])
            seg_loss_here = np.mean(outputs_collated['seg_loss'])

            if self.enable_deep_supervision_logging:
                ce_loss_here = np.mean(outputs_collated['ce_loss'])
                dice_loss_here = np.mean(outputs_collated['dice_loss'])
                fake_dice_loss_here = np.mean(outputs_collated['fake_dcl'])
                val_individual_dice_loss_here = {}
                val_individual_dice_here = {}
                for i in range(self.num_deep_supervision_levels):
                    val_individual_dice_loss_here[f'dice_loss{i}_here'] = np.mean(outputs_collated[f'dice_loss{i}'])
                    val_individual_dice_here[f'dice{i}_here'] = np.mean(outputs_collated[f'dc{i}'])

            if self.has_cls_head:
                cls_loss_here = np.mean(outputs_collated['cls_loss'])
                accuracy_here = np.mean(outputs_collated['accuracy'])
                sensitivity_here = np.mean(outputs_collated['sensitivity'])
                specificity_here = np.mean(outputs_collated['specificity'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)

        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('val_seg_losses', seg_loss_here, self.current_epoch)

        # === per-component validation loss（compound loss 才有）===
        if 'loss_components' in outputs_collated and outputs_collated['loss_components']:
            comp_dicts = outputs_collated['loss_components']
            comp_keys = comp_dicts[0].keys() if comp_dicts else []
            for k in comp_keys:
                vals = [d.get(k, float('nan')) for d in comp_dicts]
                key_name = f'val_loss_{k}'
                if key_name not in self.logger.my_fantastic_logging:
                    self.logger.my_fantastic_logging[key_name] = list()
                self.logger.log(key_name, float(np.nanmean(vals)), self.current_epoch)

        if self.enable_deep_supervision_logging:
            self.logger.log('val_ce_losses', ce_loss_here, self.current_epoch)
            self.logger.log('val_dice_losses', dice_loss_here, self.current_epoch)
            self.logger.log('val_fake_dice_losses', fake_dice_loss_here, self.current_epoch)
            for i in range(self.num_deep_supervision_levels):
                self.logger.log(f'val_dice_loss{i}', val_individual_dice_loss_here[f'dice_loss{i}_here'], self.current_epoch)
                self.logger.log(f'val_supervision_dice{i}', val_individual_dice_here[f'dice{i}_here'], self.current_epoch)

        if self.has_cls_head:
            self.logger.log('val_cls_losses', cls_loss_here, self.current_epoch)
            self.logger.log('val_accuracys', accuracy_here, self.current_epoch)
            self.logger.log('val_sensitivitys', sensitivity_here, self.current_epoch)
            self.logger.log('val_specificitys', specificity_here, self.current_epoch)

    def _on_ema_validation_epoch_end(self, val_outputs: list):
        """EMA model 的驗證結果彙總，只計算 loss + mean_fg_dice，不重複 deep supervision logging。"""
        outputs_collated = collate_outputs(val_outputs)

        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()
            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)
            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)
            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)
            ema_loss_here = self._ddp_gather_mean(outputs_collated, 'loss')
        else:
            ema_loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]
        ema_mean_fg_dice = np.nanmean(global_dc_per_class)

        self.logger.log('ema_val_losses', ema_loss_here, self.current_epoch)
        self.logger.log('ema_mean_fg_dice', ema_mean_fg_dice, self.current_epoch)
        self.logger.log('ema_dice_per_class_or_region', global_dc_per_class, self.current_epoch)

    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        def _log_print(key, label=None):
            """Helper: print + mlflow log from logger."""
            label = label or key
            val = np.round(self.logger.my_fantastic_logging[key][-1], decimals=4)
            self.print_to_log_file(label, val)
            if not self.DISABLE_BUILTIN_MLFLOW:
                mlflow.log_metric(label, val, step=self.current_epoch)

        _log_print('train_losses', 'train_loss')
        _log_print('train_seg_losses', 'train_seg_loss')

        # 印出 per-component train loss（compound loss 才有，如 CE / Tversky / Boundary）
        for _comp_key in sorted(self.logger.my_fantastic_logging.keys()):
            if _comp_key.startswith('train_loss_') and len(self.logger.my_fantastic_logging[_comp_key]) > 0:
                _log_print(_comp_key, _comp_key)   # label 跟 key 一樣，如 'train_loss_CE'

        if self.enable_deep_supervision_logging:
            _log_print('train_ce_losses', 'train_ce_loss')
            _log_print('train_dice_losses', 'train_dice_loss')
            for i in range(self.num_deep_supervision_levels):
                _log_print(f'train_dice_loss{i}')
            _log_print('train_fake_dice_losses', 'train_fake_dice_loss')

        if self.has_cls_head:
            _log_print('train_cls_losses', 'train_cls_loss')
            _log_print('train_accuracys', 'train_accuracy')
            _log_print('train_sensitivitys', 'train_sensitivity')
            _log_print('train_specificitys', 'train_specificity')

        _log_print('val_losses', 'val_loss')
        _log_print('val_seg_losses', 'val_seg_loss')

        # 印出 per-component val loss（compound loss 才有）
        for _comp_key in sorted(self.logger.my_fantastic_logging.keys()):
            if _comp_key.startswith('val_loss_') and len(self.logger.my_fantastic_logging[_comp_key]) > 0:
                _log_print(_comp_key, _comp_key)

        if self.enable_deep_supervision_logging:
            _log_print('val_ce_losses', 'val_ce_loss')
            _log_print('val_dice_losses', 'val_dice_loss')
            for i in range(self.num_deep_supervision_levels):
                _log_print(f'val_dice_loss{i}')
            _log_print('val_fake_dice_losses', 'val_fake_dice_loss')

        if self.has_cls_head:
            _log_print('val_cls_losses', 'val_cls_loss')
            _log_print('val_accuracys', 'val_accuracy')
            _log_print('val_sensitivitys', 'val_sensitivity')
            _log_print('val_specificitys', 'val_specificity')

        self.print_to_log_file('train Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['train_dice_per_class_or_region'][-1]])
        self.print_to_log_file('val Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])

        # EMA model metrics
        if self.ENABLE_EMA and 'ema_mean_fg_dice' in self.logger.my_fantastic_logging:
            ema_dice = self.logger.my_fantastic_logging['ema_mean_fg_dice'][-1]
            ema_loss = self.logger.my_fantastic_logging['ema_val_losses'][-1]
            self.print_to_log_file(f'EMA val loss: {np.round(ema_loss, decimals=4)}')
            self.print_to_log_file(f'EMA val Pseudo dice: {np.round(ema_dice, decimals=4)}')
            if 'ema_dice_per_class_or_region' in self.logger.my_fantastic_logging:
                self.print_to_log_file('EMA val Pseudo dice per class',
                                       [np.round(i, decimals=4) for i in
                                        self.logger.my_fantastic_logging['ema_dice_per_class_or_region'][-1]])
            if not self.DISABLE_BUILTIN_MLFLOW:
                mlflow.log_metric("ema_val_loss", np.round(ema_loss, decimals=4), step=self.current_epoch)
                mlflow.log_metric("ema_val_pseudo_dice", np.round(ema_dice, decimals=4), step=self.current_epoch)

        if self.enable_deep_supervision_logging:
            for i in range(self.num_deep_supervision_levels):
                _log_print(f'train_supervision_dice{i}', f'train deep_supervision dice{i}')
                _log_print(f'val_supervision_dice{i}', f'val deep_supervision dice{i}')

        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        if not self.DISABLE_BUILTIN_MLFLOW:
            mlflow.log_metric("train_loss", np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4), step=self.current_epoch)
            mlflow.log_metric("train_seg_loss", np.round(self.logger.my_fantastic_logging['train_seg_losses'][-1], decimals=4), step=self.current_epoch)
            mlflow.log_metric("train pseudo dice", np.round(self.logger.my_fantastic_logging['train_mean_fg_dice'][-1], decimals=4), step=self.current_epoch)
            mlflow.log_metric("train pseudo dice mov. avg.", np.round(self.logger.my_fantastic_logging['train_ema_fg_dice'][-1], decimals=4), step=self.current_epoch)
            mlflow.log_metric("val pseudo dice", np.round(self.logger.my_fantastic_logging['mean_fg_dice'][-1], decimals=4), step=self.current_epoch)
            mlflow.log_metric("val pseudo dice  mov. avg.", np.round(self.logger.my_fantastic_logging['ema_fg_dice'][-1], decimals=4), step=self.current_epoch)

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # 追蹤本 epoch 是否觸發 new best EMA（給 metrics CSV 用）
        _prev_best_ema = self._best_ema
        _prev_best_ema_model_dice = self._best_ema_model_dice

        # handle 'best' checkpointing
        # 如果指定了 BEST_VAL_CLASSES，根據指定 class 的 dice 判斷 best；否則用全類別 ema_fg_dice
        best_val_classes = getattr(self, 'BEST_VAL_CLASSES', None)
        if best_val_classes is not None:
            dice_per_class = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
            selected_dice = [dice_per_class[c] for c in best_val_classes if c < len(dice_per_class)]
            best_val_metric = np.nanmean(selected_dice) if len(selected_dice) > 0 else 0.0
            metric_label = f"val Pseudo Dice (classes {best_val_classes})"
        else:
            best_val_metric = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            metric_label = "val EMA pseudo Dice"

        if self._best_ema is None or best_val_metric > self._best_ema:
            self._best_ema = best_val_metric
            self.print_to_log_file(f"Yayy! New best {metric_label}: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        # handle EMA model 'best' checkpointing
        if self.ENABLE_EMA and 'ema_dice_per_class_or_region' in self.logger.my_fantastic_logging and \
                len(self.logger.my_fantastic_logging['ema_dice_per_class_or_region']) > 0:
            if best_val_classes is not None:
                ema_dpc = self.logger.my_fantastic_logging['ema_dice_per_class_or_region'][-1]
                selected = [ema_dpc[c] for c in best_val_classes if c < len(ema_dpc)]
                ema_best_metric = np.nanmean(selected) if len(selected) > 0 else 0.0
            else:
                ema_best_metric = self.logger.my_fantastic_logging['ema_mean_fg_dice'][-1]
            if self._best_ema_model_dice is None or ema_best_metric > self._best_ema_model_dice:
                self._best_ema_model_dice = ema_best_metric
                self.print_to_log_file(f"New best EMA model {metric_label}: {np.round(self._best_ema_model_dice, decimals=4)}")
                self._save_ema_checkpoint(join(self.output_folder, 'checkpoint_best_ema.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)
        
        # Early stopping check
        if self.enable_early_stopping:
            current_val_ema_dice = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            
            # Initialize best metric on first epoch
            if self.early_stopping_best_metric is None:
                self.early_stopping_best_metric = current_val_ema_dice
                self.print_to_log_file(f"Early Stopping: Initialized with val EMA pseudo Dice: {np.round(current_val_ema_dice, decimals=4)}")
            else:
                # Check if there's improvement
                improvement = current_val_ema_dice - self.early_stopping_best_metric
                
                if improvement > self.early_stopping_min_delta:
                    # Significant improvement
                    self.early_stopping_best_metric = current_val_ema_dice
                    self.early_stopping_counter = 0
                    self.print_to_log_file(f"Early Stopping: Improvement detected (+{np.round(improvement, decimals=6)}). Counter reset to 0.")
                else:
                    # No significant improvement
                    self.early_stopping_counter += 1
                    self.print_to_log_file(f"Early Stopping: No improvement (change: {np.round(improvement, decimals=6)}). "
                                         f"Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                    
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        self.should_stop_training = True
                        self.print_to_log_file(f"Early Stopping: Patience ({self.early_stopping_patience}) reached. "
                                             f"Best val EMA pseudo Dice: {np.round(self.early_stopping_best_metric, decimals=4)}",
                                             also_print_to_console=True)
                        self.print_to_log_file(f"Early Stopping: Training will stop after this epoch.", also_print_to_console=True)

        # 計算本 epoch 是否觸發 new best EMA（給 metrics CSV）
        _new_best_this_epoch = None
        if self._best_ema is not None and self._best_ema != _prev_best_ema:
            _new_best_this_epoch = float(self._best_ema)
        if self._best_ema_model_dice is not None and self._best_ema_model_dice != _prev_best_ema_model_dice:
            # 後觸發者覆蓋（與 log parser 行為一致）
            _new_best_this_epoch = float(self._best_ema_model_dice)

        # 寫入 training_metrics.csv（直接記錄關鍵指標，免去解析 training_log.txt）
        try:
            self._write_metrics_csv(new_best_ema=_new_best_this_epoch)
        except Exception as _csv_err:
            self.print_to_log_file(f"[metrics_csv] 寫入失敗（非致命）：{_csv_err}")

        self.current_epoch += 1

    def _write_metrics_csv(self, new_best_ema=None):
        """寫入本 epoch 的 metrics 到 training_metrics.csv。

        - 路徑：self.output_folder/training_metrics.csv
        - resume / continue training 時，相同 epoch 的 row 會被取代（不重複）
        - DDP 時只有 local_rank=0 寫
        - pandas 缺失時自動 skip（非致命）
        """
        if getattr(self, 'local_rank', 0) != 0:
            return
        try:
            import pandas as pd
        except ImportError:
            return

        import os as _os
        csv_path = join(self.output_folder, 'training_metrics.csv')
        log = self.logger.my_fantastic_logging

        def _last_scalar(key):
            if key in log and len(log[key]) > 0:
                v = log[key][-1]
                if isinstance(v, (list, tuple, np.ndarray)):
                    return float(v[0]) if len(v) > 0 else float('nan')
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return float('nan')
            return float('nan')

        def _last_list(key):
            """取最後一次的 per-class list。回傳 [] 表示沒值。"""
            if key in log and len(log[key]) > 0:
                v = log[key][-1]
                if isinstance(v, (list, tuple, np.ndarray)):
                    return [float(x) for x in v]
                try:
                    return [float(v)]
                except (TypeError, ValueError):
                    return []
            return []

        # epoch time
        ts_start = _last_scalar('epoch_start_timestamps')
        ts_end = _last_scalar('epoch_end_timestamps')
        epoch_time_s = ts_end - ts_start if not (np.isnan(ts_start) or np.isnan(ts_end)) else float('nan')

        row = {
            'epoch': int(self.current_epoch),
            'loss_config': getattr(self, 'loss_str', '') or '',
            'lr': _last_scalar('lrs'),
            'train_loss': _last_scalar('train_losses'),
            'train_seg_loss': _last_scalar('train_seg_losses'),
            'val_loss': _last_scalar('val_losses'),
            'val_seg_loss': _last_scalar('val_seg_losses'),
            'train_dice': _last_scalar('train_dice_per_class_or_region'),  # 第一個前景類別（向後相容）
            'val_dice': _last_scalar('dice_per_class_or_region'),
            'ema_val_loss': _last_scalar('ema_val_losses'),
            'ema_val_dice': _last_scalar('ema_mean_fg_dice'),
            'epoch_time_s': epoch_time_s,
            'new_best_ema': float(new_best_ema) if new_best_ema is not None else float('nan'),
        }
        # per-component loss 數值（compound loss 才有，例如 train_loss_Tversky_and_CE_loss）
        for k in log.keys():
            if k.startswith('train_loss_') or k.startswith('val_loss_'):
                row[k] = _last_scalar(k)

        # 每個前景類別獨立的 dice 欄位（multi-class 分析用，class 數量由 dataset 決定）
        for i, d in enumerate(_last_list('train_dice_per_class_or_region')):
            row[f'train_dice_class_{i}'] = d
        for i, d in enumerate(_last_list('dice_per_class_or_region')):
            row[f'val_dice_class_{i}'] = d
        for i, d in enumerate(_last_list('ema_dice_per_class_or_region')):
            row[f'ema_val_dice_class_{i}'] = d
        if getattr(self, 'has_cls_head', False):
            row['train_cls_loss'] = _last_scalar('train_cls_losses')
            row['train_accuracy'] = _last_scalar('train_accuracys')
            row['train_sensitivity'] = _last_scalar('train_sensitivitys')
            row['train_specificity'] = _last_scalar('train_specificitys')
            row['val_cls_loss'] = _last_scalar('val_cls_losses')
            row['val_accuracy'] = _last_scalar('val_accuracys')
            row['val_sensitivity'] = _last_scalar('val_sensitivitys')
            row['val_specificity'] = _last_scalar('val_specificitys')

        # Replace existing row for this epoch (idempotent on resume)
        if _os.path.exists(csv_path):
            try:
                existing = pd.read_csv(csv_path)
                if 'epoch' in existing.columns:
                    existing = existing[existing['epoch'] != self.current_epoch]
            except Exception:
                existing = pd.DataFrame()
        else:
            existing = pd.DataFrame()

        combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        # 確保 epoch 升冪排序
        if 'epoch' in combined.columns:
            combined = combined.sort_values('epoch').reset_index(drop=True)
        combined.to_csv(csv_path, index=False)

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                checkpoint = {
                    'network_weights': self.network.module.state_dict() if self.is_ddp else self.network.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    '_best_ema_model_dice': self._best_ema_model_dice,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                # 保存 EMA model 參數（用於 resume training）
                if self.ema_model is not None:
                    checkpoint['ema_model_weights'] = self.ema_model.state_dict()
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def _save_ema_checkpoint(self, filename: str) -> None:
        """單獨保存 EMA model 的 checkpoint（用於部署）。格式與正常 checkpoint 相同，但 network_weights 是 EMA 的。"""
        if self.local_rank == 0 and not self.disable_checkpointing and self.ema_model is not None:
            checkpoint = {
                'network_weights': self.ema_model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                'logging': self.logger.get_checkpoint(),
                '_best_ema': self._best_ema,
                '_best_ema_model_dice': self._best_ema_model_dice,
                'current_epoch': self.current_epoch + 1,
                'init_args': self.my_init_kwargs,
                'trainer_name': self.__class__.__name__,
                'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
            }
            torch.save(checkpoint, filename)

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self._best_ema_model_dice = checkpoint.get('_best_ema_model_dice', None)
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        if self.is_ddp:
            self.network.module.load_state_dict(new_state_dict)
        else:
            self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

        # 恢復 EMA model 參數
        if self.ema_model is not None and 'ema_model_weights' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_model_weights'])
            self.print_to_log_file("EMA model weights restored from checkpoint.")

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        num_seg_heads = self.label_manager.num_segmentation_heads

        inference_gaussian = torch.from_numpy(
            compute_gaussian(self.configuration_manager.patch_size, sigma_scale=1. / 8))
        # spawn allows the use of GPU in the background process in case somebody wants to do this. Not recommended. Trust me.
        # segmentation_export_pool = multiprocessing.get_context('spawn').Pool(default_num_processes)
        # let's not use this until someone really needs it!
        # segmentation_export_pool = multiprocessing.Pool(default_num_processes)
        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []
            for k in dataset_val.keys():
                proceed = not check_workers_busy(segmentation_export_pool, results,
                                                 allowed_num_queued=len(segmentation_export_pool._pool))
                while not proceed:
                    sleep(1)
                    proceed = not check_workers_busy(segmentation_export_pool, results,
                                                     allowed_num_queued=len(segmentation_export_pool._pool))

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))

                output_filename_truncated = join(validation_output_folder, k)

                try:
                    prediction = predict_sliding_window_return_logits(self.network, data, num_seg_heads,
                                                                      tile_size=self.configuration_manager.patch_size,
                                                                      mirror_axes=self.inference_allowed_mirroring_axes,
                                                                      tile_step_size=0.5,
                                                                      use_gaussian=True,
                                                                      precomputed_gaussian=inference_gaussian,
                                                                      perform_everything_on_gpu=True,
                                                                      verbose=False,
                                                                      device=self.device).cpu().numpy()
                except RuntimeError:
                    prediction = predict_sliding_window_return_logits(self.network, data, num_seg_heads,
                                                                      tile_size=self.configuration_manager.patch_size,
                                                                      mirror_axes=self.inference_allowed_mirroring_axes,
                                                                      tile_step_size=0.5,
                                                                      use_gaussian=True,
                                                                      precomputed_gaussian=inference_gaussian,
                                                                      perform_everything_on_gpu=False,
                                                                      verbose=False,
                                                                      device=self.device).cpu().numpy()

                if should_i_save_to_file(prediction, results, segmentation_export_pool):
                    np.save(output_filename_truncated + '.npy', prediction)
                    prediction_for_export = output_filename_truncated + '.npy'
                else:
                    prediction_for_export = prediction

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_softmax, (
                            (prediction_for_export, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )
                # for debug purposes
                # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                #              output_filename_truncated, save_probabilities)

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = nnUNetDataset(expected_preprocessed_folder, [k],
                                                num_images_properties_loading_threshold=0)
                            d, s, p = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file = join(output_folder, k + '.npz')

                        if should_i_save_to_file(prediction, results, segmentation_export_pool):
                            np.save(output_file[:-4] + '.npy', prediction)
                            prediction_for_export = output_file[:-4] + '.npy'
                        else:
                            prediction_for_export = prediction
                        # resample_and_save(prediction, target_shape, output_file, self.plans, self.configuration, properties,
                        #                   self.dataset_json, n)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction_for_export, target_shape, output_file, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json, n),
                            )
                        ))

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]), also_print_to_console=True)

        self.set_deep_supervision_enabled(True)

    def run_training(self):
        from contextlib import nullcontext

        # 根據程式所在的資料夾名稱動態設定 run_name
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 往上三層找到主要的專案資料夾名稱 (nnResUNet-long-BigBatch-cosine-1to1-testspeed)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        run_name = os.path.basename(project_root)
        print(run_name)
        print('self.num_iterations_per_epoch:', self.num_iterations_per_epoch)
        print('learning_rate:', self.initial_lr)
        print('num_epochs:', self.num_epochs)
        print('oversample_foreground_percent:', self.oversample_foreground_percent)
        print('oversample_foreground_percent_val:', self.oversample_foreground_percent_val)
        print('optimizer_type:', self.optimizer_type)
        print('lr_scheduler_type:', self.lr_scheduler_type)
        print('enable_early_stopping:', self.enable_early_stopping)
        print('early_stopping_patience:', self.early_stopping_patience)
        print('early_stopping_min_delta:', self.early_stopping_min_delta)

        # 當 MUTP 呼叫時，跳過 Trainer 內建的 MLflow（由 MUTP watcher thread 追蹤）
        if self.DISABLE_BUILTIN_MLFLOW:
            mlflow_ctx = nullcontext()
        else:
            mlflow.pytorch.autolog()
            experiment = mlflow.get_experiment_by_name(run_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(run_name)
            else:
                experiment_id = experiment.experiment_id
            mlflow_ctx = mlflow.start_run(run_name=run_name, experiment_id=experiment_id)

        with mlflow_ctx:
            self.on_train_start()
            if not self.DISABLE_BUILTIN_MLFLOW:
                ml_params = {
                    "epochs": self.num_epochs,
                    "learning_rate": self.initial_lr,
                    "batch_size": self.configuration_manager.batch_size,
                    "oversample_foreground_percent": self.oversample_foreground_percent
                }
                mlflow.log_params(ml_params)
            for epoch in range(self.current_epoch, self.num_epochs):
                self.on_epoch_start() #只有一個訊息而已

                self.on_train_epoch_start() #讀取網路跟進行 lr_scheduler 的控制
                train_outputs = []
                AVE_Queue = []
                start = time()
                for batch_id in range(self.num_iterations_per_epoch):
                    train_outputs.append(self.train_step(next(self.dataloader_train)))
                    # EMA 參數更新（每個 train step 後）
                    if self.ENABLE_EMA:
                        self._update_ema()
                    #print(f"[Queue] size: {self.dataloader_train._queue.qsize()}, maxsize: {self.dataloader_train._queue._maxsize}")
                    AVE_Queue.append(self.dataloader_train._queue.qsize()/self.dataloader_train._queue._maxsize)
                print(f"Data training time: {time() - start:.4f} s") #算出訓練時間
                print('AVE_Queue:', np.array(AVE_Queue).mean()) #算出暫存資料的平均時間
                self.on_train_epoch_end(train_outputs)

                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = []
                    start = time()
                    for batch_id in range(self.num_val_iterations_per_epoch):
                        val_outputs.append(self.validation_step(next(self.dataloader_val)))
                    print(f"Data valing time: {time() - start:.4f} s") #算出驗證時間
                    self.on_validation_epoch_end(val_outputs)

                    # EMA validation：用 EMA 參數跑一次驗證
                    if self.ENABLE_EMA and self.ema_model is not None:
                        ema_val_outputs = []
                        start = time()
                        # 暫時把 EMA 參數換入 network
                        source_net = self.network.module if self.is_ddp else self.network
                        # 備份原始參數
                        orig_state = {k: v.clone() for k, v in source_net.state_dict().items()}
                        # 載入 EMA 參數
                        source_net.load_state_dict(self.ema_model.state_dict())
                        for batch_id in range(self.num_val_iterations_per_epoch):
                            ema_val_outputs.append(self.validation_step(next(self.dataloader_val)))
                        # 還原原始參數
                        source_net.load_state_dict(orig_state)
                        del orig_state
                        print(f"EMA valing time: {time() - start:.4f} s")
                        self._on_ema_validation_epoch_end(ema_val_outputs)

                self.on_epoch_end()

                # Check if early stopping is triggered
                if self.should_stop_training:
                    self.print_to_log_file(f"Early Stopping: Training stopped at epoch {self.current_epoch}", also_print_to_console=True)
                    break

        self.on_train_end()

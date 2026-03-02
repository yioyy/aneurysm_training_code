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
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss, Log_DC_loss, CE_loss, DC_loss
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
        initial_num_deep_supervision_levels = len(self._get_deep_supervision_scales())
        self.logger = nnUNetLogger(verbose=False, num_deep_supervision_levels=initial_num_deep_supervision_levels)

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start

        ### initializing stuff for remembering things and such
        self._best_ema = None

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

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                           self.configuration_manager,
                                                           self.num_input_channels,
                                                           enable_deep_supervision=True).to(self.device)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.ce_loss = self._build_ce_loss()
            self.dice_loss = self._build_dice_loss() #正確賦值
            
            # Dynamically create individual dice losses for each deep supervision level
            self.num_deep_supervision_levels = len(self._get_deep_supervision_scales())
            self.individual_dice_losses = {}
            for i in range(self.num_deep_supervision_levels):
                dice_loss_name = f'dice_loss{i}'
                self.individual_dice_losses[dice_loss_name] = self._build_individual_dice_loss(i)
                # Also set as attribute for backward compatibility
                setattr(self, dice_loss_name, self.individual_dice_losses[dice_loss_name])
            
            # Reinitialize logger with correct number of deep supervision levels
            self.logger = nnUNetLogger(verbose=self.logger.verbose, num_deep_supervision_levels=self.num_deep_supervision_levels)
            
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
        return get_network_from_plans(plans_manager, dataset_json, configuration_manager,
                                      num_input_channels, deep_supervision=enable_deep_supervision)

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

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
            print('Loss: DC_and_BCE_loss')
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=NewSoftDiceLoss)
            print('nnUNet-long-BigBatch-cosine')
            print('Loss: DC_and_CE_loss')
            print('DC_and_CE_loss => NewDC_loss_and_CE_loss')   
            print('combnine FEMH!!! and Large Big Batch Size 900!!!')   
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
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
            print('Loss: DC_and_BCE_loss => 會做到這裡等於錯!!!')
        else:
            ce_loss = CE_loss({}, weight_ce=1, ignore_label=self.label_manager.ignore_label)            
            print('Loss: CE_loss')        

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        ce_loss = DeepSupervisionWrapper(ce_loss, weights)
        return ce_loss
    
    def _build_dice_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
            print('Loss: DC_and_BCE_loss => 會做到這裡等於錯!!!')
        else:            
            dice_loss = DC_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                  ignore_label=self.label_manager.ignore_label, dice_class=NewSoftDiceLoss)
            print('Loss: DC_loss')             

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        dice_loss = DeepSupervisionWrapper(dice_loss, weights)
        return dice_loss
    
    def _build_individual_dice_loss(self, target_level):
        """
        Build dice loss for a specific deep supervision level
        Args:
            target_level: which level to focus on (0-based index)
        """
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
        dataset_tr = nnUNetDataset(self.preprocessed_dataset_folder, tr_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0)
        dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0)
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

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data=3, order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
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
        category_weights = getattr(self.__class__, "SAMPLING_CATEGORY_WEIGHTS", None)
        sampling_weight_mode = getattr(self.__class__, "SAMPLING_CATEGORY_WEIGHT_MODE", "multiplier")
        sampling_probabilities = build_sampling_probabilities(
            tr_keys,
            sampling_categories=sampling_categories,
            category_weights=category_weights,
            mode=sampling_weight_mode,
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
            # 依類別 1~4 排序輸出，未出現的類別為 0
            category_ids = sorted(set(counts.keys()) | {1, 2, 3, 4})
            lines = ["Sampling category counts (training set):", "=" * 40]

            # 額外打印你設定的採樣比例（weights），例如 2:1:1:1
            weight_ids = [1, 2, 3, 4]
            weight_vals = [float(category_weights.get(c, 1.0)) if category_weights else 1.0 for c in weight_ids]
            ratio_str = ":".join(str(int(w)) if abs(w - int(w)) < 1e-9 else ("%g" % w) for w in weight_vals)
            w_sum = sum(weight_vals)
            if w_sum > 0:
                target_pct = ":".join("%.2f%%" % (w / w_sum * 100.0) for w in weight_vals)
            else:
                target_pct = ":".join("0.00%%" for _ in weight_vals)
            lines.append("Configured sampling ratio (Category 1-4 weights): %s" % ratio_str)
            lines.append("Configured sampling target proportions (normalized): %s" % target_pct)
            lines.append("Sampling category weight mode: %s" % sampling_weight_mode)

            # 校正後（依實際 sampling_probabilities 反推）的期望類別抽樣比例
            if sampling_probabilities is not None:
                expected_mass = {}
                for k, p in zip(tr_keys, sampling_probabilities):
                    c = sampling_categories.get(k, 0)
                    expected_mass[c] = expected_mass.get(c, 0.0) + float(p)
                expected_ids = sorted(set(expected_mass.keys()) | {1, 2, 3, 4})
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

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=sampling_probabilities, pad_sides=None)
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent_val,
                                        sampling_probabilities=None, pad_sides=None)
        else:
            dl_tr = nnUNetDataLoader3D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=sampling_probabilities, pad_sides=None)
            dl_val = nnUNetDataLoader3D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent_val,
                                        sampling_probabilities=None, pad_sides=None)
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
                                ignore_label: int = None) -> AbstractTransform:
        tr_transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=None,
            do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
            do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
            p_rot_per_axis=1,  # todo experiment with this
            do_scale=True, scale=(0.7, 1.4),
            border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
            random_crop=False,  # random cropping is part of our dataloaders
            p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False  # todo experiment with this
        ))

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        # tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
        #                                                     p_per_channel=0.5,
        #                                                     order_downsample=0, order_upsample=3, p_per_sample=0.25,
        #                                                     ignore_axes=ignore_axes))
        tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

        if mirror_axes is not None and len(mirror_axes) > 0:
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

        self.optimizer.zero_grad()
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)
            ce_l = self.ce_loss(output, target)
            dice_l = self.dice_loss(output, target)
            
            # Dynamically calculate individual dice losses
            individual_dice_losses = {}
            for i in range(self.num_deep_supervision_levels):
                dice_loss_name = f'dice_l{i}'
                individual_dice_losses[dice_loss_name] = self.individual_dice_losses[f'dice_loss{i}'](output, target)
                # Also set as local variable for backward compatibility
                locals()[dice_loss_name] = individual_dice_losses[dice_loss_name]

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()


        #print('output.shape:', output[0].shape) #=> torch.Size([1200, 2])
        #這裡先把label跟pred轉成one hot，這邊多做一個原本的Pseudo dice
        # we only need the output with the highest output resolution
        # 先把會輸出的log的值定義成0，後續繪圖就不會有問題
        metrics = {}
        # Dynamically initialize dc metrics for each deep supervision level
        for i in range(self.num_deep_supervision_levels):
            metrics[f'dc{i}'] = 0
        metrics['fake_dcl'] = 0

        #這邊改用for迴圈去計算跟填值
        weights = np.array([1 / (2 ** i) for i in range(len(output))])
        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        #print('train weights!!!:', weights)
        weights = weights / weights.sum()
        #print('train weights!!!:', weights)

        fake_dcl_all = []
        
        for i in range(len(output)):
            output0 = output[i]
            target0 = target[i]

            if i == 0:
                axes = [0] + list(range(2, len(output0.shape)))
                axes0 = list(range(2, len(output0.shape)))

                if self.label_manager.has_regions:
                    predicted_segmentation_onehot0 = (torch.sigmoid(output0) > 0.5).long()
                else:
                    #no need for softmax
                    output_seg = output0.argmax(1)[:, None]
                    predicted_segmentation_onehot0 = torch.zeros(output0.shape, device=output0.device, dtype=torch.float32)
                    predicted_segmentation_onehot0.scatter_(1, output_seg, 1)
                    del output_seg

                if self.label_manager.has_ignore_label:
                    if not self.label_manager.has_regions:
                        mask0 = (target0 != self.label_manager.ignore_label).float()
                        # CAREFUL that you don't rely on target after this line!
                        target0[target0 == self.label_manager.ignore_label] = 0
                    else:
                        mask0 = 1 - target0[:, -1:]
                        # CAREFUL that you don't rely on target after this line!
                        target0 = target0[:, :-1]
                else:
                    mask0 = None

                tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot0, target0, axes=axes, mask=mask0)
                tp0, fp0, fn0, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot0, target0, axes=axes0, mask=mask0)

                tp_hard = tp.detach()
                fp_hard = fp.detach()
                fn_hard = fn.detach()

                tp_hard0 = tp0.detach()
                fp_hard0 = fp0.detach()
                fn_hard0 = fn0.detach()

                #3層的dice 跟 fake dice loss能到這邊算好惹
                smooth = 1e-5
                nominator = 2 * tp_hard0
                denominator = 2 * tp_hard0 + fp_hard0 + fn_hard0
                dc0 = (nominator + smooth) / (torch.clip(denominator + smooth, 1e-8))

                if not self.label_manager.has_regions:
                    tp_hard = tp_hard[1:]
                    fp_hard = fp_hard[1:]
                    fn_hard = fn_hard[1:]

                    dc0 = dc0[:,1:].mean()
                
                #更新字典
                metrics['dc' + str(i)] = dc0.cpu().numpy()
                fake_dcl_all.append(dc0.cpu().numpy())

            else:
                axes0 = list(range(2, len(output0.shape)))
                if self.label_manager.has_regions:
                    predicted_segmentation_onehot0 = (torch.sigmoid(output0) > 0.5).long()
                else:
                    #no need for softmax
                    output_seg = output0.argmax(1)[:, None]
                    predicted_segmentation_onehot0 = torch.zeros(output0.shape, device=output0.device, dtype=torch.float32)
                    predicted_segmentation_onehot0.scatter_(1, output_seg, 1)
                    del output_seg

                if self.label_manager.has_ignore_label:
                    if not self.label_manager.has_regions:
                        mask0 = (target0 != self.label_manager.ignore_label).float()
                        # CAREFUL that you don't rely on target after this line!
                        target0[target0 == self.label_manager.ignore_label] = 0
                    else:
                        mask0 = 1 - target0[:, -1:]
                        # CAREFUL that you don't rely on target after this line!
                        target0 = target0[:, :-1]
                else:
                    mask0 = None
                
                tp0, fp0, fn0, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot0, target0, axes=axes0, mask=mask0)

                tp_hard0 = tp0.detach()
                fp_hard0 = fp0.detach()
                fn_hard0 = fn0.detach()

                #3層的dice 跟 fake dice loss能到這邊算好惹
                smooth = 1e-5
                nominator = 2 * tp_hard0
                denominator = 2 * tp_hard0 + fp_hard0 + fn_hard0
                dc0 = (nominator + smooth) / (torch.clip(denominator + smooth, 1e-8))

                if not self.label_manager.has_regions:
                    dc0 = dc0[:,1:].mean()

                #更新字典
                metrics['dc' + str(i)] = dc0.cpu().numpy()    
                fake_dcl_all.append(dc0.cpu().numpy())

        #weights
        fake_dcl = 1 - np.sum(np.array(fake_dcl_all)*weights)

        # print('dice_loss:', dice_l)
        # print('fake_dcl:', fake_dcl)
        # print('dc0:', metrics['dc0'])

        
        #fake_dcl = 1 - (dc0*0.53333333 + dc1*0.26666667 + dc2*0.13333333 + dc3*0.06666667) #應該要等效 (1-dc0)*0.533..
        #print('dc1:', dc1.shape) #torch.Size([1200, 1])

        # return {'loss': l.detach().cpu().numpy()}
        # return {'loss': l.detach().cpu().numpy(), 'ce_loss': ce_l.detach().cpu().numpy(), 'dice_loss': dice_l.detach().cpu().numpy(), 'tp_hard': tp_hard.cpu().numpy(), 'fp_hard': fp_hard.cpu().numpy(), 'fn_hard': fn_hard.cpu().numpy(),
        #         'tp_hard1': tp_hard1.cpu().numpy(), 'fp_hard1': fp_hard1.cpu().numpy(), 'fn_hard1': fn_hard1.cpu().numpy(),
        #         'tp_hard2': tp_hard2.cpu().numpy(), 'fp_hard2': fp_hard2.cpu().numpy(), 'fn_hard2': fn_hard2.cpu().numpy(),
        #         'tp_hard3': tp_hard3.cpu().numpy(), 'fp_hard3': fp_hard3.cpu().numpy(), 'fn_hard3': fn_hard3.cpu().numpy()
        #         }
        return {'loss': l.detach().cpu().numpy(), 'ce_loss': ce_l.detach().cpu().numpy(), 'dice_loss': dice_l.detach().cpu().numpy(), 
                'tp_hard': tp_hard.cpu().numpy(), 'fp_hard': fp_hard.cpu().numpy(), 'fn_hard': fn_hard.cpu().numpy(),
                **{f'dice_loss{i}': individual_dice_losses[f'dice_l{i}'].detach().cpu().numpy() for i in range(self.num_deep_supervision_levels)},
                **{f'dc{i}': metrics[f'dc{i}'] for i in range(self.num_deep_supervision_levels)}, 'fake_dcl': fake_dcl
                }

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)
        tp = np.sum(outputs['tp_hard'], 0)
        fp = np.sum(outputs['fp_hard'], 0)
        fn = np.sum(outputs['fn_hard'], 0)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()

            ce_losses = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(ce_losses, outputs['ce_loss'])
            ce_loss_here = np.vstack(ce_losses).mean()            

            dice_losses = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(dice_losses, outputs['dice_loss'])
            dice_loss_here = np.vstack(dice_losses).mean()  

            # Dynamically handle individual dice losses and dc metrics for DDP
            individual_dice_loss_here = {}
            individual_dice_here = {}
            
            for i in range(self.num_deep_supervision_levels):
                # Handle dice losses
                dice_loss_key = f'dice_loss{i}'
                dice_losses = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(dice_losses, outputs[dice_loss_key])
                individual_dice_loss_here[f'dice_loss{i}_here'] = np.vstack(dice_losses).mean()
                
                # Handle dc metrics
                dc_key = f'dc{i}'
                dcs = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(dcs, outputs[dc_key])
                individual_dice_here[f'dice{i}_here'] = np.vstack(dcs).mean() 

            fake_dc_losses = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(fake_dc_losses, outputs['fake_dcl'])
            fake_dice_loss_here = np.vstack(fake_dc_losses).mean() 

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
        else:
            loss_here = np.mean(outputs['loss'])
            ce_loss_here = np.mean(outputs['ce_loss'])
            dice_loss_here = np.mean(outputs['dice_loss'])
            fake_dice_loss_here = np.mean(outputs['fake_dcl'])
            
            # Dynamically handle individual dice losses and dc metrics for non-DDP
            individual_dice_loss_here = {}
            individual_dice_here = {}
            
            for i in range(self.num_deep_supervision_levels):
                individual_dice_here[f'dice{i}_here'] = np.mean(outputs[f'dc{i}'])
                individual_dice_loss_here[f'dice_loss{i}_here'] = np.mean(outputs[f'dice_loss{i}'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(tp, fp, fn)]] 

        mean_fg_dice = np.nanmean(global_dc_per_class) 

        self.logger.log('train_losses', loss_here, self.current_epoch)
        self.logger.log('train_ce_losses', ce_loss_here, self.current_epoch)
        self.logger.log('train_dice_losses', dice_loss_here, self.current_epoch)
        # Dynamically log individual dice losses and supervision dice
        for i in range(self.num_deep_supervision_levels):
            self.logger.log(f'train_dice_loss{i}', individual_dice_loss_here[f'dice_loss{i}_here'], self.current_epoch)
            self.logger.log(f'train_supervision_dice{i}', individual_dice_here[f'dice{i}_here'], self.current_epoch)
        
        self.logger.log('train_mean_fg_dice', mean_fg_dice, self.current_epoch)        
        self.logger.log('train_dice_per_class_or_region', global_dc_per_class, self.current_epoch)  
        self.logger.log('train_fake_dice_losses', fake_dice_loss_here, self.current_epoch)   

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

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)
            ce_l = self.ce_loss(output, target)
            dice_l = self.dice_loss(output, target)
            
            # Dynamically calculate individual dice losses
            individual_dice_losses = {}
            for i in range(self.num_deep_supervision_levels):
                dice_loss_name = f'dice_l{i}'
                individual_dice_losses[dice_loss_name] = self.individual_dice_losses[f'dice_loss{i}'](output, target)
                # Also set as local variable for backward compatibility
                locals()[dice_loss_name] = individual_dice_losses[dice_loss_name]

        #這裡先把label跟pred轉成one hot，這邊多做一個原本的Pseudo dice
        # we only need the output with the highest output resolution
        # 先把會輸出的log的值定義成0，後續繪圖就不會有問題
        metrics = {}
        # Dynamically initialize dc metrics for each deep supervision level
        for i in range(self.num_deep_supervision_levels):
            metrics[f'dc{i}'] = 0
        metrics['fake_dcl'] = 0

        #這邊改用for迴圈去計算跟填值
        weights = np.array([1 / (2 ** i) for i in range(len(output))])
        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        #print('train weights!!!:', weights)
        weights = weights / weights.sum()
        #print('train weights!!!:', weights)

        fake_dcl_all = []
        
        for i in range(len(output)):
            output0 = output[i]
            target0 = target[i]

            if i == 0:
                axes = [0] + list(range(2, len(output0.shape)))
                axes0 = list(range(2, len(output0.shape)))

                if self.label_manager.has_regions:
                    predicted_segmentation_onehot0 = (torch.sigmoid(output0) > 0.5).long()
                else:
                    #no need for softmax
                    output_seg = output0.argmax(1)[:, None]
                    predicted_segmentation_onehot0 = torch.zeros(output0.shape, device=output0.device, dtype=torch.float32)
                    predicted_segmentation_onehot0.scatter_(1, output_seg, 1)
                    del output_seg

                if self.label_manager.has_ignore_label:
                    if not self.label_manager.has_regions:
                        mask0 = (target0 != self.label_manager.ignore_label).float()
                        # CAREFUL that you don't rely on target after this line!
                        target0[target0 == self.label_manager.ignore_label] = 0
                    else:
                        mask0 = 1 - target0[:, -1:]
                        # CAREFUL that you don't rely on target after this line!
                        target0 = target0[:, :-1]
                else:
                    mask0 = None

                tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot0, target0, axes=axes, mask=mask0)
                tp0, fp0, fn0, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot0, target0, axes=axes0, mask=mask0)

                tp_hard = tp.detach()
                fp_hard = fp.detach()
                fn_hard = fn.detach()

                tp_hard0 = tp0.detach()
                fp_hard0 = fp0.detach()
                fn_hard0 = fn0.detach()

                #3層的dice 跟 fake dice loss能到這邊算好惹
                smooth = 1e-5
                nominator = 2 * tp_hard0
                denominator = 2 * tp_hard0 + fp_hard0 + fn_hard0
                dc0 = (nominator + smooth) / (torch.clip(denominator + smooth, 1e-8))

                if not self.label_manager.has_regions:
                    tp_hard = tp_hard[1:]
                    fp_hard = fp_hard[1:]
                    fn_hard = fn_hard[1:]

                    dc0 = dc0[:,1:].mean()
                
                #更新字典
                metrics['dc' + str(i)] = dc0.cpu().numpy()
                fake_dcl_all.append(dc0.cpu().numpy())

            else:
                axes0 = list(range(2, len(output0.shape)))
                if self.label_manager.has_regions:
                    predicted_segmentation_onehot0 = (torch.sigmoid(output0) > 0.5).long()
                else:
                    #no need for softmax
                    output_seg = output0.argmax(1)[:, None]
                    predicted_segmentation_onehot0 = torch.zeros(output0.shape, device=output0.device, dtype=torch.float32)
                    predicted_segmentation_onehot0.scatter_(1, output_seg, 1)
                    del output_seg

                if self.label_manager.has_ignore_label:
                    if not self.label_manager.has_regions:
                        mask0 = (target0 != self.label_manager.ignore_label).float()
                        # CAREFUL that you don't rely on target after this line!
                        target0[target0 == self.label_manager.ignore_label] = 0
                    else:
                        mask0 = 1 - target0[:, -1:]
                        # CAREFUL that you don't rely on target after this line!
                        target0 = target0[:, :-1]
                else:
                    mask0 = None
                
                tp0, fp0, fn0, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot0, target0, axes=axes0, mask=mask0)

                tp_hard0 = tp0.detach()
                fp_hard0 = fp0.detach()
                fn_hard0 = fn0.detach()

                #3層的dice 跟 fake dice loss能到這邊算好惹
                smooth = 1e-5
                nominator = 2 * tp_hard0
                denominator = 2 * tp_hard0 + fp_hard0 + fn_hard0
                dc0 = (nominator + smooth) / (torch.clip(denominator + smooth, 1e-8))

                if not self.label_manager.has_regions:
                    dc0 = dc0[:,1:].mean()

                #更新字典
                metrics['dc' + str(i)] = dc0.cpu().numpy()    
                fake_dcl_all.append(dc0.cpu().numpy())

        #weights
        fake_dcl = 1 - np.sum(np.array(fake_dcl_all)*weights)

        # return {'loss': l.detach().cpu().numpy(), 'ce_loss': ce_l.detach().cpu().numpy(), 'dice_loss': dice_l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard,
        #         'tp_hard1': tp_hard1, 'fp_hard1': fp_hard1, 'fn_hard1': fn_hard1,
        #         'tp_hard2': tp_hard2, 'fp_hard2': fp_hard2, 'fn_hard2': fn_hard2,
        #         'tp_hard3': tp_hard3, 'fp_hard3': fp_hard3, 'fn_hard3': fn_hard3
        #         }
        return {'loss': l.detach().cpu().numpy(), 'ce_loss': ce_l.detach().cpu().numpy(), 'dice_loss': dice_l.detach().cpu().numpy(), 
                'tp_hard': tp_hard.cpu().numpy(), 'fp_hard': fp_hard.cpu().numpy(), 'fn_hard': fn_hard.cpu().numpy(),
                **{f'dice_loss{i}': individual_dice_losses[f'dice_l{i}'].detach().cpu().numpy() for i in range(self.num_deep_supervision_levels)},
                **{f'dc{i}': metrics[f'dc{i}'] for i in range(self.num_deep_supervision_levels)}, 'fake_dcl': fake_dcl
                }

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

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()

            ce_losses = [None for _ in range(world_size)]
            dist.all_gather_object(ce_losses, outputs_collated['ce_loss'])
            ce_loss_here = np.vstack(ce_losses).mean()  

            dice_losses = [None for _ in range(world_size)]
            dist.all_gather_object(dice_losses, outputs_collated['dice_loss'])
            dice_loss_here = np.vstack(dice_losses).mean()  

            # Dynamically handle individual dice losses and dc metrics for validation DDP
            val_individual_dice_loss_here = {}
            val_individual_dice_here = {}
            
            for i in range(self.num_deep_supervision_levels):
                # Handle dice losses
                dice_loss_key = f'dice_loss{i}'
                dice_losses = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(dice_losses, outputs_collated[dice_loss_key])
                val_individual_dice_loss_here[f'dice_loss{i}_here'] = np.vstack(dice_losses).mean()
                
                # Handle dc metrics
                dc_key = f'dc{i}'
                dcs = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(dcs, outputs_collated[dc_key])
                val_individual_dice_here[f'dice{i}_here'] = np.vstack(dcs).mean() 

            fake_dc_losses = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(fake_dc_losses, outputs_collated['fake_dcl'])
            fake_dice_loss_here = np.vstack(fake_dc_losses).mean() 
        else:
            loss_here = np.mean(outputs_collated['loss'])
            ce_loss_here = np.mean(outputs_collated['ce_loss'])
            dice_loss_here = np.mean(outputs_collated['dice_loss'])
            # Dynamically handle individual dice losses and dc metrics for validation non-DDP
            val_individual_dice_loss_here = {}
            val_individual_dice_here = {}
            
            for i in range(self.num_deep_supervision_levels):
                val_individual_dice_loss_here[f'dice_loss{i}_here'] = np.mean(outputs_collated[f'dice_loss{i}'])
                val_individual_dice_here[f'dice{i}_here'] = np.mean(outputs_collated[f'dc{i}'])
            fake_dice_loss_here = np.mean(outputs_collated['fake_dcl'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(tp, fp, fn)]]    

        mean_fg_dice = np.nanmean(global_dc_per_class)

        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch) 
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('val_ce_losses', ce_loss_here, self.current_epoch)
        self.logger.log('val_dice_losses', dice_loss_here, self.current_epoch)
        # Dynamically log validation individual dice losses and supervision dice
        for i in range(self.num_deep_supervision_levels):
            self.logger.log(f'val_dice_loss{i}', val_individual_dice_loss_here[f'dice_loss{i}_here'], self.current_epoch)
            self.logger.log(f'val_supervision_dice{i}', val_individual_dice_here[f'dice{i}_here'], self.current_epoch)  
        self.logger.log('val_fake_dice_losses', fake_dice_loss_here, self.current_epoch)  

    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # todo find a solution for this stupid shit
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('train_ce_loss', np.round(self.logger.my_fantastic_logging['train_ce_losses'][-1], decimals=4))
        self.print_to_log_file('train_dice_loss', np.round(self.logger.my_fantastic_logging['train_dice_losses'][-1], decimals=4))
        # Dynamically print train dice losses
        for i in range(self.num_deep_supervision_levels):
            self.print_to_log_file(f'train_dice_loss{i}', np.round(self.logger.my_fantastic_logging[f'train_dice_loss{i}'][-1], decimals=4))
        self.print_to_log_file('train_fake_dice_loss', np.round(self.logger.my_fantastic_logging['train_fake_dice_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('val_ce_loss', np.round(self.logger.my_fantastic_logging['val_ce_losses'][-1], decimals=4))
        self.print_to_log_file('val_dice_loss', np.round(self.logger.my_fantastic_logging['val_dice_losses'][-1], decimals=4))
        # Dynamically print validation dice losses
        for i in range(self.num_deep_supervision_levels):
            self.print_to_log_file(f'val_dice_loss{i}', np.round(self.logger.my_fantastic_logging[f'val_dice_loss{i}'][-1], decimals=4))
        self.print_to_log_file('val_fake_dice_loss', np.round(self.logger.my_fantastic_logging['val_fake_dice_losses'][-1], decimals=4))
        self.print_to_log_file('train Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['train_dice_per_class_or_region'][-1]]) 
        # Dynamically print train deep supervision dice
        for i in range(self.num_deep_supervision_levels):
            self.print_to_log_file(f'train deep_supervision dice{i}', np.round(self.logger.my_fantastic_logging[f'train_supervision_dice{i}'][-1], decimals=4))   
        self.print_to_log_file('val Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        # Dynamically print validation deep supervision dice
        for i in range(self.num_deep_supervision_levels):
            self.print_to_log_file(f'val deep_supervision dice{i}', np.round(self.logger.my_fantastic_logging[f'val_supervision_dice{i}'][-1], decimals=4))   
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")
        
        mlflow.log_metric("train_loss", np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4), step=self.current_epoch) # metric 是可以持續被更新的.
        mlflow.log_metric("train_ce_loss", np.round(self.logger.my_fantastic_logging['train_ce_losses'][-1], decimals=4), step=self.current_epoch) # metric 是可以持續被更新的.
        mlflow.log_metric("train_dice_loss", np.round(self.logger.my_fantastic_logging['train_dice_losses'][-1], decimals=4), step=self.current_epoch) # metric 是可以持續被更新的.
        # Dynamically log train dice losses to mlflow
        for i in range(self.num_deep_supervision_levels):
            mlflow.log_metric(f"train_dice_loss{i}", np.round(self.logger.my_fantastic_logging[f'train_dice_loss{i}'][-1], decimals=4), step=self.current_epoch)
        mlflow.log_metric("train_fake_dice_loss", np.round(self.logger.my_fantastic_logging['train_fake_dice_losses'][-1], decimals=4), step=self.current_epoch) # metric 是可以持續被更新的.
        mlflow.log_metric("val_loss", np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4), step=self.current_epoch) # metric 是可以持續被更新的.
        mlflow.log_metric("val_ce_loss", np.round(self.logger.my_fantastic_logging['val_ce_losses'][-1], decimals=4), step=self.current_epoch) # metric 是可以持續被更新的.
        mlflow.log_metric("val_dice_loss", np.round(self.logger.my_fantastic_logging['val_dice_losses'][-1], decimals=4), step=self.current_epoch) # metric 是可以持續被更新的.
        # Dynamically log validation dice losses to mlflow
        for i in range(self.num_deep_supervision_levels):
            mlflow.log_metric(f"val_dice_loss{i}", np.round(self.logger.my_fantastic_logging[f'val_dice_loss{i}'][-1], decimals=4), step=self.current_epoch)
        mlflow.log_metric("val_fake_dice_loss", np.round(self.logger.my_fantastic_logging['val_fake_dice_losses'][-1], decimals=4), step=self.current_epoch) # metric 是可以持續被更新的.
        mlflow.log_metric("train pseudo dice", np.round(self.logger.my_fantastic_logging['train_mean_fg_dice'][-1], decimals=4), step=self.current_epoch) # metric 是可以持續被更新的.
        mlflow.log_metric("train pseudo dice mov. avg.", np.round(self.logger.my_fantastic_logging['train_ema_fg_dice'][-1], decimals=4), step=self.current_epoch) # metric 是可以持續被更新的.
        mlflow.log_metric("val pseudo dice", np.round(self.logger.my_fantastic_logging['mean_fg_dice'][-1], decimals=4), step=self.current_epoch) # metric 是可以持續被更新的.
        mlflow.log_metric("val pseudo dice  mov. avg.", np.round(self.logger.my_fantastic_logging['ema_fg_dice'][-1], decimals=4), step=self.current_epoch) # metric 是可以持續被更新的.
        # Dynamically log deep supervision dice to mlflow
        for i in range(self.num_deep_supervision_levels):
            mlflow.log_metric(f"train deep_supervision dice{i}", np.round(self.logger.my_fantastic_logging[f'train_supervision_dice{i}'][-1], decimals=4), step=self.current_epoch)
            mlflow.log_metric(f"val deep_supervision dice{i}", np.round(self.logger.my_fantastic_logging[f'val_supervision_dice{i}'][-1], decimals=4), step=self.current_epoch)

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best val EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

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

        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                checkpoint = {
                    'network_weights': self.network.module.state_dict() if self.is_ddp else self.network.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

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
        ## Auto log all MLflow entities
        mlflow.pytorch.autolog()

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
        
        experiment = mlflow.get_experiment_by_name(run_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(run_name)
        else:
            experiment_id = experiment.experiment_id

        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id): # 所有的mlflow寫入指令必須放置在 mlflow.start_run 範圍中. 
            self.on_train_start()
            ml_params = {
                "epochs": self.num_epochs,
                "learning_rate": self.initial_lr,
                "batch_size": self.configuration_manager.batch_size,
                "oversample_foreground_percent": self.oversample_foreground_percent
            }
            # Log training parameters.
            mlflow.log_params(ml_params)
            for epoch in range(self.current_epoch, self.num_epochs):
                self.on_epoch_start() #只有一個訊息而已

                self.on_train_epoch_start() #讀取網路跟進行 lr_scheduler 的控制
                train_outputs = []
                AVE_Queue = []
                start = time()
                for batch_id in range(self.num_iterations_per_epoch):
                    train_outputs.append(self.train_step(next(self.dataloader_train)))
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

                self.on_epoch_end()
                
                # Check if early stopping is triggered
                if self.should_stop_training:
                    self.print_to_log_file(f"Early Stopping: Training stopped at epoch {self.current_epoch}", also_print_to_console=True)
                    break

        self.on_train_end()

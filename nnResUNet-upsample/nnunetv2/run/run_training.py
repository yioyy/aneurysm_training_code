import json
import os
import socket
from typing import Union, Optional, Dict

import nnunetv2
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from torch.backends import cudnn


def parse_sampling_category_weights(raw: str) -> Dict[int, float]:
    """
    Parse sampling category weights from CLI string.

    Supported formats:
    - Ratio: "2:1:1:1" or "2,1,1,1" (maps to categories 1..4)
    - Key=Value pairs: "1=2,2=1,3=1,4=1" (separator can be ',' or ';')
    - JSON: "{\"1\": 2, \"2\": 1, \"3\": 1, \"4\": 1}" or "{\"1\": 2}"
    """
    if raw is None:
        raise ValueError("raw must not be None")
    s = raw.strip()
    if not s:
        raise ValueError("sampling_category_weights is empty")

    # JSON dict (keys may be strings)
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for sampling_category_weights: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError("JSON sampling_category_weights must be an object/dict")
        out: Dict[int, float] = {}
        for k, v in obj.items():
            try:
                kk = int(k)
            except Exception as e:
                raise ValueError(f"Invalid category id key '{k}' (must be int)") from e
            out[kk] = float(v)
        return out

    # key=value list
    if "=" in s:
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
        out: Dict[int, float] = {}
        for p in parts:
            if "=" not in p:
                raise ValueError(f"Invalid key=value token '{p}'")
            k_str, v_str = [x.strip() for x in p.split("=", 1)]
            out[int(k_str)] = float(v_str)
        return out

    # ratio list -> map to 1..N (expects 4 for vessel 4-class use)
    parts = [p.strip() for p in s.replace(":", ",").split(",") if p.strip()]
    vals = [float(p) for p in parts]
    if len(vals) != 4:
        raise ValueError("Ratio format must provide exactly 4 values, e.g. '2:1:1:1'")
    return {1: vals[0], 2: vals[1], 3: vals[2], 4: vals[3]}


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_name: str = 'nnUNetTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          use_compressed: bool = False,
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
                          early_stopping_min_delta: float = 0.0001,
                          sampling_category_weights: Optional[Dict[int, float]] = None,
                          sampling_category_weight_mode: Optional[str] = None):
    # load nnunet class and do sanity checks
    nnunet_trainer = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')
    if nnunet_trainer is None:
        raise RuntimeError(f'Could not find requested nnunet trainer {trainer_name} in '
                           f'nnunetv2.training.nnUNetTrainer ('
                           f'{join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}). If it is located somewhere '
                           f'else, please move it there.')
    assert issubclass(nnunet_trainer, nnUNetTrainer), 'The requested nnunet trainer class must inherit from ' \
                                                    'nnUNetTrainer'

    # optional: override sampling weights/mode at runtime (applies to this run only)
    if sampling_category_weights is not None:
        if hasattr(nnunet_trainer, "SAMPLING_CATEGORY_WEIGHTS"):
            nnunet_trainer.SAMPLING_CATEGORY_WEIGHTS = sampling_category_weights
        else:
            print("WARNING: Trainer does not define SAMPLING_CATEGORY_WEIGHTS; ignoring --sampling_category_weights.")
    if sampling_category_weight_mode is not None:
        if hasattr(nnunet_trainer, "SAMPLING_CATEGORY_WEIGHT_MODE"):
            nnunet_trainer.SAMPLING_CATEGORY_WEIGHT_MODE = sampling_category_weight_mode
        else:
            print("WARNING: Trainer does not define SAMPLING_CATEGORY_WEIGHT_MODE; ignoring --sampling_category_weight_mode.")

    # handle dataset input. If it's an ID we need to convert to int from string
    if dataset_name_or_id.startswith('Dataset'):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                             f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your '
                             f'input: {dataset_name_or_id}')

    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    nnunet_trainer = nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
                                    dataset_json=dataset_json, unpack_dataset=not use_compressed, device=device,
                                    initial_lr=initial_lr,
                                    oversample_foreground_percent=oversample_foreground_percent,
                                    oversample_foreground_percent_val=oversample_foreground_percent_val,
                                    num_iterations_per_epoch=num_iterations_per_epoch,
                                    num_epochs=num_epochs,
                                    optimizer_type=optimizer_type,
                                    lr_scheduler_type=lr_scheduler_type,
                                    enable_early_stopping=enable_early_stopping,
                                    early_stopping_patience=early_stopping_patience,
                                    early_stopping_min_delta=early_stopping_min_delta)
    
    return nnunet_trainer


def maybe_load_checkpoint(nnunet_trainer: nnUNetTrainer, continue_training: bool, validation_only: bool,
                          pretrained_weights_file: str = None):
    if continue_training:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_latest.pth')
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        if not isfile(expected_checkpoint_file):
            print(f"WARNING: Cannot continue training because there seems to be no checkpoint available to "
                               f"continue from. Starting a new training...")
    elif validation_only:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            raise RuntimeError(f"Cannot run validation because the training is not finished yet!")
    else:
        if pretrained_weights_file is not None:
            load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True)
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)


def setup_ddp(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def run_ddp(rank, dataset_name_or_id, configuration, fold, tr, p, use_compressed, disable_checkpointing, c, val, pretrained_weights, npz, world_size,
            initial_lr, oversample_foreground_percent, oversample_foreground_percent_val, num_iterations_per_epoch, num_epochs, optimizer_type, lr_scheduler_type,
            enable_early_stopping, early_stopping_patience, early_stopping_min_delta,
            sampling_category_weights, sampling_category_weight_mode):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(torch.device('cuda', dist.get_rank()))

    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, tr, p, use_compressed,
                                           initial_lr=initial_lr,
                                           oversample_foreground_percent=oversample_foreground_percent,
                                           oversample_foreground_percent_val=oversample_foreground_percent_val,
                                           num_iterations_per_epoch=num_iterations_per_epoch,
                                           num_epochs=num_epochs,
                                           optimizer_type=optimizer_type,
                                           lr_scheduler_type=lr_scheduler_type,
                                           enable_early_stopping=enable_early_stopping,
                                           early_stopping_patience=early_stopping_patience,
                                           early_stopping_min_delta=early_stopping_min_delta,
                                           sampling_category_weights=sampling_category_weights,
                                           sampling_category_weight_mode=sampling_category_weight_mode)
    
    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (c and val), f'Cannot set --c and --val flag at the same time. Dummy.'

    maybe_load_checkpoint(nnunet_trainer, c, val, pretrained_weights)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not val:
        nnunet_trainer.run_training()

    nnunet_trainer.perform_actual_validation(npz)
    cleanup_ddp()


def run_training(dataset_name_or_id: Union[str, int],
                 configuration: str, fold: Union[int, str],
                 trainer_class_name: str = 'nnUNetTrainer',
                 plans_identifier: str = 'nnUNetPlans',
                 pretrained_weights: Optional[str] = None,
                 num_gpus: int = 1,
                 use_compressed_data: bool = False,
                 export_validation_probabilities: bool = False,
                 continue_training: bool = False,
                 only_run_validation: bool = False,
                 disable_checkpointing: bool = False,
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
                 early_stopping_min_delta: float = 0.0001,
                 sampling_category_weights: Optional[Dict[int, float]] = None,
                 sampling_category_weight_mode: Optional[str] = None):
    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
                raise e

    if num_gpus > 1:
        assert device.type == 'cuda', f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"

        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ['MASTER_PORT'] = port  # str(port)

        mp.spawn(run_ddp,
                 args=(
                     dataset_name_or_id,
                     configuration,
                     fold,
                     trainer_class_name,
                     plans_identifier,
                     use_compressed_data,
                     disable_checkpointing,
                     continue_training,
                     only_run_validation,
                     pretrained_weights,
                     export_validation_probabilities,
                     num_gpus,
                     initial_lr,
                     oversample_foreground_percent,
                     oversample_foreground_percent_val,
                     num_iterations_per_epoch,
                     num_epochs,
                     optimizer_type,
                     lr_scheduler_type,
                     enable_early_stopping,
                     early_stopping_patience,
                     early_stopping_min_delta,
                     sampling_category_weights,
                     sampling_category_weight_mode),
                 nprocs=num_gpus,
                 join=True)
    else:
        nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name,
                                               plans_identifier, use_compressed_data, device=device,
                                               initial_lr=initial_lr,
                                               oversample_foreground_percent=oversample_foreground_percent,
                                               oversample_foreground_percent_val=oversample_foreground_percent_val,
                                               num_iterations_per_epoch=num_iterations_per_epoch,
                                               num_epochs=num_epochs,
                                               optimizer_type=optimizer_type,
                                               lr_scheduler_type=lr_scheduler_type,
                                               enable_early_stopping=enable_early_stopping,
                                               early_stopping_patience=early_stopping_patience,
                                               early_stopping_min_delta=early_stopping_min_delta,
                                               sampling_category_weights=sampling_category_weights,
                                               sampling_category_weight_mode=sampling_category_weight_mode)

        #已經確認load pretrain weights，所以先初始化一次
        if pretrained_weights is not None:
            nnunet_trainer.initialize()
            #因為已經做過初始化，所以

        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        if nnunet_trainer.network is None:
            print("Network is not initialized correctly!")
        else:
            print("Network is initialized successfully.")

        assert not (continue_training and only_run_validation), f'Cannot set --c and --val flag at the same time. Dummy.'

        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, pretrained_weights)

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if not only_run_validation:
            nnunet_trainer.run_training()

        nnunet_trainer.perform_actual_validation(export_validation_probabilities)


def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')
    parser.add_argument("--use_compressed", default=False, action="store_true", required=False,
                        help="[OPTIONAL] If you set this flag the training cases will not be decompressed. Reading compressed "
                             "data is much more CPU and (potentially) RAM intensive and should only be used if you "
                             "know what you are doing")
    parser.add_argument('--npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted '
                             'segmentations). Needed for finding the best ensemble.')
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')
    parser.add_argument('--val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and '
                             'you dont want to flood your hard drive with checkpoints.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                    help="Use this to set the device the training should run with. Available options are 'cuda' "
                         "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                         "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!")
    
    # Training hyperparameters
    parser.add_argument('--initial_lr', type=float, default=1e-4, required=False,
                        help='[OPTIONAL] Initial learning rate. Default: 1e-4')
    parser.add_argument('--oversample_foreground_percent', type=float, default=0.5, required=False,
                        help='[OPTIONAL] Oversample foreground percent for training. Default: 0.5')
    parser.add_argument('--oversample_foreground_percent_val', type=float, default=0.2, required=False,
                        help='[OPTIONAL] Oversample foreground percent for validation. Default: 0.2')
    parser.add_argument('--num_iterations_per_epoch', type=int, default=500, required=False,
                        help='[OPTIONAL] Number of iterations per epoch. Default: 500')
    parser.add_argument('--num_epochs', type=int, default=1000, required=False,
                        help='[OPTIONAL] Number of training epochs. Default: 1000')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'AdamW'], required=False,
                        help='[OPTIONAL] Optimizer type. Choose from ["SGD", "AdamW"]. Default: AdamW')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR', 
                        choices=['PolyLRScheduler', 'CosineAnnealingLR'], required=False,
                        help='[OPTIONAL] Learning rate scheduler type. Choose from ["PolyLRScheduler", "CosineAnnealingLR"]. Default: CosineAnnealingLR')
    
    # Early stopping parameters
    parser.add_argument('--enable_early_stopping', action='store_true', required=False,
                        help='[OPTIONAL] Enable early stopping based on validation EMA pseudo Dice. Default: False')
    parser.add_argument('--early_stopping_patience', type=int, default=200, required=False,
                        help='[OPTIONAL] Number of epochs with no improvement before stopping. Default: 50')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0001, required=False,
                        help='[OPTIONAL] Minimum change in validation EMA pseudo Dice to qualify as improvement. Default: 0.0001')

    # Sampling category upsample configuration (case-level sampling)
    parser.add_argument('--sampling_category_weights', type=str, default=None, required=False,
                        help='[OPTIONAL] Override SAMPLING_CATEGORY_WEIGHTS at runtime. '
                             'Formats: "2:1:1:1" or "2,1,1,1" or "1=2,2=1,3=1,4=1" or JSON like \'{"1":2,"2":1,"3":1,"4":1}\'.')
    parser.add_argument('--sampling_category_weight_mode', type=str, default=None, required=False,
                        choices=['multiplier', 'target_proportion'],
                        help='[OPTIONAL] Override SAMPLING_CATEGORY_WEIGHT_MODE at runtime. '
                             'Choose from ["multiplier", "target_proportion"].')
    
    args = parser.parse_args()

    assert args.device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    sampling_category_weights = parse_sampling_category_weights(args.sampling_category_weights) \
        if args.sampling_category_weights is not None else None

    run_training(args.dataset_name_or_id, args.configuration, args.fold, args.tr, args.p, args.pretrained_weights,
                 args.num_gpus, args.use_compressed, args.npz, args.c, args.val, args.disable_checkpointing,
                 device=device,
                 initial_lr=args.initial_lr,
                 oversample_foreground_percent=args.oversample_foreground_percent,
                 oversample_foreground_percent_val=args.oversample_foreground_percent_val,
                 num_iterations_per_epoch=args.num_iterations_per_epoch,
                 num_epochs=args.num_epochs,
                 optimizer_type=args.optimizer,
                 lr_scheduler_type=args.lr_scheduler,
                 enable_early_stopping=args.enable_early_stopping,
                 early_stopping_patience=args.early_stopping_patience,
                 early_stopping_min_delta=args.early_stopping_min_delta,
                 sampling_category_weights=sampling_category_weights,
                 sampling_category_weight_mode=args.sampling_category_weight_mode)


if __name__ == '__main__':
    run_training_entry()

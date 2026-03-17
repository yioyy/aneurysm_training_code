import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def load_pretrained_weights(network, fname, verbose=False):
    """
    THIS DOES NOT TRANSFER SEGMENTATION HEADS!

    network can be either a plain model or DDP. We need to account for that in the parameter names
    """
    saved_model = torch.load(fname, weights_only=False)
    pretrained_dict = saved_model['network_weights']
    is_ddp = isinstance(network, DDP)

    # skip_strings_in_pretrained = [
    #     '.seg_layers.',
    # ]

    skip_strings_in_pretrained = []

    model_dict = network.state_dict()

    # 比對新模型的每個 key：
    #   - 在 pretrained 中存在且 shape 相同 → 載入
    #   - 在 pretrained 中存在但 shape 不同 → 跳過（隨機初始化）
    #   - 在 pretrained 中不存在 → 跳過（隨機初始化）
    loaded_keys = []
    skipped_missing = []
    skipped_shape = []

    for key, param in model_dict.items():
        key_pretrained = key[7:] if is_ddp else key
        if any(s in key for s in skip_strings_in_pretrained):
            continue
        if key_pretrained not in pretrained_dict:
            skipped_missing.append(key_pretrained)
        elif param.shape != pretrained_dict[key_pretrained].shape:
            skipped_shape.append(
                f"{key_pretrained}: model={param.shape} vs pretrained={pretrained_dict[key_pretrained].shape}"
            )
        else:
            loaded_keys.append(key_pretrained)

    pretrained_dict = {'module.' + k if is_ddp else k: v
                       for k, v in pretrained_dict.items()
                       if (('module.' + k if is_ddp else k) in model_dict)
                       and model_dict['module.' + k if is_ddp else k].shape == v.shape
                       and all([i not in k for i in skip_strings_in_pretrained])}

    model_dict.update(pretrained_dict)

    print("################### Loading pretrained weights from file ", fname, '###################')
    print(f"  Loaded {len(loaded_keys)} / {len(model_dict)} parameter blocks.")
    if skipped_missing:
        print(f"  Skipped (not in pretrained, will use random init): {len(skipped_missing)} blocks")
        if verbose:
            for k in skipped_missing:
                print(f"    - {k}")
    if skipped_shape:
        print(f"  Skipped (shape mismatch, will use random init): {len(skipped_shape)} blocks")
        if verbose:
            for k in skipped_shape:
                print(f"    - {k}")
    if verbose:
        print("  Loaded blocks:")
        for key in loaded_keys:
            print(f"    + {key}")
    print("################### Done ###################")
    network.load_state_dict(model_dict)



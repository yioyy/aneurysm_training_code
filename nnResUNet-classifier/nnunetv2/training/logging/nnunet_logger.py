import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join

matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt


class nnUNetLogger(object):
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """
    def __init__(self, verbose: bool = False, num_deep_supervision_levels: int = 5):
        # Base logging items that don't depend on deep supervision levels
        self.my_fantastic_logging = {
            'mean_fg_dice': list(),
            'ema_fg_dice': list(),
            'dice_per_class_or_region': list(),
            'train_losses': list(),
            'train_ce_losses': list(),
            'train_dice_losses': list(),
            'train_mean_fg_dice': list(),
            'train_ema_fg_dice': list(),
            'train_fake_dice_losses': list(),
            'train_dice_per_class_or_region': list(),          
            'val_ce_losses': list(),
            'val_dice_losses': list(),
            'val_losses': list(),
            'val_fake_dice_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list(),
            'train_seg_losses': list(),
            'train_cls_losses': list(),
            'val_seg_losses': list(),
            'val_cls_losses': list(),
            'train_accuracys': list(),
            'train_sensitivitys': list(),
            'train_specificitys': list(),
            'val_accuracys': list(),
            'val_sensitivitys': list(),
            'val_specificitys': list()
        }
        
        # Dynamically add deep supervision related logging items
        self.num_deep_supervision_levels = num_deep_supervision_levels
        for i in range(num_deep_supervision_levels):
            self.my_fantastic_logging[f'train_dice_loss{i}'] = list()
            self.my_fantastic_logging[f'train_supervision_dice{i}'] = list()
            self.my_fantastic_logging[f'val_dice_loss{i}'] = list()
            self.my_fantastic_logging[f'val_supervision_dice{i}'] = list()
            
        self.verbose = verbose
        # shut up, this logging is great

    def log(self, key, value, epoch: int):
        """
        sometimes shit gets messed up. We try to catch that here
        """
        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

        if self.verbose: print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == (epoch + 1), 'something went horribly wrong. My logging ' \
                                                                       'lists length is off by more than 1'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

        # handle the ema_fg_dice special case! It is automatically logged when we add a new mean_fg_dice
        if key == 'mean_fg_dice':
            new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_fg_dice']) > 0 else value
            self.log('ema_fg_dice', new_ema_pseudo_dice, epoch)

        # handle the ema_fg_dice special case! It is automatically logged when we add a new mean_fg_dice
        if key == 'train_mean_fg_dice':
            new_ema_pseudo_dice = self.my_fantastic_logging['train_ema_fg_dice'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['train_ema_fg_dice']) > 0 else value
            self.log('train_ema_fg_dice', new_ema_pseudo_dice, epoch)

    def plot_progress_png(self, output_folder):
        # we infer the epoch form our internal logging
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1  # lists of epoch 0 have len 1
        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(6, 1, figsize=(30, 110))
        # regular progress.png as we are used to from previous nnU-Net versions
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label="loss_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['train_seg_losses'][:epoch + 1], color='cyan', ls='-', label="seg_loss_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['train_cls_losses'][:epoch + 1], color='purple', ls='-', label="cls_loss_tr", linewidth=4)
        #ax.plot(x_values, self.my_fantastic_logging['train_fake_dice_losses'][:epoch + 1], color='violet', ls='dotted', label="fake_dice_loss_tr", linewidth=3)
        ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-', label="loss_val", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_seg_losses'][:epoch + 1], color='orange', ls='-', label="seg_loss_val", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_cls_losses'][:epoch + 1], color='magenta', ls='-', label="cls_loss_val", linewidth=4)
        #ax.plot(x_values, self.my_fantastic_logging['val_fake_dice_losses'][:epoch + 1], color='sienna', ls='dotted', label="fake_dice_loss_val", linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['train_mean_fg_dice'][:epoch + 1], color='y', ls='dotted', label="train pseudo dice",
                 linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['train_ema_fg_dice'][:epoch + 1], color='y', ls='-', label="train pseudo dice (mov. avg.)",
                 linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label="val pseudo dice",
                 linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label="val pseudo dice (mov. avg.)",
                 linewidth=4)
        #ax2.plot(x_values, self.my_fantastic_logging['train_supervision_dice0'][:epoch + 1], color='goldenrod', ls='-', label="train deep_supervision dice0",
        #         linewidth=4)
        #ax2.plot(x_values, self.my_fantastic_logging['val_supervision_dice0'][:epoch + 1], color='greenyellow', ls='-', label="val deep_supervision dice0",
        #         linewidth=4)

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # epoch times to see whether the training speed is consistent (inconsistent means there are other jobs
        # clogging up the system)
        # ax = ax_all[1]
        # ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
        #                                          self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1], color='b',
        #         ls='-', label="epoch duration", linewidth=4)
        # ylim = [0] + [ax.get_ylim()[1]]
        # ax.set(ylim=ylim)
        # ax.set_xlabel("epoch")
        # ax.set_ylabel("time [s]")
        # ax.legend(loc=(0, 1))

        #第二格改成只看dice，不然原圖會太亂
        ax = ax_all[1]
        x_values = list(range(epoch + 1))
        
        # Dynamic plotting for deep supervision dice
        colors = ['b', 'cyan', 'purple', 'g', 'gold', 'navy', 'darkgreen', 'orange', 'red', 'brown']
        val_colors = ['r', 'orange', 'magenta', 'darkred', 'brown', 'pink', 'lime', 'yellow', 'blue', 'gray']
        
        for i in range(self.num_deep_supervision_levels):
            train_key = f'train_supervision_dice{i}'
            val_key = f'val_supervision_dice{i}'
            if train_key in self.my_fantastic_logging and val_key in self.my_fantastic_logging:
                color_idx = i % len(colors)
                val_color_idx = i % len(val_colors)
                ax.plot(x_values, self.my_fantastic_logging[train_key][:epoch + 1], 
                       color=colors[color_idx], ls='-', label=f"train deep_supervision dice{i}", linewidth=4)
                ax.plot(x_values, self.my_fantastic_logging[val_key][:epoch + 1], 
                       color=val_colors[val_color_idx], ls='-', label=f"val deep_supervision dice{i}", linewidth=4)
        
        ax.set_xlabel("epoch")
        ax.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))

        ax = ax_all[2]
        x_values = list(range(epoch + 1))
        
        # Dynamic plotting for dice losses
        for i in range(self.num_deep_supervision_levels):
            train_key = f'train_dice_loss{i}'
            val_key = f'val_dice_loss{i}'
            if train_key in self.my_fantastic_logging and val_key in self.my_fantastic_logging:
                color_idx = i % len(colors)
                val_color_idx = i % len(val_colors)
                ax.plot(x_values, self.my_fantastic_logging[train_key][:epoch + 1], 
                       color=colors[color_idx], ls='-', label=f"train_dice_loss{i}", linewidth=4)
                ax.plot(x_values, self.my_fantastic_logging[val_key][:epoch + 1], 
                       color=val_colors[val_color_idx], ls='-', label=f"val_dice_loss{i}", linewidth=4)
        
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(loc=(0, 1))

        #第三格改成只看ce_loss，不然原圖loss太小看不出變化
        ax = ax_all[3]
        x_values = list(range(epoch + 1))
        ax.plot(x_values, self.my_fantastic_logging['train_ce_losses'][:epoch + 1], color='cyan', ls='-', label="ce_loss_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_ce_losses'][:epoch + 1], color='orange', ls='-', label="ce_loss_val", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(loc=(0, 1))

        #第四格改成只看分類器的loss跟accuracy等等等，不然原圖loss太小看不出變化
        ax = ax_all[4]
        x_values = list(range(epoch + 1))
        ax.plot(x_values, self.my_fantastic_logging['train_accuracys'][:epoch + 1], color='b', ls='-', label="accuracy_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_accuracys'][:epoch + 1], color='r', ls='-', label="accuracy_val", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['train_sensitivitys'][:epoch + 1], color='cyan', ls='-', label="sensitivity_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_sensitivitys'][:epoch + 1], color='orange', ls='-', label="sensitivity_val", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['train_specificitys'][:epoch + 1], color='y', ls='-', label="specificity_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_specificitys'][:epoch + 1], color='g', ls='-', label="specificity_val", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("percent")
        ax.legend(loc=(0, 1))

        # learning rate
        ax = ax_all[5]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint

import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm
from itertools import combinations
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

from utils.dataset_synapse import Synapse_dataset, RandomGenerator
from utils.utils import powerset#, cal_params_flops
from utils.utils import one_hot_encoder
from utils.utils import DiceLoss
from utils.utils import val_single_volume


def _worker_init_fn(worker_id, base_seed=1234):
    """Module-level so Windows `spawn` workers can pickle it (see
    tools/apply_low_vram_patch.py). Seeding is unchanged: base_seed + worker_id."""
    random.seed(base_seed + worker_id)


class WeightedFusion(nn.Module):
    """
    Computes per-pixel attention weights for each prediction from a set of decoder stages,
    then fuses them as a weighted sum. This preserves the original logit values without adding
    extra non-linearities that might distort class scores.
    """
    def __init__(self, num_stages, num_classes):
        """
        Args:
            num_stages (int): Number of predictions (decoder stages) to fuse.
            num_classes (int): Number of channels (classes) in each prediction.
        """
        super(WeightedFusion, self).__init__()
        # For each stage, learn a 1x1 conv that outputs a single-channel weight map.
        self.weight_convs = nn.ModuleList(
            [nn.Conv2d(num_classes, 1, kernel_size=1) for _ in range(num_stages)]
        )
    
    def forward(self, predictions):
        """
        Args:
            predictions (list[Tensor]): List of predictions, each of shape [B, C, H, W].
        Returns:
            fused (Tensor): Fused prediction of shape [B, C, H, W].
        """
        # Compute weight maps for each stage.
        weight_maps = [conv(pred) for pred, conv in zip(predictions, self.weight_convs)]
        # Stack weight maps: [B, num_stages, 1, H, W]
        weights = torch.stack(weight_maps, dim=1)
        # Normalize weights per pixel across stages.
        weights = F.softmax(weights, dim=1)
        # Stack predictions: [B, num_stages, C, H, W]
        preds = torch.stack(predictions, dim=1)
        # Fuse via weighted sum.
        fused = torch.sum(weights * preds, dim=1)
        return fused
    
class CombinatorialMutationsLossModule(nn.Module):
    """
    Free-floating approach:
      - Each raw param -> a positive weight via Softplus.
      - No sum constraint.
      - The ratio w_i / w_j is determined by (theta_i - theta_j).
      - The total can be anything the model finds optimal.
    """
    def __init__(
        self,
        original_num_maps,
        num_classes,
        selecetd_num_maps,
        operations=None,
        use_learnable_weights=True,
        supervision='mutation',
        lc1=0.3,
        lc2=0.7
    ):
        super().__init__()
        if operations is None:
            operations = ['add','sub','mul','concat','weighted_fusion','avg','max']

        self.num_maps = selecetd_num_maps
        self.num_classes = num_classes
        self.operations = operations
        self.use_learnable_weights = use_learnable_weights
        self.supervision = supervision
        self.lc1 = lc1
        self.lc2 = lc2

        # Create raw parameters for original outputs
        if use_learnable_weights:
            self.original_weights = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in range(original_num_maps)]
            )

        self.combination_indices = {}
        if use_learnable_weights:
            self.synthesized_weights = nn.ModuleDict()

        for op in self.operations:
            comb_list = []
            for k in range(2, selecetd_num_maps + 1):
                comb_list.extend(list(combinations(range(selecetd_num_maps), k)))
            self.combination_indices[op] = comb_list

            if use_learnable_weights:
                self.synthesized_weights[op] = nn.ParameterList(
                    [nn.Parameter(torch.zeros(1)) for _ in range(len(comb_list))]
                )

        # Weighted fusion modules
        self.weighted_fusion_modules = nn.ModuleDict({
            str(k): WeightedFusion(k, num_classes)
            for k in range(2, selecetd_num_maps + 1)
        })

    def _compute_all_weights(self):
        """
        Convert each raw param to a positive weight using softplus -> (0,∞).
        No sum constraint.
        """
        # original
        if not self.use_learnable_weights:
            return None, None

        orig_vals = [F.softplus(w) for w in self.original_weights]

        synth_vals = {}
        for op, plist in self.synthesized_weights.items():
            synth_vals[op] = [F.softplus(p) for p in plist]

        return orig_vals, synth_vals

    def forward(self, output_maps, label_batch=None, ce_loss=None, dice_loss=None):
        device = output_maps[0].device
        deep_supervision_loss = 0.0
        mutation_loss = 0.0
        fused_logits = []
        if not self.use_learnable_weights:
            # uniform weighting
            orig_weights = [torch.ones(1, device=device) for _ in output_maps]
            syn_weights = {}
            for op, combos in self.combination_indices.items():
                syn_weights[op] = [torch.ones(1, device=device) for _ in combos]
        else:
            orig_vals, synth_vals = self._compute_all_weights()
            orig_weights = orig_vals
            syn_weights = synth_vals

        # 1) Original maps
        for i, fmap in enumerate(output_maps):
            w_i = orig_weights[i]
            if label_batch==None and ce_loss==None and dice_loss==None:
                continue
            loss_ce = ce_loss(fmap, label_batch.long())
            loss_dice = dice_loss(fmap, label_batch, softmax=True)
            combined_loss = self.lc1 * loss_ce + self.lc2 * loss_dice
            deep_supervision_loss += w_i * combined_loss

        # 2) Synthesis combos
        for op in self.operations:
            combos = self.combination_indices[op]
            w_list = syn_weights[op] if self.use_learnable_weights else [torch.ones(1, device=device)]*len(combos)
            for idx, comb in enumerate(combos):
                # produce mutated map
                if op == 'add':
                    mutated = sum(output_maps[c] for c in comb)
                elif op == 'avg':
                    mutated = sum(output_maps[c] for c in comb) / len(comb)
                elif op == 'sub':
                    mutated = output_maps[comb[0]] - sum(output_maps[c] for c in comb[1:])
                elif op == 'mul':
                    mutated = output_maps[comb[0]]
                    for c in comb[1:]:
                        mutated = mutated * output_maps[c]
                elif op == 'concat':
                    cat = torch.cat([output_maps[c] for c in comb], dim=1)
                    conv = nn.Conv2d(cat.size(1), self.num_classes, kernel_size=1).to(device)
                    mutated = conv(cat)
                elif op in ['weighted_fusion','wf']:
                    mod = self.weighted_fusion_modules[str(len(comb))]
                    mutated = mod([output_maps[c] for c in comb])
                elif op == 'max':
                    mutated = torch.stack([output_maps[c] for c in comb], dim=0).max(dim=0).values
                else:
                    raise ValueError(f"Unsupported op: {op}")
                fused_logits.append(mutated)
                if label_batch==None and ce_loss==None and dice_loss==None:
                    continue
                loss_ce = ce_loss(mutated, label_batch.long())
                loss_dice = dice_loss(mutated, label_batch, softmax=True)
                combined_loss = self.lc1 * loss_ce + self.lc2 * loss_dice
                mutation_loss += w_list[idx] * combined_loss

        if label_batch==None and ce_loss==None and dice_loss==None:
            return fused_logits
        if self.supervision in ['mutation', 'lomix']:
            final_loss = deep_supervision_loss + mutation_loss
        else:
            final_loss = deep_supervision_loss
        return final_loss, deep_supervision_loss, mutation_loss

    def print_weights(self):
        """Log raw params + effective weights (softplus) without normalization."""
        if not self.use_learnable_weights:
            logging.info("No learnable weights. Using uniform weighting.")
            return

        # raw
        raw_orig = [p.item() for p in self.original_weights]
        raw_synth = {}
        for op, plist in self.synthesized_weights.items():
            raw_synth[op] = [p.item() for p in plist]

        # transform
        orig_vals, synth_vals = self._compute_all_weights()

        # log original
        logging.info("Original Weights (raw): %s", " ".join(f"{x:.4f}" for x in raw_orig))
        logging.info("Original Weights (softplus): %s",
                     " ".join(f"{v.item():.4f}" for v in orig_vals))
        logging.info("   => sum(original) = %.4f", float(torch.stack(orig_vals).sum()))
        print("Original Weights (raw): %s", " ".join(f"{x:.4f}" for x in raw_orig))
        print("Original Weights (softplus): %s",
                     " ".join(f"{v.item():.4f}" for v in orig_vals))
        print("   => sum(original) = %.4f", float(torch.stack(orig_vals).sum()))

        # combos
        for op in self.operations:
            if op not in raw_synth:
                continue
            rvals = raw_synth[op]
            svals = synth_vals[op]
            logging.info("Synthesized Weights for '%s' (raw): %s", op,
                         " ".join(f"{rv:.4f}" for rv in rvals))
            logging.info("Synthesized Weights for '%s' (softplus): %s", op,
                         " ".join(f"{sv.item():.4f}" for sv in svals))
            logging.info("   => sum(%s) = %.4f", op, float(torch.stack(svals).sum()))
            print("Synthesized Weights for '%s' (raw): %s", op,
                         " ".join(f"{rv:.4f}" for rv in rvals))
            print("Synthesized Weights for '%s' (softplus): %s", op,
                         " ".join(f"{sv.item():.4f}" for sv in svals))
            print("   => sum(%s) = %.4f", op, float(torch.stack(svals).sum()))


    def save_weights(self, save_path):
        torch.save(self.state_dict(), save_path)
        logging.info(f"Saved parameters to {save_path}")

    def load_weights(self, load_path):
        self.load_state_dict(torch.load(load_path))
        logging.info(f"Loaded parameters from {load_path}")


def inference(args, model, best_performance):
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, nclass=args.num_classes)
    
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = val_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, best_performance))
    return performance

def trainer_synapse(args, model, snapshot_path, supervision='lomix', operations=['add', 'mul', 'wf','concat'], n_outs=4, use_learnable_weights=True):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", nclass=args.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=int(getattr(args, 'num_workers', 2)), pin_memory=True,
                             worker_init_fn=functools.partial(_worker_init_fn,
                                                              base_seed=args.seed))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    #if args.n_gpu > 1:
    #    model = nn.DataParallel(model)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    loss_module = CombinatorialMutationsLossModule(4, num_classes, selecetd_num_maps=n_outs, operations=operations, use_learnable_weights=use_learnable_weights).to(device) #'concat',
    #'add', 'mul', 'wf', 
    # Include parameters of both the model and the loss module.
    optimizer = optim.AdamW(list(model.parameters()) + list(loss_module.parameters()), lr=base_lr, weight_decay=0.0001)

    # --- low-VRAM adaptation (see tools/apply_low_vram_patch.py) -------------
    _amp = bool(getattr(args, 'amp', False))
    _accum = max(1, int(getattr(args, 'accum', 1)))
    _eval_every = max(1, int(getattr(args, 'eval_every', 1)))
    _scaler = torch.cuda.amp.GradScaler(enabled=_amp)
    _smoke_steps = int(getattr(args, 'smoke_steps', 0))
    logging.info('[low-vram] amp=%s accum=%d eval_every=%d num_workers=%s '
                 'per_step_batch=%d effective_batch=%d'
                 % (_amp, _accum, _eval_every, getattr(args, 'num_workers', 2),
                    batch_size, batch_size * _accum))
    # ------------------------------------------------------------------------

    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    #optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    _start_epoch = 0
    _state_path = os.path.join(snapshot_path, 'train_state.pth')
    if getattr(args, 'resume', False) and os.path.isfile(_state_path):
        _ck = torch.load(_state_path, map_location=device)
        model.load_state_dict(_ck['model'])
        optimizer.load_state_dict(_ck['optimizer'])
        loss_module.load_state_dict(_ck['loss_module'])
        _scaler.load_state_dict(_ck['scaler'])
        best_performance = _ck['best_performance']
        iter_num = _ck['iter_num']
        _start_epoch = _ck['epoch'] + 1
        logging.info('[resume] full state restored from %s: resuming at epoch %d, '
                     'iter %d, best_dice %.6f (optimizer and scaler included)'
                     % (_state_path, _start_epoch, iter_num, best_performance))
    elif getattr(args, 'resume', False):
        logging.info('[resume] requested but no checkpoint at %s -- starting from scratch'
                     % _state_path)
    iterator = tqdm(range(_start_epoch, max_epoch), ncols=70)

    def _save_train_state(_epoch):
        """Everything needed to continue identically. Written atomically
        (tmp + os.replace) so a shutdown mid-write cannot truncate it. Reads
        best_performance/iter_num from the enclosing scope at call time."""
        _tmp = _state_path + '.tmp'
        torch.save({'epoch': _epoch,
                    'iter_num': iter_num,
                    'best_performance': best_performance,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss_module': loss_module.state_dict(),
                    'scaler': _scaler.state_dict()}, _tmp)
        os.replace(_tmp, _state_path)
    
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    for epoch_num in iterator:
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            
            with torch.cuda.amp.autocast(enabled=_amp):
                P = model(image_batch, mode='train')
            if _amp:   # losses are computed in fp32 so the loss maths is unchanged
                P = [p.float() for p in P] if isinstance(P, list) else P.float()

            if  not isinstance(P, list):
                P = [P]
            
            loss = 0.0
            deep_supervision_loss = 0.0
            mutation_loss = 0.0
            lc1, lc2 = 0.3, 0.7 #0.3, 0.7
            #print(label_batch.shape)
            if supervision in ['mutation', 'lomix', 'deep_supervision']:
                loss, deep_supervision_loss, mutation_loss = loss_module(P, label_batch, ce_loss, dice_loss)
            else:
                loss_ce = ce_loss(P[-1], label_batch[:].long())
                loss_dice = dice_loss(P[-1], label_batch, softmax=True)
                loss += (lc1 * loss_ce + lc2 * loss_dice)

            _scaler.scale(loss / _accum).backward()
            if (i_batch + 1) % _accum == 0:
                _scaler.step(optimizer)
                _scaler.update()
                optimizer.zero_grad(set_to_none=True)
            lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            if _smoke_steps and iter_num >= _smoke_steps:
                # Save on the way out so the smoke path exercises the real
                # checkpoint-writing code, not a separate shortcut.
                _save_train_state(epoch_num)
                logging.info('[smoke] reached %d iterations, loss %f -- state saved '
                             'to %s, stopping before validation'
                             % (iter_num, float(loss), _state_path))
                iterator.close()
                writer.close()
                return "Smoke run finished"
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/deep_supervision_loss', deep_supervision_loss, iter_num)
            writer.add_scalar('info/mutation_loss', mutation_loss, iter_num)

            if iter_num % 50 == 0:
                logging.info('iteration %d, epoch %d : loss : %f, deep_supervision_loss : %f, mutation_loss : %f, lr: %f' % (iter_num, epoch_num, loss.item() if isinstance(loss, torch.Tensor) else loss, deep_supervision_loss.item() if isinstance(deep_supervision_loss, torch.Tensor) else deep_supervision_loss, mutation_loss.item() if isinstance(mutation_loss, torch.Tensor) else mutation_loss, lr_))
     
        logging.info('iteration %d, epoch %d : loss : %f, deep_supervision_loss : %f, mutation_loss : %f, lr: %f' % (iter_num, epoch_num, loss.item() if isinstance(loss, torch.Tensor) else loss, deep_supervision_loss.item() if isinstance(deep_supervision_loss, torch.Tensor) else deep_supervision_loss, mutation_loss.item() if isinstance(mutation_loss, torch.Tensor) else mutation_loss, lr_))
        
        logging.info('epoch %d : peak_train_vram_MiB : %.1f'
                     % (epoch_num, torch.cuda.max_memory_allocated() / 2 ** 20))
        torch.cuda.reset_peak_memory_stats()

        # Log the weights.
        loss_module.print_weights()
        
        # Later, load weights back.
        #loss_module.load_weights("combinatorial_loss_weights.pth")
        
        save_mode_path = os.path.join(snapshot_path, 'last.pth')
        torch.save(model.state_dict(), save_mode_path)
        _save_train_state(epoch_num)
        
        if (epoch_num + 1) % _eval_every == 0 or epoch_num >= max_epoch - 1:
            performance = inference(args, model, best_performance)
        else:
            performance = -1.0   # skipped this epoch; never beats best_performance

        # Upstream calls model.train() ONCE before the epoch loop, while
        # inference() calls model.eval() at the end of every epoch and never
        # switches back -- so from epoch 1 onward training runs in eval mode
        # (decoder BatchNorm uses frozen running stats, dropout/drop-path off).
        # Default OFF: the published numbers were produced with this behaviour,
        # so leaving it intact is what "replication" means. Turn it on only as
        # an explicit, separately reported experiment.
        if getattr(args, 'restore_train_mode', False):
            model.train()
        
        save_interval = 50

        if(best_performance <= performance):
            best_performance = performance
            save_mode_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            # Save weights.
            loss_module.save_weights(os.path.join(snapshot_path,'combinatorial_loss_weights_best.pth'))
            
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            # Save weights.
            loss_module.save_weights(os.path.join(snapshot_path,'combinatorial_loss_weights_'+'epoch_' + str(epoch_num) + '.pth'))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            # Save weights.
            loss_module.save_weights(os.path.join(snapshot_path,'combinatorial_loss_weights_'+'epoch_' + str(epoch_num) + '.pth'))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

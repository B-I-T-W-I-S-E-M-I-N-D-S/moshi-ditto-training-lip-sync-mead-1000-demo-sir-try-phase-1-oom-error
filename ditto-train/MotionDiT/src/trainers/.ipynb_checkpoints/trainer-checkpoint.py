import torch
import os
import time
from tqdm import trange, tqdm
import traceback
import numpy as np

from ..utils.utils import load_json, DictAverageMeter, dump_pkl
from ..models.modules.adan import Adan
from ..models.LMDM import LMDM
from ..datasets.s2_dataset_v2 import Stage2Dataset as Stage2DatasetV2
from ..options.option import TrainOptions


class Trainer:
    def __init__(self, opt: TrainOptions):
        self.opt = opt

        print(time.asctime(), '_init_accelerate')
        self._init_accelerate()

        print(time.asctime(), '_init_LMDM')
        self.LMDM = self._init_LMDM()

        print(time.asctime(), '_init_dataset')
        self.data_loader = self._init_dataset()

        print(time.asctime(), '_init_optim')
        self.optim = self._init_optim()

        print(time.asctime(), '_set_accelerate')
        self._set_accelerate()

        print(time.asctime(), '_init_log')
        self._init_log()

        # Lip-sync loss (optional)
        if opt.use_lip_sync_loss:
            print(time.asctime(), '_init_lip_sync')
            self._init_lip_sync()

    def _init_accelerate(self):
        opt = self.opt
        if opt.use_accelerate:
            from accelerate import Accelerator
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            self.is_main_process = self.accelerator.is_main_process
            self.process_index = self.accelerator.process_index
        else:
            self.accelerator = None
            self.device = 'cuda'
            self.is_main_process = True
            self.process_index = 0

    def _set_accelerate(self):
        if self.accelerator is None:
            return
        
        self.LMDM.use_accelerator(self.accelerator)
        self.optim = self.accelerator.prepare(self.optim)
        self.data_loader = self.accelerator.prepare(self.data_loader)

        self.accelerator.wait_for_everyone()

    def _init_LMDM(self):
        opt = self.opt

        part_w_dict = None
        if opt.part_w_dict_json:
            part_w_dict = load_json(opt.part_w_dict_json)
        dim_ws = None
        if opt.dim_ws_npy:
            dim_ws = np.load(opt.dim_ws_npy)

        lmdm = LMDM(
            motion_feat_dim=opt.motion_feat_dim,
            audio_feat_dim=opt.audio_feat_dim,
            seq_frames=opt.seq_frames,
            part_w_dict=part_w_dict,   # only for train
            checkpoint=opt.checkpoint,
            device=self.device,
            use_last_frame_loss=opt.use_last_frame_loss,
            use_reg_loss=opt.use_reg_loss,
            dim_ws=dim_ws,
        )

        return lmdm

    def _init_dataset(self):
        opt = self.opt

        if opt.dataset_version in ['v2']:
            Stage2Dataset = Stage2DatasetV2
        else:
            raise NotImplementedError()

        dataset = Stage2Dataset(
            data_list_json=opt.data_list_json, 
            seq_len=opt.seq_frames,
            preload=opt.data_preload, 
            cache=opt.data_cache, 
            preload_pkl=opt.data_preload_pkl, 
            motion_feat_dim=opt.motion_feat_dim, 
            motion_feat_start=opt.motion_feat_start,
            motion_feat_offset_dim_se=opt.motion_feat_offset_dim_se,
            use_eye_open=opt.use_eye_open,
            use_eye_ball=opt.use_eye_ball,
            use_emo=opt.use_emo,
            use_sc=opt.use_sc,
            use_last_frame=opt.use_last_frame,
            use_lmk=opt.use_lmk,
            use_cond_end=opt.use_cond_end,
            mtn_mean_var_npy=opt.mtn_mean_var_npy,
            reprepare_idx_map=opt.reprepare_idx_map,
            use_lip_sync_loss=opt.use_lip_sync_loss,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        return data_loader
    
    def _init_optim(self):
        opt = self.opt
        optim = Adan(self.LMDM.model.parameters(), lr=opt.lr, weight_decay=0.02)
        return optim

    def _init_log(self):
        opt = self.opt
        
        experiment_path = os.path.join(opt.experiment_dir, opt.experiment_name)
        self.experiment_path = experiment_path
        self.error_log_path = os.path.join(experiment_path, 'error')
        
        if not self.is_main_process:
            return

        # ckpt
        self.ckpt_path = os.path.join(experiment_path, 'ckpts')
        os.makedirs(self.ckpt_path, exist_ok=True)

        # lip-sync debug directory
        self.lip_debug_path = os.path.join(experiment_path, 'lip_debug')
        if opt.use_lip_sync_loss and opt.lip_sync_debug:
            os.makedirs(self.lip_debug_path, exist_ok=True)
            print(f"[LipSync Debug] Images will be saved to: {self.lip_debug_path}")

        # save opt
        opt_pkl = os.path.join(experiment_path, 'opt.pkl')
        dump_pkl(vars(opt), opt_pkl)

        # loss log
        loss_log = os.path.join(experiment_path, 'loss.log')
        self.loss_logger = open(loss_log, 'a')

        self.ckpt_file_list_for_clear = []
        self.best_loss = float('inf')  # track best model

    def _init_lip_sync(self):
        """Initialize frozen SyncNet + renderer for lip-sync loss."""
        import os as _os
        from ..models.lip_sync_loss import LipSyncLoss

        opt = self.opt
        warp_ckpt = _os.path.join(opt.ditto_pytorch_path, 'models', 'warp_network.pth')
        decoder_ckpt = _os.path.join(opt.ditto_pytorch_path, 'models', 'decoder.pth')

        assert _os.path.isfile(opt.syncnet_checkpoint), \
            f"SyncNet checkpoint not found: {opt.syncnet_checkpoint}"
        assert _os.path.isfile(warp_ckpt), \
            f"WarpingNetwork checkpoint not found: {warp_ckpt}"
        assert _os.path.isfile(decoder_ckpt), \
            f"SPADEDecoder checkpoint not found: {decoder_ckpt}"

        self.lip_sync_module = LipSyncLoss(
            syncnet_ckpt=opt.syncnet_checkpoint,
            warp_ckpt=warp_ckpt,
            decoder_ckpt=decoder_ckpt,
            device=self.device,
            num_frames=opt.lip_sync_num_frames,
        )
        print(f"[LipSync] Initialized — λ1={opt.lip_sync_lambda1}, "
              f"λ2={opt.lip_sync_lambda2}, "
              f"every={opt.lip_sync_every_n_steps} steps, "
              f"batch={opt.lip_sync_batch_size}")

    def _compute_lip_sync_loss(self, x_recon, data_dict, debug=False):
        """
        Compute lip-sync loss on a subset of the batch.

        Args:
            x_recon:   (B, L, 265) predicted clean motion from diffusion
            data_dict: batch data dict with lipsync_* fields
            debug:     if True, log diagnostics and save images

        Returns:
            lip_loss:      scalar total lip-sync loss (weighted)
            lip_loss_dict: dict with individual loss components
        """
        import random as _random

        opt = self.opt
        B = x_recon.shape[0]

        # Check which samples have valid lip-sync features
        if 'lipsync_valid' not in data_dict:
            return None, {}
        valid_mask = data_dict['lipsync_valid']  # (B,) bool tensor
        valid_indices = [i for i in range(B) if valid_mask[i]]
        if len(valid_indices) == 0:
            if debug:
                print("  [DEBUG] No valid lip-sync samples in batch!")
            return None, {}

        B_sync = min(opt.lip_sync_batch_size, len(valid_indices))

        if debug:
            print(f"  [DEBUG] Valid samples: {len(valid_indices)}/{B}, using {B_sync}")

        # Get lip-sync data from data_dict
        t_starts = data_dict['lipsync_t_start']        # (B,) int
        f_s_batch = data_dict['lipsync_f_s']            # (B, 32, 16, 64, 64)
        x_s_batch = data_dict['lipsync_x_s']            # (B, 21, 3)
        kp_c_batch = data_dict['lipsync_kp_canonical']   # (B, 63)
        A_batch = data_dict['lipsync_A']                 # (B, 512)
        sim_gt_batch = data_dict['lipsync_sim_gt']       # (B,)

        # Select random subset from valid samples only
        if B_sync < len(valid_indices):
            indices = _random.sample(valid_indices, B_sync)
        else:
            indices = valid_indices

        # Extract 5-frame predicted motion windows per sample
        pred_windows = []
        for i in indices:
            ts = int(t_starts[i])
            te = ts + opt.lip_sync_num_frames
            pred_windows.append(x_recon[i, ts:te])     # (5, 265)
        pred_windows = torch.stack(pred_windows)         # (B_sync, 5, 265)

        if debug:
            print(f"  [DEBUG] pred_windows shape: {pred_windows.shape}")
            print(f"  [DEBUG] pred_windows requires_grad: {pred_windows.requires_grad}")
            print(f"  [DEBUG] pred_windows range: [{pred_windows.min():.4f}, {pred_windows.max():.4f}]")
            # Check if motion varies across frames
            motion_diff = (pred_windows[:, 1:] - pred_windows[:, :-1]).abs().mean()
            print(f"  [DEBUG] inter-frame motion difference: {motion_diff:.6f}")

        # Move precomputed features to device
        def _to_dev(t, idx):
            v = t[idx] if isinstance(idx, list) else t[idx:idx+1]
            if isinstance(idx, list):
                v = torch.stack([t[i] for i in idx])
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            return v.to(self.device, dtype=torch.float32)

        kp_canonical = _to_dev(kp_c_batch, indices)     # (B_sync, 63)
        f_s = _to_dev(f_s_batch, indices)                # (B_sync, 32, 16, 64, 64)
        x_s = _to_dev(x_s_batch, indices)                # (B_sync, 21, 3)
        syncnet_A = _to_dev(A_batch, indices)             # (B_sync, 512)
        sim_gt = _to_dev(sim_gt_batch, indices)           # (B_sync,)

        # Compute lip-sync loss (with debug outputs if requested)
        if debug:
            debug_dir = getattr(self, 'lip_debug_path', None)
            if debug_dir is None:
                debug_dir = os.path.join(
                    getattr(self, 'experiment_path', '/tmp'), 'lip_debug'
                )
            os.makedirs(debug_dir, exist_ok=True)
            l_sync, l_stable, sim_pred = self.lip_sync_module(
                pred_windows, kp_canonical, f_s, x_s, syncnet_A, sim_gt,
                debug=True, debug_dir=debug_dir,
                debug_step=self.global_step,
            )
        else:
            l_sync, l_stable, sim_pred = self.lip_sync_module(
                pred_windows, kp_canonical, f_s, x_s, syncnet_A, sim_gt
            )

        if debug:
            print(f"  [DEBUG] l_sync: {float(l_sync):.6f}")
            print(f"  [DEBUG] l_stable: {float(l_stable):.6f}")
            print(f"  [DEBUG] sim_pred: {float(sim_pred):.6f}")
            print(f"  [DEBUG] sim_gt: {sim_gt.mean():.6f} ± {sim_gt.std():.6f}")
            print(f"  [DEBUG] l_sync requires_grad: {l_sync.requires_grad}")

        # Weighted combination (L_stable disabled by default)
        lip_loss = opt.lip_sync_lambda1 * l_sync
        if opt.lip_sync_use_stable:
            lip_loss = lip_loss + opt.lip_sync_lambda2 * l_stable

        lip_loss_dict = {
            'l_sync': float(l_sync),
            'l_stable': float(l_stable),
            'sim_pred': float(sim_pred),
            'lip_total': float(lip_loss),
        }

        return lip_loss, lip_loss_dict

    def _loss_backward(self, loss):
        self.optim.zero_grad()

        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        self.optim.step()

    def _train_one_step(self, data_dict):
        x = data_dict["kp_seq"]             # (B, L, kp_dim)
        cond_frame = data_dict["kp_cond"]   # (B, kp_dim)
        cond = data_dict["aud_cond"]        # (B, L, aud_dim)

        if not self.opt.use_accelerate:
            x = x.to(self.device)
            cond_frame = cond_frame.to(self.device)
            cond = cond.to(self.device)

        loss, loss_dict, x_recon = self.LMDM.diffusion(
            x, cond_frame, cond, t_override=None
        )

        # ── Lip-sync loss (optional, weak regularizer) ─────────────────────
        # Print conditions once to verify config is loaded
        if (self.opt.use_lip_sync_loss and self.is_main_process
                and self.global_step == 1):
            print(f"\n[LipSync Config Check]")
            print(f"  use_lip_sync_loss: {self.opt.use_lip_sync_loss}")
            print(f"  lip_sync_debug: {self.opt.lip_sync_debug}")
            print(f"  lip_sync_warmup_epochs: {self.opt.lip_sync_warmup_epochs}")
            print(f"  lip_sync_every_n_steps: {self.opt.lip_sync_every_n_steps}")
            print(f"  current epoch: {self.epoch}")
            print(f"  lip_debug_path: {getattr(self, 'lip_debug_path', 'NOT SET')}")
            import sys; sys.stdout.flush()

        if (self.opt.use_lip_sync_loss
                and self.epoch > self.opt.lip_sync_warmup_epochs
                and self.global_step % self.opt.lip_sync_every_n_steps == 0):
            try:
                # Debug every 10 lip-sync steps for fast feedback
                is_debug = (self.opt.lip_sync_debug and
                            self.is_main_process and
                            self.global_step % 10 == 0)

                if is_debug:
                    import sys
                    print(f"\n[LipSync Debug] Step {self.global_step}, Epoch {self.epoch}", flush=True)
                    print(f"  x_recon shape: {x_recon.shape}")
                    print(f"  x_recon requires_grad: {x_recon.requires_grad}")
                    print(f"  x_recon range: [{x_recon.min():.4f}, {x_recon.max():.4f}]")
                    debug_dir = getattr(self, 'lip_debug_path', None)
                    if debug_dir is None:
                        debug_dir = os.path.join(self.experiment_path, 'lip_debug')
                        self.lip_debug_path = debug_dir
                    os.makedirs(debug_dir, exist_ok=True)
                    print(f"  debug images path: {debug_dir}")
                    sys.stdout.flush()

                lip_loss, lip_loss_dict = self._compute_lip_sync_loss(
                    x_recon, data_dict, debug=is_debug
                )
                if lip_loss is not None:
                    if is_debug:
                        print(f"  lip_loss value: {lip_loss.item():.6f}")
                        print(f"  lip_loss requires_grad: {lip_loss.requires_grad}")

                    # Gradient clipping: detach if lip loss is too large
                    if lip_loss.item() > 2.0:
                        lip_loss = lip_loss.detach()  # skip this step
                        lip_loss_dict['lip_clipped'] = 1.0
                    else:
                        loss = loss + lip_loss
                    loss_dict.update(lip_loss_dict)
            except Exception:
                # Don't crash training if lip-sync fails on a batch
                if self.is_main_process and self.global_step % 100 == 0:
                    traceback.print_exc()

        return loss, loss_dict

    def _train_one_epoch(self):
        data_loader = self.data_loader

        DAM = DictAverageMeter()

        self.LMDM.train()
        self.local_step = 0
        for data_dict in tqdm(data_loader, disable=not self.is_main_process):
            self.global_step += 1
            self.local_step += 1

            loss, loss_dict = self._train_one_step(data_dict)
            self._loss_backward(loss)

            if self.is_main_process:
                loss_dict['total_loss'] = loss
                loss_dict_val = {k: float(v) for k, v in loss_dict.items()}
                DAM.update(loss_dict_val)

        return DAM

    def _show_and_save(self, DAM: DictAverageMeter):
        if not self.is_main_process:
            return
        
        self.LMDM.eval()

        epoch = self.epoch

        # show all loss
        avg_loss_msg = "|"
        avg = DAM.average()
        if avg is not None:
            for k, v in avg.items():
                avg_loss_msg += " %s: %.6f |" % (k, v)
        else:
            avg_loss_msg += " NO DATA |"
        msg = f'Epoch: {epoch}, Global_Steps: {self.global_step}, {avg_loss_msg}'
        print(msg, file=self.loss_logger)
        self.loss_logger.flush()

        # save model
        if self.accelerator is not None:
            state_dict = self.accelerator.unwrap_model(self.LMDM.model).state_dict()
        else:
            state_dict = self.LMDM.model.state_dict()

        ckpt = {
            "model_state_dict": state_dict,
        }
        ckpt_p = os.path.join(self.ckpt_path, f"train_{epoch}.pt")
        torch.save(ckpt, ckpt_p)
        tqdm.write(f"[MODEL SAVED at Epoch {epoch}] ({len(self.ckpt_file_list_for_clear)})")

        # ── Save best model as lmdm_v0.4_hubert.pth ──────────────────
        if avg is not None:
            total_loss = sum(v for k, v in avg.items() if k != 'sim_pred')
            if total_loss < self.best_loss:
                self.best_loss = total_loss
                best_path = os.path.join(self.ckpt_path, "lmdm_v0.4_hubert.pth")
                torch.save({"model_state_dict": state_dict}, best_path)
                tqdm.write(f"[BEST MODEL] Epoch {epoch}, loss={total_loss:.6f} → {best_path}")

        # clear model
        if epoch % self.opt.save_ckpt_freq != 0:
            self.ckpt_file_list_for_clear.append(ckpt_p)

        if len(self.ckpt_file_list_for_clear) > 5:
            _ckpt = self.ckpt_file_list_for_clear.pop(0)
            try:
                os.remove(_ckpt)
            except:
                traceback.print_exc()
                self.ckpt_file_list_for_clear.insert(0, _ckpt)

    def _train_loop(self):
        print(time.asctime(), 'start ...')

        opt = self.opt

        start_epoch = 1
        self.global_step = 0
        self.local_step = 0
        for epoch in trange(start_epoch, opt.epochs + 1, disable=not self.is_main_process):
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            self.epoch = epoch
            DAM = self._train_one_epoch()

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            if self.is_main_process:
                self.LMDM.eval()
                self._show_and_save(DAM)

        print(time.asctime(), 'done.')

    def train_loop(self):
        try:
            self._train_loop()
        except:
            msg = traceback.format_exc()
            error_msg = f'{time.asctime()} \n {msg} \n'
            print(error_msg)
            t = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            logname = f'{t}_rank{self.process_index}_error.log'
            os.makedirs(self.error_log_path, exist_ok=True)
            errorfile = os.path.join(self.error_log_path, logname)
            with open(errorfile, 'a') as f:
                f.write(error_msg)
            print(f'error msg write into {errorfile}')
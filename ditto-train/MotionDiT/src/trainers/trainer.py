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


def _linear_ramp(epoch: int, warmup_epochs: int, final_val: float) -> float:
    """Linear ramp from 0 to final_val over warmup_epochs epochs."""
    if warmup_epochs <= 0:
        return final_val
    return min(1.0, epoch / warmup_epochs) * final_val


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

        if opt.use_lip_sync_loss:
            print(time.asctime(), '_init_lip_sync')
            self._init_lip_sync()

    def _init_accelerate(self):
        opt = self.opt
        if opt.use_accelerate:
            from accelerate import Accelerator
            self.accelerator       = Accelerator()
            self.device            = self.accelerator.device
            self.is_main_process   = self.accelerator.is_main_process
            self.process_index     = self.accelerator.process_index
        else:
            self.accelerator       = None
            self.device            = 'cuda'
            self.is_main_process   = True
            self.process_index     = 0

    def _set_accelerate(self):
        if self.accelerator is None:
            return
        self.LMDM.use_accelerator(self.accelerator)
        self.optim       = self.accelerator.prepare(self.optim)
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
            part_w_dict=part_w_dict,
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
            lip_sync_num_frames=opt.lip_sync_num_frames,
            use_lip_landmark_loss=opt.use_lip_landmark_loss,
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
        return Adan(self.LMDM.model.parameters(), lr=opt.lr, weight_decay=0.02)

    def _init_log(self):
        opt = self.opt
        experiment_path      = os.path.join(opt.experiment_dir, opt.experiment_name)
        self.experiment_path = experiment_path
        self.error_log_path  = os.path.join(experiment_path, 'error')

        if not self.is_main_process:
            return

        self.ckpt_path = os.path.join(experiment_path, 'ckpts')
        os.makedirs(self.ckpt_path, exist_ok=True)

        self.lip_debug_path = os.path.join(experiment_path, 'lip_debug')
        if opt.use_lip_sync_loss and opt.lip_sync_debug:
            os.makedirs(self.lip_debug_path, exist_ok=True)
            print(f"[LipSync Debug] Images → {self.lip_debug_path}")

        opt_pkl = os.path.join(experiment_path, 'opt.pkl')
        dump_pkl(vars(opt), opt_pkl)

        loss_log         = os.path.join(experiment_path, 'loss.log')
        self.loss_logger = open(loss_log, 'a')

        self.ckpt_file_list_for_clear = []
        self.best_loss                = float('inf')

    def _init_lip_sync(self):
        """Initialize frozen SyncNet + renderer for lip-sync loss."""
        import os as _os
        from ..models.lip_sync_loss import LipSyncLoss

        opt         = self.opt
        warp_ckpt   = _os.path.join(opt.ditto_pytorch_path, 'models', 'warp_network.pth')
        decoder_ckpt = _os.path.join(opt.ditto_pytorch_path, 'models', 'decoder.pth')

        assert _os.path.isfile(opt.syncnet_checkpoint), \
            f"SyncNet checkpoint not found: {opt.syncnet_checkpoint}"
        assert _os.path.isfile(warp_ckpt), \
            f"WarpingNetwork checkpoint not found: {warp_ckpt}"
        assert _os.path.isfile(decoder_ckpt), \
            f"SPADEDecoder checkpoint not found: {decoder_ckpt}"

        max_shift = opt.lip_sync_max_shift if opt.lip_sync_use_delay_aware else 0

        self.lip_sync_module = LipSyncLoss(
            syncnet_ckpt=opt.syncnet_checkpoint,
            warp_ckpt=warp_ckpt,
            decoder_ckpt=decoder_ckpt,
            device=self.device,
            num_frames=opt.lip_sync_num_frames,
            max_shift=max_shift,
        )
        # Note: FrozenRenderer is intentionally kept on CPU inside LipSyncLoss
        # to avoid consuming GPU VRAM. Rendered frames (~4MB) are moved to GPU
        # after rendering. No .half() needed — CPU fp16 is slower, not faster.
        print(f"[LipSync] FrozenRenderer on CPU (GPU VRAM preserved for training).")
        print(f"[LipSync] λ1={opt.lip_sync_lambda1}  λ2={opt.lip_sync_lambda2}  "
              f"delay_λ={opt.lip_sync_delay_lambda}  "
              f"hard_mode={opt.lip_sync_hard_mode}  "
              f"num_frames={opt.lip_sync_num_frames}  max_shift={max_shift}")


    # -----------------------------------------------------------------------
    # Lip-sync loss computation
    # -----------------------------------------------------------------------

    def _compute_lip_sync_loss(self, x_recon, data_dict, epoch: int, debug: bool = False):
        """
        Compute lip-sync loss on a random subset of the batch.

        Returns (lip_loss_tensor, lip_loss_dict) or (None, {}).
        """
        import random as _random

        opt = self.opt
        B   = x_recon.shape[0]
        L   = x_recon.shape[1]    # sequence length
        T   = opt.lip_sync_num_frames
        max_shift = opt.lip_sync_max_shift if opt.lip_sync_use_delay_aware else 0

        # — Effective lambdas after warmup ramps —
        eff_lambda1 = _linear_ramp(
            epoch - opt.lip_sync_warmup_epochs,
            opt.lip_sync_lambda1_warmup_epochs,
            opt.lip_sync_lambda1_final if opt.lip_sync_lambda1_final > 0
            else opt.lip_sync_lambda1,
        )
        eff_delay_lambda = _linear_ramp(
            epoch - opt.lip_sync_warmup_epochs,
            opt.lip_sync_delay_warmup_epochs,
            opt.lip_sync_delay_lambda,
        ) if opt.lip_sync_use_delay_aware else 0.0

        eff_lmk_lambda = _linear_ramp(
            epoch - opt.lip_sync_warmup_epochs,
            opt.lip_landmark_warmup_epochs,
            opt.lip_landmark_lambda,
        ) if opt.use_lip_landmark_loss else 0.0

        if debug:
            print(f"  [Warmup] eff_λ1={eff_lambda1:.4f}  "
                  f"eff_delay_λ={eff_delay_lambda:.4f}  "
                  f"eff_lmk_λ={eff_lmk_lambda:.4f}")

        # — Validity check —
        if 'lipsync_valid' not in data_dict:
            return None, {}
        valid_mask    = data_dict['lipsync_valid']
        valid_indices = [i for i in range(B) if valid_mask[i]]
        if not valid_indices:
            if debug:
                print("  [DEBUG] No valid lip-sync samples in batch!")
            return None, {}

        B_sync  = min(opt.lip_sync_batch_size, len(valid_indices))
        indices = (_random.sample(valid_indices, B_sync)
                   if B_sync < len(valid_indices) else valid_indices)

        # — Extract extended motion windows (T + 2*max_shift frames) —
        t_starts   = data_dict['lipsync_t_start']
        pred_windows = []
        for i in indices:
            ts      = int(t_starts[i])
            ext_s   = max(0, ts - max_shift)
            ext_e   = min(L, ts + T + max_shift)
            window  = x_recon[i, ext_s:ext_e]             # may be short at edges

            # Pad if at sequence boundaries
            need_pre  = max_shift - (ts - ext_s)
            need_post = max_shift - ((ts + T + max_shift) - ext_e)
            if need_pre > 0:
                window = torch.cat(
                    [window[:1].expand(need_pre, -1), window], dim=0)
            if need_post > 0:
                window = torch.cat(
                    [window, window[-1:].expand(need_post, -1)], dim=0)

            target_len = T + 2 * max_shift
            # Final safety clamp
            if window.shape[0] < target_len:
                window = torch.cat(
                    [window,
                     window[-1:].expand(target_len - window.shape[0], -1)], dim=0)
            elif window.shape[0] > target_len:
                window = window[:target_len]

            pred_windows.append(window)  # (T_ext, 265)

        pred_windows = torch.stack(pred_windows)  # (B_sync, T_ext, 265)

        if debug:
            print(f"  [DEBUG] pred_windows shape: {pred_windows.shape}, "
                  f"requires_grad: {pred_windows.requires_grad}")

        # — Move precomputed features to device —
        def _to_dev(t, idx):
            if isinstance(idx, list):
                v = torch.stack([t[i] for i in idx])
            else:
                v = t[idx:idx + 1]
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            return v.to(self.device, dtype=torch.float32)

        kp_canonical = _to_dev(data_dict['lipsync_kp_canonical'], indices)
        f_s          = _to_dev(data_dict['lipsync_f_s'],          indices)
        x_s          = _to_dev(data_dict['lipsync_x_s'],          indices)
        syncnet_A    = _to_dev(data_dict['lipsync_A'],             indices)
        sim_gt       = _to_dev(data_dict['lipsync_sim_gt'],        indices)

        # Free unused GPU memory before the expensive renderer forward pass
        torch.cuda.empty_cache()

        # Optional landmark window
        lmk_window = None
        if opt.use_lip_landmark_loss and 'lipsync_lmk_window' in data_dict:
            lmk_window = _to_dev(data_dict['lipsync_lmk_window'], indices)

        # — Call lip-sync module —
        debug_dir  = getattr(self, 'lip_debug_path', None)
        (l_sync, l_stable, delay_penalty,
         sim_pred, best_shift, hard_weight, lmk_loss) = self.lip_sync_module(
            pred_windows, kp_canonical, f_s, x_s, syncnet_A, sim_gt,
            hard_mode=opt.lip_sync_hard_mode,
            hard_cap=opt.lip_sync_hard_cap,
            hard_min_weight=opt.lip_sync_hard_min_weight,
            use_delay_aware=opt.lip_sync_use_delay_aware,
            delay_mode=opt.lip_sync_delay_mode,
            lmk_window=lmk_window,
            debug=debug,
            debug_dir=debug_dir,
            debug_step=self.global_step,
        )

        if debug:
            print(f"  [DEBUG] l_sync={float(l_sync):.6f}  l_stable={float(l_stable):.6f}  "
                  f"delay_penalty={float(delay_penalty):.6f}  "
                  f"best_shift={best_shift:.2f}  hard_weight={hard_weight:.4f}")

        # — Legacy detach path (opt-in only) —
        if opt.legacy_detach and float(l_sync) > opt.lip_sync_hard_cap:
            return None, {'lip_clipped_legacy': 1.0}

        # — Weighted loss combination —
        lip_loss = eff_lambda1 * hard_weight * l_sync

        if opt.lip_sync_use_stable:
            lip_loss = lip_loss + opt.lip_sync_lambda2 * l_stable

        if opt.lip_sync_use_delay_aware and eff_delay_lambda > 0:
            lip_loss = lip_loss + eff_delay_lambda * delay_penalty

        if opt.use_lip_landmark_loss and eff_lmk_lambda > 0:
            lip_loss = lip_loss + eff_lmk_lambda * lmk_loss

        lip_loss_dict = {
            'l_sync':            float(l_sync),
            'l_stable':          float(l_stable),
            'delay_penalty':     float(delay_penalty),
            'best_shift':        float(best_shift),
            'hard_weight':       float(hard_weight),
            'landmark_loss':     float(lmk_loss),
            'sim_pred':          float(sim_pred),
            'lip_total':         float(lip_loss),
            'eff_lambda1':       eff_lambda1,
            'eff_delay_lambda':  eff_delay_lambda,
            'eff_lmk_lambda':    eff_lmk_lambda,
        }
        return lip_loss, lip_loss_dict

    # -----------------------------------------------------------------------

    def _loss_backward(self, loss):
        self.optim.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        self.optim.step()

    def _train_one_step(self, data_dict):
        x           = data_dict["kp_seq"]
        cond_frame  = data_dict["kp_cond"]
        cond        = data_dict["aud_cond"]

        if not self.opt.use_accelerate:
            x          = x.to(self.device)
            cond_frame = cond_frame.to(self.device)
            cond       = cond.to(self.device)

        loss, loss_dict, x_recon = self.LMDM.diffusion(
            x, cond_frame, cond, t_override=None
        )

        # — Config check (printed once) —
        if (self.opt.use_lip_sync_loss and self.is_main_process
                and self.global_step == 1):
            print(f"\n[LipSync Config]  warmup={self.opt.lip_sync_warmup_epochs}  "
                  f"every={self.opt.lip_sync_every_n_steps}  "
                  f"num_frames={self.opt.lip_sync_num_frames}  "
                  f"hard_mode={self.opt.lip_sync_hard_mode}  "
                  f"delay_aware={self.opt.lip_sync_use_delay_aware}  "
                  f"lmk_loss={self.opt.use_lip_landmark_loss}")
            import sys; sys.stdout.flush()

        in_warmup = (self.epoch <= self.opt.lip_sync_warmup_epochs)
        compute_lip = (
            self.opt.use_lip_sync_loss
            and not in_warmup
            and self.global_step % self.opt.lip_sync_every_n_steps == 0
        )

        if compute_lip:
            try:
                is_debug = (self.opt.lip_sync_debug
                            and self.is_main_process
                            and self.global_step % 10 == 0)

                if is_debug:
                    print(f"\n[LipSync] Step {self.global_step}  Epoch {self.epoch}", flush=True)

                lip_loss, lip_loss_dict = self._compute_lip_sync_loss(
                    x_recon, data_dict, epoch=self.epoch, debug=is_debug
                )
                if lip_loss is not None:
                    loss = loss + lip_loss
                loss_dict.update(lip_loss_dict)

            except Exception:
                if self.is_main_process and self.global_step % 100 == 0:
                    traceback.print_exc()

        return loss, loss_dict

    def _train_one_epoch(self):
        DAM             = DictAverageMeter()
        self.LMDM.train()
        self.local_step = 0

        for data_dict in tqdm(self.data_loader, disable=not self.is_main_process):
            self.global_step += 1
            self.local_step  += 1

            loss, loss_dict = self._train_one_step(data_dict)
            self._loss_backward(loss)

            if self.is_main_process:
                loss_dict['total_loss'] = loss
                DAM.update({k: float(v) for k, v in loss_dict.items()})

        return DAM

    def _show_and_save(self, DAM: DictAverageMeter):
        if not self.is_main_process:
            return

        self.LMDM.eval()
        epoch = self.epoch

        avg           = DAM.average()
        avg_loss_msg  = "|"
        if avg:
            for k, v in avg.items():
                avg_loss_msg += " %s: %.6f |" % (k, v)
        else:
            avg_loss_msg += " NO DATA |"

        msg = f'Epoch: {epoch}, Global_Steps: {self.global_step}, {avg_loss_msg}'
        print(msg, file=self.loss_logger)
        self.loss_logger.flush()

        if self.accelerator is not None:
            state_dict = self.accelerator.unwrap_model(self.LMDM.model).state_dict()
        else:
            state_dict = self.LMDM.model.state_dict()

        ckpt   = {"model_state_dict": state_dict}
        ckpt_p = os.path.join(self.ckpt_path, f"train_{epoch}.pt")
        torch.save(ckpt, ckpt_p)
        tqdm.write(f"[MODEL SAVED  Epoch {epoch}]")

        if avg:
            skip_keys   = {'sim_pred', 'best_shift', 'hard_weight',
                           'eff_lambda1', 'eff_delay_lambda', 'eff_lmk_lambda'}
            total_loss  = sum(v for k, v in avg.items() if k not in skip_keys)
            if total_loss < self.best_loss:
                self.best_loss = total_loss
                best_path      = os.path.join(self.ckpt_path, "lmdm_v0.4_hubert.pth")
                torch.save({"model_state_dict": state_dict}, best_path)
                tqdm.write(f"[BEST MODEL]  Epoch {epoch}  loss={total_loss:.6f}  → {best_path}")

        if epoch % self.opt.save_ckpt_freq != 0:
            self.ckpt_file_list_for_clear.append(ckpt_p)

        if len(self.ckpt_file_list_for_clear) > 5:
            _ckpt = self.ckpt_file_list_for_clear.pop(0)
            try:
                os.remove(_ckpt)
            except Exception:
                traceback.print_exc()
                self.ckpt_file_list_for_clear.insert(0, _ckpt)

    def _train_loop(self):
        print(time.asctime(), 'start ...')
        opt         = self.opt
        start_epoch = 1
        self.global_step = 0
        self.local_step  = 0

        for epoch in trange(start_epoch, opt.epochs + 1, disable=not self.is_main_process):
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            self.epoch = epoch
            DAM        = self._train_one_epoch()

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            if self.is_main_process:
                self.LMDM.eval()
                self._show_and_save(DAM)

        print(time.asctime(), 'done.')

    def train_loop(self):
        try:
            self._train_loop()
        except Exception:
            msg       = traceback.format_exc()
            error_msg = f'{time.asctime()} \n {msg} \n'
            print(error_msg)
            t         = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            logname   = f'{t}_rank{self.process_index}_error.log'
            os.makedirs(self.error_log_path, exist_ok=True)
            errorfile = os.path.join(self.error_log_path, logname)
            with open(errorfile, 'a') as f:
                f.write(error_msg)
            print(f'error msg write into {errorfile}')
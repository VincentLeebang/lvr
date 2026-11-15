import os

import torch
import torch.nn as nn
import wandb
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


class QwenLVRSFTTrainer(Trainer):

    def __init__(self, *args, temp_folder=None, oci_handler=None, **kwargs):
        super(QwenLVRSFTTrainer, self).__init__(*args, **kwargs)

        # Optional online checkpointing.
        if oci_handler:
            self.oci_handler = oci_handler
            self.temp_folder = temp_folder

    def create_optimizer(self):
        """
        Set up the optimizer, including optional separate learning rates for
        the vision tower, vision merger, and LVR head.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(
                opt_model, ALL_LAYERNORM_LAYERS
            )
            decay_parameters = [
                name for name in decay_parameters if "bias" not in name
            ]

            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []
            lvr_head_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "visual" in name and "merger" not in name
                ]

            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "merger" in name
                ]

            if self.args.lvr_head_lr is not None:
                lr_mapper["lvr_head"] = self.args.lvr_head_lr
                lvr_head_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "lvr_head" in name
                ]

            if len(lr_mapper) > 0:
                special_lr_parameters = (
                    merger_parameters
                    + visual_parameters
                    + lvr_head_parameters
                )

                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in special_lr_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in special_lr_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                ]

                if visual_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n in decay_parameters
                                        and n in visual_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in visual_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )

                if merger_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n in decay_parameters
                                        and n in merger_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in merger_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )

                if lvr_head_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n in decay_parameters
                                        and n in lvr_head_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.lvr_head_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in lvr_head_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.lvr_head_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if n in decay_parameters and p.requires_grad
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if n not in decay_parameters and p.requires_grad
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = (
                Trainer.get_optimizer_cls_and_kwargs(self.args)
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = (
                    bitsandbytes.optim.GlobalOptimManager.get_instance()
                )

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel()
                                for p in module.parameters()
                            }.values()
                        )
                        logger.info(
                            f"skipped {module}: {skipped / 2**20}M params"
                        )
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(
                            "bitsandbytes: will optimize this embedding in fp32"
                        )
                logger.info(f"skipped: {skipped / 2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        """
        Save model, optimizer, scheduler, RNG, trainer state, and optional
        remote checkpoint using the existing Trainer workflow.
        """
        checkpoint_folder = (
            f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        )

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if (
            self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH]
            and self.state.best_global_step
        ):
            best_checkpoint_folder = (
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
            )
            best_checkpoint_dir = os.path.join(
                run_dir, best_checkpoint_folder
            )

            if os.path.exists(best_checkpoint_dir):
                self.state.best_model_checkpoint = best_checkpoint_dir

        if not self.args.save_only_model:
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            self._save_rng_state(output_dir)

        if self.args.should_save:
            for cb in [
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state

            self.state.save_to_json(
                os.path.join(output_dir, TRAINER_STATE_NAME)
            )

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Upload the local checkpoint when online checkpointing is enabled.
        if self.temp_folder:
            remote_chkpt_folder = os.path.join(
                self.args.remote_output_dir, checkpoint_folder
            )
            if remote_chkpt_folder.startswith("/"):
                remote_chkpt_folder = remote_chkpt_folder[1:]

            self.oci_handler.save_checkpoint(
                output_dir, remote_chkpt_folder
            )
            self.temp_folder.cleanup(checkpoint_name=checkpoint_folder)

        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    def compute_loss(
        self,
        model,
        inputs,
        num_items_in_batch=None,
        return_outputs=False,
    ):
        """
        Combine cross-entropy, LVR, and optional mode-switch losses.

        Auxiliary losses are added only when they are present. If mode-switch
        loss is explicitly enabled but its required output is missing, raise
        a clear error rather than failing during tensor arithmetic.
        """
        if self.args.enable_data_packing:
            batch_size = inputs["input_ids"].size(0)
            total_tokens = inputs["input_ids"].numel()

            self.log(
                {
                    "batch_size": batch_size,
                    "tokens_per_device": total_tokens,
                }
            )

        outputs = model(**inputs)

        loss_ce = outputs.loss_ce
        loss_lvr = outputs.loss_lvr
        loss_mode_switch = outputs.loss_mode_switch

        if loss_ce is None:
            raise ValueError(
                "Cross-entropy loss (outputs.loss_ce) is None. "
                "Check the labels and the selected LVR forward function."
            )

        # Start with cross-entropy loss.
        loss = loss_ce

        # Add the LVR loss only when enabled and returned by the model.
        if self.args.loss_lvr_lambda > 0 and loss_lvr is not None:
            loss = loss + self.args.loss_lvr_lambda * loss_lvr

        # Add mode-switch loss only when the feature is enabled.
        if self.args.mode_switch_loss:
            if loss_mode_switch is None:
                raise ValueError(
                    "Mode-switch loss is enabled, but "
                    "outputs.loss_mode_switch is None."
                )

            loss = (
                loss
                + self.args.loss_mode_switch_lambda * loss_mode_switch
            )

        self.log(
            {
                "loss_total": loss.detach().item(),
                "loss_ce": loss_ce.detach().item(),
                "loss_lvr": (
                    loss_lvr.detach().item()
                    if loss_lvr is not None
                    else 0.0
                ),
                "loss_mode_switch": (
                    loss_mode_switch.detach().item()
                    if loss_mode_switch is not None
                    else 0.0
                ),
            }
        )

        return (loss, outputs) if return_outputs else loss

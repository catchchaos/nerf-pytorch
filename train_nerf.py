import argparse
import glob
import os
import time

import numpy as np
import torch
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_blender_data, load_llff_data, meshgrid_xy, models,
                  mse2psnr, run_one_iter_of_nerf)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    configargs = parser.parse_args()

    # Read config file.
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # Load dataset
    if cfg.dataset.type.lower() == "llff":
        images, poses, _, _, i_test = load_llff_data(
            cfg.dataset.basedir, factor=cfg.dataset.downsample_factor
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        if not isinstance(i_test, list):
            i_test = [i_test]
        if cfg.dataset.llffhold > 0:
            i_test = np.arange(images.shape[0])[:: cfg.dataset.llffhold]
        i_val = i_test
        i_train = np.array(
            [
                i
                for i in np.arange(images.shape[0])
                if (i not in i_test and i not in i_val)
            ]
        )
        H, W, focal = hwf
        H, W = int(H), int(W)
        images = torch.from_numpy(images)
        poses = torch.from_numpy(poses)
        USE_HR_LR = False

        # Load LR images
        if hasattr(cfg.dataset, "relative_lr_factor"):
            assert hasattr(cfg.dataset, "hr_fps")
            assert hasattr(cfg.dataset, "hr_frequency")
            USE_HR_LR = True
            images_lr, poses_lr, _, _, i_test = load_llff_data(
                cfg.dataset.basedir, factor=cfg.dataset.downsample_factor * cfg.dataset.relative_lr_factor
            )
            hwf = poses_lr[0, :3, -1]
            poses_lr = poses_lr[:, :3, :4]
            if not isinstance(i_test, list):
                i_test = [i_test]
            if cfg.dataset.llffhold > 0:
                i_test = np.arange(images_lr.shape[0])[:: cfg.dataset.llffhold]
            i_train_lr = np.array(
                [
                    i
                    for i in np.arange(images_lr.shape[0])
                    if (i not in i_test)
                ]
            )
            H_lr, W_lr, focal_lr = hwf
            H_lr, W_lr = int(H_lr), int(W_lr)
            images_lr = torch.from_numpy(images_lr)
            poses_lr = torch.from_numpy(poses_lr)

            # Expose only some HR images
            i_train = i_train[::cfg.dataset.hr_fps]

            print(f'LR summary: N={i_train_lr.shape[0]}, '
                  f'resolution={H_lr}x{W_lr}, focal-length={focal_lr}')
            print(f'HR summary: N={i_train.shape[0]}, '
                  f'resolution={H}x{W}, focal-length={focal}')

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    device = f"cuda:{cfg.experiment.gpu_id}"

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse and fine resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
    )
    model_coarse.to(device)

    model_fine = getattr(models, cfg.models.fine.type)(
        num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
        include_input_xyz=cfg.models.fine.include_input_xyz,
        include_input_dir=cfg.models.fine.include_input_dir,
        use_viewdirs=cfg.models.fine.use_viewdirs,
    )
    model_fine.to(device)

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters()) + list(model_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    # Write out config parameters.
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_iter = 0

    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]

    # # TODO: Prepare raybatch tensor if batching random rays

    for i in trange(start_iter, cfg.experiment.train_iters):

        model_coarse.train()
        model_fine.train()

        if USE_HR_LR and i % cfg.dataset.hr_frequency != 0:
            H_iter, W_iter, focal_iter = H_lr, W_lr, focal_lr
            i_train_iter = i_train_lr
            images_iter, poses_iter = images_lr, poses_lr
        else:
            H_iter, W_iter, focal_iter = H, W, focal
            i_train_iter = i_train
            images_iter, poses_iter = images, poses

        img_idx = np.random.choice(i_train_iter)
        img_target = images_iter[img_idx].to(device)
        pose_target = poses_iter[img_idx, :3, :4].to(device)
        ray_origins, ray_directions = get_ray_bundle(H_iter, W_iter, focal_iter, pose_target)
        coords = torch.stack(
            meshgrid_xy(torch.arange(H_iter).to(device), torch.arange(W_iter).to(device)),
            dim=-1,
        )
        coords = coords.reshape((-1, 2))
        select_inds = np.random.choice(
            coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
        )
        select_inds = coords[select_inds]
        ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
        ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
        target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]

        rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
            H_iter,
            W_iter,
            focal_iter,
            model_coarse,
            model_fine,
            ray_origins,
            ray_directions,
            cfg,
            mode="train",
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
        )
        target_ray_values = target_s

        coarse_loss = torch.nn.functional.mse_loss(
            rgb_coarse[..., :3], target_ray_values[..., :3]
        )
        fine_loss = torch.nn.functional.mse_loss(
            rgb_fine[..., :3], target_ray_values[..., :3]
        )
        loss = coarse_loss + fine_loss
        loss.backward()
        psnr = mse2psnr(loss.item())
        optimizer.step()
        optimizer.zero_grad()

        # Learning rate updates
        num_decay_steps = cfg.scheduler.lr_decay * 1000
        lr_new = cfg.optimizer.lr * (
            cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write(
                "[TRAIN] Iter: "
                + str(i)
                + " Loss: "
                + str(loss.item())
                + " PSNR: "
                + str(psnr)
            )
        writer.add_scalar("train/loss", loss.item(), i)
        writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        writer.add_scalar("train/fine_loss", fine_loss.item(), i)
        writer.add_scalar("train/psnr", psnr, i)

        # Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1
        ):
            tqdm.write("[VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            model_fine.eval()

            start = time.time()
            with torch.no_grad():
                img_idx = np.random.choice(i_val)
                img_target = images[img_idx].to(device)
                pose_target = poses[img_idx, :3, :4].to(device)
                ray_origins, ray_directions = get_ray_bundle(
                    H, W, focal, pose_target
                )
                rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                    H,
                    W,
                    focal,
                    model_coarse,
                    model_fine,
                    ray_origins,
                    ray_directions,
                    cfg,
                    mode="validation",
                    encode_position_fn=encode_position_fn,
                    encode_direction_fn=encode_direction_fn,
                )
                target_ray_values = img_target
                coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                loss = coarse_loss + fine_loss
                psnr = mse2psnr(loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
                writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                writer.add_scalar("validation/fine_loss", fine_loss.item(), i)
                writer.add_scalar("validataion/psnr", psnr, i)
                writer.add_image(
                    "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i
                )
                writer.add_image(
                    "validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i
                )
                writer.add_image(
                    "validation/img_target",
                    cast_to_image(target_ray_values[..., :3]),
                    i,
                )
                tqdm.write(
                    "Validation loss: "
                    + str(loss.item())
                    + " Validation PSNR: "
                    + str(psnr)
                    + " Time: "
                    + str(time.time() - start)
                )

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
            }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")

    print("Done!")


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


if __name__ == "__main__":
    main()
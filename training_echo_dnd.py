# training_echo_dnd.py
import argparse
from guided_diffusion import logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.echo_dnd_dataset import EchoDNDDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
from torchvision import transforms
from guided_diffusion.utils import get_transform_train

def main():
    """Main function to run Echo-DND training."""
    print("GPU AVAILABLE:", th.cuda.is_available())
    args = create_argparser().parse_args()
    print("Arguments: ", args)
    
    logger.configure(dir=args.out_dir)

    logger.log("Creating Echo-DND data loader...")
    

    image_transforms_list = get_transform_train(args=args, augmentation=True)
    args.in_channels = 2

    ds = EchoDNDDataset(
        data_dir=args.data_dir,
        image_size=(args.image_size, args.image_size),
        split=args.dataset_split,
        common_transforms=image_transforms_list
    )
        
    print(f"Number of samples in '{args.dataset_split}' split: {len(ds)}")
    if len(ds) == 0:
        logger.log(f"Error: No data found for split '{args.dataset_split}'. Please check data_dir and custom_dataset.py implementation.")
        return
    
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers # Added num_workers
    )
    
    data_iter = iter(datal)

    logger.log("Creating Echo-DND model and diffusion process...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print(f"Number of trainable parameters (Millions): {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    
    if args.pretrained_path is not None:
        logger.log(f"Loading pre-trained weights from: {args.pretrained_path}")
        state_dict = th.load(args.pretrained_path, map_location="cpu")
        from collections import OrderedDict # Keep import local if only used here
        new_state_dict = OrderedDict()
        # Handle 'module.' prefix from DataParallel training
        has_module_prefix = any('module.' in k for k in state_dict.keys())
        if has_module_prefix:
            for k, v in state_dict.items():
                if 'module.' in k:
                    new_state_dict[k[7:]] = v 
                else:
                    new_state_dict[k] = v
        else:
            new_state_dict = state_dict
        model.load_state_dict(new_state_dict)
        logger.log("Successfully loaded pre-trained weights.")
    
    model.to(args.device)
    if args.multi_gpu:
        try:
            devices_arr = [int(id.strip()) for id in args.multi_gpu.split(',')]
            logger.log(f"Using multiple GPUs: {devices_arr}")
            model = th.nn.DataParallel(model, device_ids=devices_arr)
        except ValueError:
            logger.log(f"Error: Invalid multi_gpu format: {args.multi_gpu}. Expected comma-separated integers.")
            return
            
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Starting Echo-DND training loop...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None, # Echo-DND does not use a separate classifier for guidance in this setup
        data=data_iter,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        device=args.device,
    ).run_loop()

def create_argparser():
    """Creates and returns an ArgumentParser for Echo-DND training."""
    defaults = dict(
        data_dir="/path/to/your/dataset_root_dir", # User needs to set this
        dataset_split="train",                     # Specify 'train', 'val', etc.
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=1000, # Log more frequently
        save_interval=10000, # Save less frequently than logging
        resume_checkpoint=None,
        use_fp16=False, # Set to True if GPU supports and desired
        fp16_scale_growth=1e-3,
        device="cuda" if th.cuda.is_available() else "cpu",
        multi_gpu=None, # e.g., "0,1"
        pretrained_path=None, # Path to load pre-trained model weights
        out_dir='./results_echodnd_training', # Default output directory
        num_workers=4, # Number of workers for DataLoader
    )
    defaults.update(model_and_diffusion_defaults())
    # Ensure image_size is part of defaults if not in model_and_diffusion_defaults()
    if 'image_size' not in defaults:
        defaults['image_size'] = 256 # Default image size for Echo-DND

    parser = argparse.ArgumentParser(description="Echo-DND Training Script")
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
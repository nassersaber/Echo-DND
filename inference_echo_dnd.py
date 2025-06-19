import argparse
import os
from PIL import Image
import time
import random
import sys, cv2
import numpy as np
import torch as th
from guided_diffusion import logger
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from guided_diffusion.utils import get_transform_train

"""
Inference Script for Echo-DND: Dual Noise Diffusion Model.

This script performs segmentation of a single echocardiography image using a
pre-trained Echo-DND model. It handles image loading, preprocessing, model
inference via the dual-noise diffusion sampling loop, optional ensembling of
multiple predictions, STAPLE-based fusion of ensemble members, and
post-processing of the final segmentation mask.

For more details on Echo-DND, refer to the paper:
Rahman, A., Balraj, K., Ramteke, M., & Rathore, A. S. (2025).
Echo-DND: a dual noise diffusion model for robust and precise left ventricle
segmentation in echocardiography. Discover Applied Sciences, 7(514).
https://doi.org/10.1007/s42452-025-07055-5
"""

seed = 10
th.manual_seed(seed)
if th.cuda.is_available():
    th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    """Normalizes and prepares an image tensor for visualization or saving."""
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

from skimage import measure, morphology
def post_process_image(input_image_pil, threshold=0.8, disk_size=3):
    """
    Post-processes a predicted segmentation mask (PIL Image).
    Steps include:
    1. Normalization and Binarization based on a threshold.
    2. Identifying the largest connected component.
    3. Applying morphological erosion followed by dilation to smooth boundaries.
    """
    input_image = np.array(input_image)/255.0
    binary_image = input_image > threshold
    labels = measure.label(binary_image, connectivity=2)
    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    central_mask = (labels == largest_label)
    central_mask_smooth = morphology.binary_erosion(central_mask, morphology.disk(disk_size))
    processed_image = morphology.binary_dilation(central_mask_smooth, morphology.disk(disk_size))
    processed_image = (processed_image*255).astype(np.uint8)
    return Image.fromarray(processed_image)

# Inference main function
def main():
    """Main function to run Echo-DND inference on a single image."""
    args = create_argparser().parse_args()
    
    # Setup logger and output directory
    logger.configure(dir=args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Image Loading and Preprocessing ---
    image_path = args.image_path
    try:
        img_pil = Image.open(image_path, mode='r').convert('L') # Load as grayscale PIL Image
    except FileNotFoundError:
        logger.log(f"Error: Image not found at {image_path}")
        sys.exit(1)
    logger.log(f"Image loaded from: {image_path}")
    
    # Get transformation pipeline (resize and ToTensor)
    transform_pipeline = get_transform_train(args=args, augmentation=False) 
    transform_for_resizing, _, transform_for_converting_to_tensor = transform_pipeline
    
    # Apply resizing and tensor conversion
    # The processed_img is expected to be the conditional input for the model. Shape: [1 (batch_size), 1 (image_channels), H, W]
    processed_img_tensor = transform_for_converting_to_tensor(transform_for_resizing(img_pil)).unsqueeze(0) # Add batch dimension

    logger.log("Image preprocessed")
    
    if args.batch_size != 1:
        logger.log("ERROR - This inference script currently supports Batch Size 1 only!")
        sys.exit(1)

    # --- Model and Diffusion Setup ---
    logger.log("Creating Echo-DND model and diffusion process handler...")
    # `model` is EchoDNDUNet, `diffusion` is EchoDNDDiffusion
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    logger.log(f"Loading pre-trained Echo-DND model weights from: {args.model_path}")
    try:
        state_dict = th.load(args.model_path, map_location="cpu")
    except FileNotFoundError:
        logger.log(f"Error: Model weights not found at {args.model_path}")
        sys.exit(1)
        
    # Handle 'module.' prefix if the model was trained with DataParallel
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    has_module_prefix = any('module.' in k for k in state_dict.keys())
    if has_module_prefix:
        for k, v in state_dict.items():
            if 'module.' in k:
                new_state_dict[k[7:]] = v # remove `module.`
            else:
                # This case should ideally not happen if 'module.' is present in some keys
                new_state_dict[k] = v 
    else:
        new_state_dict = state_dict # Use as is if no 'module.' prefix

    model.load_state_dict(new_state_dict)
    model.to(args.device) # Move model to specified device (cuda or cpu)
    logger.log(f"Model loaded on device: {next(model.parameters()).device}")

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval() # Set model to evaluation mode

    # --- Sampling Process ---
    logger.log(f"Starting Echo-DND sampling with {args.num_ensemble} ensemble(s)...")
    
    enslist_gaussian = []
    enslist_bernoulli = []
    enslist_cal = [] # To store calibration maps from MFCM

    total_sampling_time = 0
    for i in range(args.num_ensemble):
        logger.log(f"Generating ensemble member {i+1}/{args.num_ensemble}...")
        model_kwargs = {} # Placeholder for any future model conditioning arguments
        
        start_time_sample = time.time()
        
        # Select sampling function based on whether DDIM is used (currently DDIM is not implemented in EchoDNDDiffusion)
        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )
        
        # Perform one full reverse diffusion process.
        # `processed_img_tensor` is the conditional image.
        # The `p_sample_loop_known` in EchoDNDDiffusion will internally create initial noise maps.
        sample_gaussian, sample_bernoulli, _, _, cal_map = sample_fn(
            model,                      # The EchoDNDUNet model
            processed_img_tensor,       # Preprocessed conditional input image tensor
            step=args.diffusion_steps,  # Number of diffusion steps for reverse process
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=args.device,
            progress=True               # Show progress bar for sampling steps
        )
        # sample_gaussian: Raw output from the Gaussian diffusion pathway (e.g., predicted noise or x0)
        # sample_bernoulli: Raw output from the Bernoulli diffusion pathway (e.g., predicted x0 logits/probs)
        # cal_map: Calibration map output from the MFCM

        sample_time = time.time() - start_time_sample
        total_sampling_time += sample_time
        logger.log(f"Time for generating ensemble member {i+1}: {sample_time:.2f}s")

        # Ensure single-channel output for segmentation masks
        assert sample_gaussian.shape[1] == 1, "Gaussian output should be single channel."
        assert sample_bernoulli.shape[1] == 1, "Bernoulli output should be single channel."
        assert cal_map.shape[1] == 1, "Calibration map should be single channel."

        enslist_gaussian.append(sample_gaussian.detach().cpu())
        enslist_bernoulli.append(sample_bernoulli.detach().cpu())
        enslist_cal.append(cal_map.detach().cpu())
    
    logger.log(f"Total sampling time for {args.num_ensemble} ensembles: {total_sampling_time:.2f}s")

    # --- Ensemble Fusion and Saving ---
    logger.log("Fusing ensemble members using STAPLE...")
    # STAPLE algorithm estimates a consensus segmentation from multiple inputs.
    # Squeeze(0) removes the ensemble dimension after stacking if batch size was 1.
    if args.num_ensemble > 0:
        ensres_gaussian = staple(th.stack(enslist_gaussian, dim=0)).squeeze(0) if enslist_gaussian else None
        ensres_bernoulli = staple(th.stack(enslist_bernoulli, dim=0)).squeeze(0) if enslist_bernoulli else None
        ensres_cal = staple(th.stack(enslist_cal, dim=0)).squeeze(0) if enslist_cal else None # Fused calibration map
    else: # Should not happen if num_ensemble >= 1
        ensres_gaussian, ensres_bernoulli, ensres_cal = None, None, None

    # Example strategies for combining the fused outputs from different pathways
    if ensres_gaussian is not None and ensres_bernoulli is not None and ensres_cal is not None:
        ensres_overall_avg = (ensres_gaussian + ensres_bernoulli + ensres_cal) / 3.0
        # STAPLE of already STAPLEd maps from different pathways
        ensres_overall_stack_staple = staple(th.stack([ensres_gaussian, ensres_bernoulli, ensres_cal], dim=0)).squeeze(0)
    else: # Handle cases where some components might be missing if num_ensemble was 0 or lists were empty
        logger.log("Warning: One or more ensemble lists were empty, cannot compute combined results.")
        ensres_overall_avg, ensres_overall_stack_staple = None, None
    
    # Generate a unique ID for saving this run's outputs
    from datetime import datetime
    # Using only one image per run, so sample_idx is effectively 0
    # If processing multiple images in future, slice_ID should be per image
    unique_id_prefix = os.path.splitext(os.path.basename(image_path))[0] # Use image name as prefix
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    slice_id = f"{unique_id_prefix}_{timestamp}" 
    logger.log(f"Generated Slice ID for saving: {slice_id}")

    # Save intermediate and final fused segmentation masks
    # Visualize function normalizes tensor to [0,1] for saving as image
    if ensres_gaussian is not None:
        vutils.save_image(visualize(ensres_gaussian), fp=os.path.join(args.out_dir, f"{slice_id}_ens_gaussian.png"), nrow=1)
    if ensres_bernoulli is not None:
        vutils.save_image(visualize(ensres_bernoulli), fp=os.path.join(args.out_dir, f"{slice_id}_ens_bernoulli.png"), nrow=1)
    if ensres_cal is not None:
        vutils.save_image(visualize(ensres_cal), fp=os.path.join(args.out_dir, f"{slice_id}_ens_cal.png"), nrow=1)
    if ensres_overall_avg is not None:
        vutils.save_image(visualize(ensres_overall_avg), fp=os.path.join(args.out_dir, f"{slice_id}_ens_avg.png"), nrow=1)
    if ensres_overall_stack_staple is not None:
        vutils.save_image(visualize(ensres_overall_stack_staple), fp=os.path.join(args.out_dir, f"{slice_id}_ens_stack_staple.png"), nrow=1)
    
    # Save original input image (ensure it's in a displayable format if not already)
    # Assuming processed_img_tensor is [1,1,H,W] and normalized [0,1] or [-1,1]
    # If it's normalized [-1,1], it should be shifted to [0,1] for saving: (tensor + 1) / 2
    # For simplicity, assuming it's already suitable or `visualize` handles it.
    vutils.save_image(visualize(processed_img_tensor.squeeze(0)), fp=os.path.join(args.out_dir, f"{slice_id}_input.png"), nrow=1)
    
    # --- Post-processing ---
    if ensres_overall_stack_staple is not None:
        logger.log("Applying post-processing to the STAPLEd consensus segmentation...")
        # Convert tensor to PIL Image for post_process_image function
        # Ensure ensres_overall_stack_staple is [H, W] or [1, H, W] and on CPU
        stapled_output_pil = transforms.ToPILImage()(visualize(ensres_overall_stack_staple.cpu().squeeze(0))) # Squeeze channel if present
        
        try:
            post_processed_pil = post_process_image(stapled_output_pil, threshold=args.post_process_threshold)
            logger.log("Post-processed Image Saved")
        except Exception as e:
            post_processed_pil = stapled_output_pil # Fallback to original STAPLEd output if post-processing fails
            logger.log(f"Post-processing Failed! Error: {e}. Saving non-post-processed version.")
        post_processed_pil.save(os.path.join(args.out_dir, f"{slice_id}_post_processed_stack_staple.png"))
    else:
        logger.log("Skipping post-processing as STAPLEd consensus segmentation is not available.")

    logger.log(f"Inference complete. Outputs saved in {args.out_dir}")

def create_argparser():
    """
    Creates and returns an ArgumentParser for the Echo-DND inference script.
    Sets up default command-line arguments for model paths, data, diffusion steps, etc.
    """
    defaults = dict(
        image_path='test.jpg',                  # Path to the input echocardiogram image
        data_name='CAMUS',                      # Dataset name context (primarily for reference/config)
        # data_dir: Not directly used in this single-image script, but often part of model_and_diffusion_defaults
        clip_denoised=True,                     # Whether to clip denoised model outputs
        # num_samples: Typically for generating multiple distinct samples from noise, not used here with p_sample_loop_known
        batch_size=1,                           # Batch size for inference (fixed to 1 in this script)
        use_ddim=False,                         # Whether to use DDIM sampler (EchoDNDDiffusion currently uses p_sample_loop_known)
        model_path="path/to/your/echodnd_model.pt", # <<< --- UPDATE THIS DEFAULT --- Path to pre-trained Echo-DND model
        num_ensemble=5,                         # Number of samples in the ensemble for robust prediction
        device="cuda" if th.cuda.is_available() else "cpu", # Device to run inference on
        # debug=False, # Not used in this script
        out_dir='results_echo_dnd_inference',     # Directory to save output segmentations
        post_process_threshold=0.5,             # Threshold for binarizing segmentation in post-processing
    )
    defaults.update(model_and_diffusion_defaults()) # Adds model and diffusion specific arguments
    parser = argparse.ArgumentParser(description="Echo-DND Inference Script")
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    # This allows the script to be run from the command line.
    # Example: python your_script_name.py --image_path /path/to/image.png --model_path /path/to/model.pt --out_dir ./outputs
    main()

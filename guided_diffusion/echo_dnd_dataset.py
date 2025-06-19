import os
import torch
import numpy as np
from PIL import Image
import SimpleITK as sitk # For CAMUS .mhd files
from torch.utils.data import Dataset

class EchoDNDDataset(Dataset):
    """
    Unified Dataset for Echo-DND, loading a combined dataset from
    predefined CAMUS and EchoNet-Dynamic subdirectories.

    This class expects a root data directory containing 'CAMUS' and
    'EchoNet-Dynamic' subfolders with their respective data.

    It returns a tuple of (image_tensor, mask_tensor, image_path)
    in the __getitem__ method.
    - image_tensor: The preprocessed conditional image (e.g., [1, H, W]).
    - mask_tensor: The ground truth segmentation mask (e.g., [1, H, W], binary 0 or 1).
    - image_path: The original path to the image file.
    """
    def __init__(self, data_root_dir, transform_tuple=None,
                 camus_split='train', echonet_split='Train', mode='Training'):
        """
        Initialize the combined dataset.

        :param data_root_dir: Path to the root directory containing 'CAMUS' and
                              'EchoNet-Dynamic' subfolders.
        :param transform_tuple: A tuple of three torchvision.transforms.Compose objects:
                                (transform_for_resizing,
                                 transform_for_image_augmentation,
                                 transform_for_converting_to_tensor_and_common_ops).
                                 This structure is expected from guided_diffusion.utils.get_transform_train.
        :param camus_split: String indicating the CAMUS split (e.g., 'train', 'val').
        :param echonet_split: String indicating the EchoNet split (e.g., 'Train', 'Val', 'Test').
        :param mode: String, typically 'Training' or 'Inference'.
        """
        self.data_dir_camus = os.path.join(data_root_dir, "CAMUS")
        self.data_dir_echonet = os.path.join(data_root_dir, "EchoNet-Dynamic")
        self.camus_split = camus_split
        self.echonet_split = echonet_split
        self.mode = mode

        self.all_samples = [] # List to store (image_path, mask_path, 'camus'/'echonet')

        self.transform_active = (transform_tuple is not None)
        self.transform_resizing = None
        self.transform_image_augment = None
        self.transform_common_to_tensor = None

        if self.transform_active:
            if isinstance(transform_tuple, tuple) and len(transform_tuple) == 3:
                self.transform_resizing = transform_tuple[0]
                self.transform_image_augment = transform_tuple[1]
                self.transform_common_to_tensor = transform_tuple[2]
            else:
                print("Warning: transform_tuple is not a tuple of 3 components. Transformations will not be applied correctly.")
                self.transform_active = False

        # Load file lists from both datasets
        self._load_camus_files()
        self._load_echonet_files()
        
        if not self.all_samples:
            print(f"Warning: No samples loaded. Check data directories '{self.data_dir_camus}' (split: {self.camus_split}) "
                  f"and '{self.data_dir_echonet}' (split: {self.echonet_split}).")

    def _load_camus_files(self):
        """Helper function to load file paths for the CAMUS dataset."""
        if not os.path.isdir(self.data_dir_camus):
            print(f"Warning: CAMUS data directory not found at {self.data_dir_camus}. Skipping CAMUS.")
            return

        try:
            patients_list_all = sorted([x for x in os.listdir(self.data_dir_camus)
                                    if (os.path.isdir(os.path.join(self.data_dir_camus, x))
                                        and x.startswith('patient'))])
        except FileNotFoundError:
            print(f"Error accessing CAMUS data directory at {self.data_dir_camus}")
            return

        current_patients_list = []
        if self.camus_split == 'train':
            current_patients_list = patients_list_all[:400]
        elif self.camus_split == 'val':
            current_patients_list = patients_list_all[400:450]
        else:
            print(f"Warning: CAMUS split '{self.camus_split}' not recognized or not implemented for specific slicing. Consider adding logic if this split is needed.")
            current_patients_list = []

        num_loaded_camus = 0
        for patient_unique_id in current_patients_list:
            patient_dir = os.path.join(self.data_dir_camus, patient_unique_id)
            # ED Frame
            ed_image_path = os.path.join(patient_dir, f"{patient_unique_id}_4CH_ED.mhd")
            ed_mask_path = os.path.join(patient_dir, f"{patient_unique_id}_4CH_ED_gt.mhd")
            if os.path.exists(ed_image_path) and os.path.exists(ed_mask_path):
                self.all_samples.append((ed_image_path, ed_mask_path, 'camus'))
                num_loaded_camus +=1
            
            # ES Frame
            es_image_path = os.path.join(patient_dir, f"{patient_unique_id}_4CH_ES.mhd")
            es_mask_path = os.path.join(patient_dir, f"{patient_unique_id}_4CH_ES_gt.mhd")
            if os.path.exists(es_image_path) and os.path.exists(es_mask_path):
                self.all_samples.append((es_image_path, es_mask_path, 'camus'))
                num_loaded_camus +=1
        print(f"Loaded {num_loaded_camus} CAMUS samples for split '{self.camus_split}'.")


    def _load_echonet_files(self):
        """Helper function to load file paths for the EchoNet-Dynamic dataset."""
        echonet_split_dir = os.path.join(self.data_dir_echonet, self.echonet_split)
        frames_dir = os.path.join(echonet_split_dir, 'Frames')
        masks_dir = os.path.join(echonet_split_dir, 'Masks')

        if not os.path.isdir(frames_dir) or not os.path.isdir(masks_dir):
            print(f"Warning: EchoNet Frames or Masks directory not found for split '{self.echonet_split}' at {echonet_split_dir}. Skipping EchoNet.")
            return

        num_loaded_echonet = 0
        try:
            image_filenames = [x for x in sorted(os.listdir(frames_dir)) if x.endswith('.png')]
            for img_fn in image_filenames:
                mask_fn = img_fn
                image_path = os.path.join(frames_dir, img_fn)
                mask_path = os.path.join(masks_dir, mask_fn)
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    self.all_samples.append((image_path, mask_path, 'echonet'))
                    num_loaded_echonet += 1
                else:
                    print(f"Warning: Missing image or mask for EchoNet sample: {img_fn} in {masks_dir}")
        except FileNotFoundError:
            print(f"Error accessing EchoNet Frames/Masks directory contents for split '{self.echonet_split}'.")
            return
        print(f"Loaded {num_loaded_echonet} EchoNet samples for split '{self.echonet_split}'.")

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        image_path, mask_path, dataset_type = self.all_samples[index]
        
        image_pil = None
        mask_pil = None

        try:
            if dataset_type == 'camus':
                image_np = sitk.GetArrayFromImage(sitk.ReadImage(image_path, sitk.sitkFloat32))[0,:,:]
                mask_np = sitk.GetArrayFromImage(sitk.ReadImage(mask_path, sitk.sitkFloat32))[0,:,:]
                mask_np[mask_np != 1] = 0
                mask_np[mask_np == 1] = 255
                image_pil = Image.fromarray(image_np.astype(np.uint8)).convert('L')
                mask_pil = Image.fromarray(mask_np.astype(np.uint8)).convert('L')
            elif dataset_type == 'echonet':
                image_pil = Image.open(image_path).convert('L')
                mask_pil = Image.open(mask_path).convert('L')
        except Exception as e:
            print(f"Error loading image or mask for index {index} ({dataset_type}) - Image path: {image_path}\nMask path: {mask_path}\nError: {e}")
            raise e

        if image_pil is None or mask_pil is None:
            print(f"Critical error: image_pil or mask_pil is None for {image_path} after loading.")
            raise ValueError("Image or mask could not be loaded properly.")

        # Apply transformations
        if self.transform_active:
            # Apply resizing to both image and mask (first transform component)
            image_pil = self.transform_resizing(image_pil)
            mask_pil = self.transform_resizing(mask_pil)
            
            # Apply image-only augmentations (second transform component)
            image_pil_augmented = self.transform_image_augment(image_pil)
            
            # Apply common transformations (third transform component)
            state = torch.get_rng_state()
            image_tensor = self.transform_common_to_tensor(image_pil_augmented)
            torch.set_rng_state(state)
            mask_tensor = self.transform_common_to_tensor(mask_pil)
        else:
            raise ValueError("No transformations provided. Please provide a valid transform tuple.")

        # Ensure mask is binary [0.0, 1.0]
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor, image_path
"""
Echo-DND: Dual Noise Diffusion Model for Left Ventricle Segmentation in Echocardiography

This script implements the core diffusion process for Echo-DND, a novel dual-noise
diffusion probabilistic model (DPM) designed for robust and precise segmentation
of the left ventricle (LV) in echocardiography images.

The model leverages a synergistic combination of Gaussian and Bernoulli noise
diffusion processes to effectively model the complex noise characteristics and
binary nature of segmentation tasks in medical imaging. Key components include:
- Dual Noise Strategy: Simultaneous handling of Gaussian noise (for continuous
  variations) and Bernoulli noise (for discrete mask elements).
- Independent Noise Estimation Modules (GNEM & BNEM): Separate neural network
  branches (implicitly handled by the 'model' passed to this class) are expected
  to predict parameters for each noise type.
- Multi-Scale Fusion Conditioning: Assumed to be part of the 'model' architecture,
  providing rich contextual features.
- Spatial Coherence Calibration: The training loss incorporates terms for this,
  and the sampling process can output calibration maps.

This implementation is based on foundational DPM principles and is adapted to
handle the dual-noise paradigm specific to Echo-DND. It supports both training
(calculating losses) and sampling (generating segmentation masks).

For more details, please refer to the original paper:
Rahman, A., Balraj, K., Ramteke, M., & Rathore, A. S. (2025).
Echo-DND: a dual noise diffusion model for robust and precise left ventricle
segmentation in echocardiography. Discover Applied Sciences, 7(514).
https://doi.org/10.1007/s42452-025-07055-5
"""


from torch.autograd import Variable
import enum
import torch.nn.functional as F
from torchvision.utils import save_image
import torch
import math
import os
# from visdom import Visdom
# viz = Visdom(port=8850)
import numpy as np
import torch as th
import torch.nn as nn
from tqdm.auto import tqdm
from guided_diffusion.train_util import visualize
from guided_diffusion.nn import mean_flat
from guided_diffusion.losses import normal_kl, discretized_gaussian_log_likelihood
from guided_diffusion.losses import binomial_kl, binomial_log_likelihood
from scipy import ndimage
from torchvision import transforms
from guided_diffusion.utils import staple, dice_score, norm
import torchvision.utils as vutils
from guided_diffusion.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
import string
import random

from torch.distributions.binomial import Binomial

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
    BCE_DICE = enum.auto()
    BCE = enum.auto()  # use raw BCE loss
    MIX = enum.auto()  # combine BCE loss and kl loss

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class EchoDNDDiffusion:
    """
    Implements the core dual-noise diffusion process for Echo-DND.

    This class handles both the forward (noising) and reverse (denoising)
    processes for the combined Gaussian and Bernoulli noise diffusion,
    as described in the Echo-DND paper. It calculates relevant distributions,
    manages noise schedules, and provides methods for training loss computation
    and mask generation via sampling.

    The 'model' argument passed during initialization is expected to be a neural
    network (e.g., a U-Net with appropriate modifications for dual output and
    conditioning via MFCM) that predicts the parameters for both the Gaussian
    and Bernoulli noise components.

    :param betas: A 1-D numpy array of betas (variance schedule) for each
                diffusion timestep.
    :param model_mean_type: A ModelMeanType (enum) determining what the Gaussian
                            component of the model outputs (e.g., EPSILON).
                            Note: Bernoulli component typically predicts START_X.
    :param model_var_type: A ModelVarType (enum) determining how variance is
                        output for the Gaussian component.
    :param loss_type: A LossType (enum) determining the loss function structure
                    (primarily for consistency, specific terms are handled
                    in training_losses_segmentation).
    :param dpm_solver: Boolean, indicates if DPM-Solver is to be used (currently
                    raises NotImplementedError if True).
    :param rescale_timesteps: If True, scale timesteps passed to the model to
                            a [0, 1000] range.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        dpm_solver,
        rescale_timesteps=False,
    ):
        self.model_mean_type = ModelMeanType.EPSILON
        self.bernoulli_model_mean_type = ModelMeanType.START_X
        self.model_var_type = ModelVarType.LEARNED_RANGE
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.dpm_solver = dpm_solver

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:

            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def _predict_xstart_from_eps_gaussian(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _predict_xstart_from_eps_bernoulli(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            th.abs(x_t - eps).to(device=t.device).float()
        )

    def _predict_xstart_from_xprev_gaussian(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )
    
    def _predict_xstart_from_xprev_bernoulli(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        A = (_extract_into_tensor(self.alphas, t, x_t.shape) * (1-x_t) + (1 - _extract_into_tensor(self.alphas, t, x_t.shape)) / 2)
        B = (_extract_into_tensor(self.alphas, t, x_t.shape) * x_t + (1 - _extract_into_tensor(self.alphas, t, x_t.shape)) / 2)
        C = (1 - _extract_into_tensor(self.alphas_cumprod, t-1, x_t.shape)) / 2
        numerator = A * C * xprev + B * C * (xprev -  1) + A * xprev * _extract_into_tensor(self.alphas_cumprod, t-1, x_t.shape)
        denominator = (B  + A  * xprev - B * xprev) * _extract_into_tensor(self.alphas_cumprod, t-1, x_t.shape)
        return (numerator / denominator)

    def q_mean_gaussian(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_mean_bernoulli(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: Binomial distribution parameters, of x_start's shape.
        """
        mean = _extract_into_tensor(self.alphas_cumprod, t, x_start.shape) * x_start 
        + (1 - _extract_into_tensor(self.alphas_cumprod, t, x_start.shape)) / 2
        
        return mean

    def q_sample_gaussian(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )
    
    def q_sample_bernoulli(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A noisy version of x_start.
        """

        mean = self.q_mean_bernoulli(x_start, t)
        return Binomial(1, mean).sample()
    
    

    def q_posterior_mean_gaussian(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_posterior_mean_bernoulli(self, x_start, x_t, t):
        """
        Get the distribution q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape

        theta_1 = (_extract_into_tensor(self.alphas, t, x_start.shape) * (1-x_t) + (1 - _extract_into_tensor(self.alphas, t, x_start.shape)) / 2) * (_extract_into_tensor(self.alphas_cumprod, t-1, x_start.shape) * (1-x_start) + (1 - _extract_into_tensor(self.alphas_cumprod, t-1, x_start.shape)) / 2)
        theta_2 = (_extract_into_tensor(self.alphas, t, x_start.shape) * x_t + (1 - _extract_into_tensor(self.alphas, t, x_start.shape)) / 2) * (_extract_into_tensor(self.alphas_cumprod, t-1, x_start.shape) * x_start + (1 - _extract_into_tensor(self.alphas_cumprod, t-1, x_start.shape)) / 2)

        posterior_mean = theta_2 / (theta_1 + theta_2)

        return posterior_mean


    def p_mean(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.shape[:2]
        assert C == 3, "Expected 3 channels, got {}".format(C)
        C=1
        assert t.shape == (B,)
        model_output, model_output_bernoulli, cal = model(x, self._scale_timesteps(t), **model_kwargs)
        assert model_output.shape == (B, C * 2, *x.shape[2:]), "Model output shape: {}".format(model_output.shape)
        assert model_output_bernoulli.shape == (B, C, *x.shape[2:]), "Model output shape: {}".format(model_output_bernoulli.shape)
        assert cal.shape == (B, 1, *x.shape[2:]), "Model output shape: {}".format(cal.shape)
        
        ### Gaussian Noise Component Processing ###
        x_gaussian=x[:,1:2,...]
        if self.model_var_type == ModelVarType.LEARNED_RANGE:
            assert model_output.shape == (B, C * 2, *x_gaussian.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x_gaussian.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x_gaussian.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
            raise NotImplementedError(self.model_var_type)

        def process_xstart_gaussian(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart_gaussian(
                self._predict_xstart_from_eps_gaussian(x_t=x_gaussian, t=t, eps=model_output)
            )
            model_mean, _, _ = self.q_posterior_mean_gaussian(
                x_start=pred_xstart, x_t=x_gaussian, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_gaussian.shape
        )

        ### Bernoulli Noise Component Processing ###
        x_bernoulli=x[:,2:3,...]
        
        def process_xstart_beroulli(x):
            # Only denoise - # We'll clamp Bernoulli samples after p_mean
            if denoised_fn is not None:
                x = denoised_fn(x)
            return x
        
        if self.bernoulli_model_mean_type == ModelMeanType.START_X:
            pred_xstart_bernoulli = process_xstart_beroulli(model_output_bernoulli)
            model_mean_bernoulli = self.q_posterior_mean_bernoulli(x_start=pred_xstart_bernoulli, x_t=x_bernoulli, t=t)
            model_mean_bernoulli = th.where((t == 0)[:,None, None, None], pred_xstart_bernoulli, model_mean_bernoulli)
        else:
            raise NotImplementedError(self.bernoulli_model_mean_type)
        
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            'cal': cal,
        }, {
            "mean": model_mean_bernoulli,
            "pred_xstart": pred_xstart_bernoulli,
        }
        
    def p_mean_gaussian_for_vb(
        self, model, x_gaussian, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Simpler p_mean for calculating the variational bound for (only) Gaussian part.
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x_gaussian.shape[:2]
        assert C == 1, "Expected 1 channel, got {}".format(C)
        C=1
        assert t.shape == (B,)
        model_output = model(x_gaussian, self._scale_timesteps(t), **model_kwargs)
        assert model_output.shape == (B, C * 2, *x_gaussian.shape[2:]), "Model output shape: {}".format(model_output.shape)
        
        if self.model_var_type == ModelVarType.LEARNED_RANGE:
            assert model_output.shape == (B, C * 2, *x_gaussian.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x_gaussian.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x_gaussian.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
            raise NotImplementedError(self.model_var_type)

        def process_xstart_gaussian(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart_gaussian(
                self._predict_xstart_from_eps_gaussian(x_t=x_gaussian, t=t, eps=model_output)
            )
            model_mean, _, _ = self.q_posterior_mean_gaussian(
                x_start=pred_xstart, x_t=x_gaussian, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_gaussian.shape
        )
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def p_mean_bernoulli_for_vb(
        self, model, x_bernoulli, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Simpler p_mean for calculating the variational bound for (only) Bernoulli part.
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x_bernoulli.shape[:2]
        assert C == 1, "Expected 1 channel, got {}".format(C)
        C=1
        assert t.shape == (B,)
        model_output_bernoulli = model(x_bernoulli, self._scale_timesteps(t), **model_kwargs)
        assert model_output_bernoulli.shape == (B, C, *x_bernoulli.shape[2:]), "Model output shape: {}".format(model_output_bernoulli.shape)
        
        def process_xstart_beroulli(x):
            # Only denoise - # We'll clamp Bernoulli samples after p_mean
            if denoised_fn is not None:
                x = denoised_fn(x)
            return x
        
        if self.bernoulli_model_mean_type == ModelMeanType.START_X:
            pred_xstart_bernoulli = process_xstart_beroulli(model_output_bernoulli)
            model_mean_bernoulli = self.q_posterior_mean_bernoulli(x_start=pred_xstart_bernoulli, x_t=x_bernoulli, t=t)
            model_mean_bernoulli = th.where((t == 0)[:,None, None, None], pred_xstart_bernoulli, model_mean_bernoulli)
        else:
            raise NotImplementedError(self.bernoulli_model_mean_type)
        
        return {
            "mean": model_mean_bernoulli,
            "pred_xstart": pred_xstart_bernoulli,
        }
    
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out, out_bernoulli = self.p_mean(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        
        ### Gaussian Noise Component Processing ###
        noise = th.randn_like(x[:, -1:,...])
        # Create a mask for the Gaussian noise when t!=0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))
        # When not at t=0, we add Gaussian noise to the output
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        ### Bernoulli Noise Component Processing ###
        out_bernoulli["mean"] = th.clamp(out_bernoulli["mean"], 0, 1)
        noise = Binomial(1, out_bernoulli["mean"]).sample()
        sample_bernoulli = noise if t[0] != 0 else sample

        return {"sample": sample, "pred_xstart": out["pred_xstart"], "cal": out["cal"], "sample_bernoulli": sample_bernoulli}

    def p_sample_loop_known(
        self,
        model,
        img,
        step = 1000,
        org=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        conditioner = None,
        classifier=None
    ):
        if device is None:
            device = next(model.parameters()).device
        
        img = img.to(device)
        assert img.shape[1] == 1, "Input image should have 1 channel. Current img's shape: {}".format(img.shape)
        noise = th.randn_like(img).to(device)
        bernoulli_noise = th.bernoulli(th.ones_like(img)* 0.5).to(device)
        
        x_noisy = torch.cat((img, noise, bernoulli_noise), dim=1)  #add noise as the last channel
        x_noisy = x_noisy.to(device)

        if self.dpm_solver:
            raise NotImplementedError("DPM Solver")
        else:
            print('no dpm-solver')
            # i = 0
            # letters = string.ascii_lowercase
            # name = ''.join(random.choice(letters) for i in range(10)) 
            for sample in self.p_sample_loop_progressive(
                model,
                time = step,
                noise=x_noisy,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                org=org,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
            ):
                final = sample
                # i += 1
                # '''vis each step sample'''
                # if i % 5 == 0:

                #     o1 = th.tensor(img)[:,0,:,:].unsqueeze(1)
                #     o2 = th.tensor(img)[:,1,:,:].unsqueeze(1)
                #     o3 = th.tensor(img)[:,2,:,:].unsqueeze(1)
                #     o4 = th.tensor(img)[:,3,:,:].unsqueeze(1)
                #     s = th.tensor(final["sample"])[:,-1,:,:].unsqueeze(1)
                #     tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),s)
                #     compose = th.cat(tup,0)
                #     vutils.save_image(s, fp = os.path.join('../res_temp_norm_6000_100', name+str(i)+".jpg"), nrow = 1, padding = 10)

            # if dice_score(final["sample"][:,-1,:,:].unsqueeze(1), final["cal"]) < 0.65:
            #     cal_out = torch.clamp(final["cal"] + 0.25 * final["sample"][:,-1,:,:].unsqueeze(1), 0, 1)
            # else:
            #     cal_out = torch.clamp(final["cal"] * 0.5 + 0.5 * final["sample"][:,-1,:,:].unsqueeze(1), 0, 1)
            

        return final["sample"], final["sample_bernoulli"], x_noisy, img, final["cal"]#, cal_out
    
    
    def p_sample_loop_progressive(
        self,
        model,
        time=1000,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        org=None,
        model_kwargs=None,
        device=None,
        progress=False,
        ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """

        if device is None:
            device = next(model.parameters()).device
        
        if noise is not None:
            img = noise
        else:
            raise ValueError("Noise should be provided")
        indices = list(range(time))[::-1]
        B, org_c = img.size(0), img.size(1)
        assert org_c == 3, "Input image in p_sample_loop_progressive should have 3 channels. Current img's shape: {}".format(img.shape)
        org_MRI = img[:, :-2, ...]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        # Loops over all timesteps in reverse order
        for i in indices:
            t = th.tensor([i] * B, device=device)
            # if i%100==0:
                # print('sampling step', i)
                # viz.image(visualize(img.cpu()[0, -1,...]), opts=dict(caption="sample"+ str(i) ))

            with th.no_grad():
                # print('img bef size',img.size())
                if img.size(1) != org_c:
                    img = torch.cat((org_MRI,img,img_bernoulli), dim=1)       #in every step, make sure to concatenate the original image to the sampled segmentation mask
                # image -> torch.Size([1, 4, 256, 256])
                # In each timestep, the p_sample function generates a sample from the model at that timestep
                out = self.p_sample(
                    model,
                    img.float(), # 
                    t, # A 1-D array of size batch-size (Timestep for each sample)
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                # out["sample"] -> torch.Size([1, 4, 256, 256])
                img = out["sample"]
                img_bernoulli = out["sample_bernoulli"]
        
    def ddim_sample(
        self,**kwargs):
        raise NotImplementedError("DDIM Not implemented yet!")


    def ddim_reverse_sample(
        self,**kwargs):
        raise NotImplementedError("DDIM Not implemented yet!")
        
    def ddim_sample_loop_interpolation(
        self,**kwargs):
        raise NotImplementedError("DDIM Not implemented yet!")
        
    def ddim_sample_loop(
        self,**kwargs):
        raise NotImplementedError("DDIM Not implemented yet!")

    def ddim_sample_loop_known(
            self,**kwargs):
        raise NotImplementedError("DDIM Not implemented yet!")

    def ddim_sample_loop_progressive(
        self,**kwargs):
        raise NotImplementedError("DDIM Not implemented yet!")
        
    def _vb_terms_bpd_gaussian(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_gaussian(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_gaussian_for_vb(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}
    
    def _vb_terms_bpd_bernoulli(
        self, model, x_start, x_t, t, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean = self.q_posterior_mean_bernoulli(x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_bernoulli_for_vb(
            model, x_t, t, model_kwargs=model_kwargs
        )
        kl = binomial_kl(true_mean, out["mean"])

        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -binomial_log_likelihood(x_start, means=out["mean"])
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}
    
    def training_losses_segmentation(self, model, classifier, x_start, t, model_kwargs=None):
        """
        Compute the training losses for the segmentation task.
        This function computes the losses for both Gaussian and Bernoulli noise components,
        as well as a spatial coherence calibration loss.
        :param model: the model to compute the losses for.
        :param classifier: the classifier to compute the losses for -> None here, as we don't use it in this function.
        :param x_start: the [N x C x ...] tensor of inputs, where C is the number of channels (including the segmentation channel).
        :param t: a 1-D tensor of timesteps, where each element corresponds to the timestep for each sample in the batch.
        :param model_kwargs: additional keyword arguments to pass to the model. This can be used for conditioning.
        :return: a tuple containing the loss terms and the model outputs for Gaussian and Bernoulli noise components.
        
        """
        
        mask = x_start[:, -1:, ...]
        res = torch.where(mask > 0, 1, 0)
        x_t = x_start.float()
        x_t_image_original = x_t[:, :-1, ...]
        if model_kwargs is None:
            model_kwargs = {}
            
        # Gaussian Noise
        gaussian_noise = th.randn_like(x_start[:, -1:, ...])
        res_t_gaussian = self.q_sample_gaussian(res, t, noise=gaussian_noise) # add noise to the segmentation channel
        res_t_gaussian=res_t_gaussian.float()
        
        # Bernoulli Noise
        res_t_bernoulli = self.q_sample_bernoulli(res, t) # add noise to the segmentation channel
        res_t_bernoulli = res_t_bernoulli.float()
        
        terms = {}
        
        x_input = torch.cat((x_t_image_original, res_t_gaussian, res_t_bernoulli), dim=1)
        model_out_gaussian, model_out_bernoulli, cal = model(x_input, self._scale_timesteps(t), **model_kwargs)
        B = x_t.shape[0]
        C = 1
        
        # Spatial Coherence Calibration
        terms["loss_cal"] = mean_flat((res - cal) ** 2)
        
        model_out_gaussian, model_var_gaussian = th.split(model_out_gaussian, C, dim=1)
        frozen_out_gaussian = th.cat([model_out_gaussian.detach(), model_var_gaussian], dim=1)
        terms["vb_gaussian"] = self._vb_terms_bpd_gaussian(
            model=lambda *args, r=frozen_out_gaussian: r,
            x_start=res,
            x_t=res_t_gaussian,
            t=t,
            clip_denoised=False,
        )["output"]
        target_gaussian = gaussian_noise
        terms["loss_gaussian_diff"] = mean_flat((target_gaussian - model_out_gaussian) ** 2 )
        
        terms["vb_bernoulli"] = self._vb_terms_bpd_bernoulli(
            model=lambda *args, r=model_out_bernoulli: r,
                x_start=res,
                x_t=res_t_bernoulli,
                t=t,
                model_kwargs=model_kwargs,
            )["output"]
        target_bernoulli = res
        terms["loss_bernoulli_diff"] = th.nn.functional.binary_cross_entropy_with_logits(model_out_bernoulli.float(), target_bernoulli.float(), reduction='mean') / np.log(2.0)
        
        # L_total = λ1*L_Gaussian + λ2*L_Bernoulli + λ3*L_KL,Gaussian + λ4*L_KL,Bernoulli + λ5*L_SCC
        terms["loss"] = 1*terms["loss_gaussian_diff"] + 1*terms["loss_bernoulli_diff"] + 0.01*terms["vb_gaussian"] + 0.01*terms["vb_bernoulli"] + 0.1* terms["loss_cal"]

        return terms, (model_out_gaussian, model_out_bernoulli)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

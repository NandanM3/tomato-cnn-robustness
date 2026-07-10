DEGRADATION_CONFIG = {
 
    "brightness_contrast": {
        # alpha = contrast multiplier (1.0 = unchanged, <1.0 = lower contrast)
        # beta  = brightness offset in pixel units on 0-255 scale
        # Negative beta simulates shade/overcast - positive simulates overexposure
        "mild":     {"alpha": 0.9,  "beta": -10},
        "moderate": {"alpha": 0.75, "beta": -30},
        "severe":   {"alpha": 0.6,  "beta": -50},
    },
 
    "blur_gaussian": {
        # ksize = kernel size (must be odd) - controls spatial spread of blur
        # sigma = standard deviation - controls blur intensity within that kernel
        # Higher values simulate camera shake or out of focus capture
        "mild":     {"ksize": 3,  "sigma": 1},
        "moderate": {"ksize": 7,  "sigma": 3},
        "severe":   {"ksize": 15, "sigma": 6},
    },
 
    "noise_gaussian": {
        # sigma = noise standard deviation in pixel units (0-255 scale)
        # Simulates sensor noise in low-light or low-quality camera conditions
        # sigma=5 is barely perceptible; sigma=30 is visibly grainy
        "mild":     {"sigma": 5},
        "moderate": {"sigma": 15},
        "severe":   {"sigma": 30},
    },
 
    "resolution_reduction": {
        # scale = downscale factor before upscaling back to original dimensions
        # Simulates low-resolution smartphone cameras or compressed image transfer
        # scale=0.5 = half resolution; scale=0.1 = one-tenth resolution
        "mild":     {"scale": 0.5},
        "moderate": {"scale": 0.25},
        "severe":   {"scale": 0.1},
    },
}
 



#Convenience list for the generation loop that preserves insertio order
CONDITIONS = list(DEGRADATION_CONFIG.keys())
SEVERITY_LEVELS = ["mild", "moderate", "severe"]



#--- Functions --- # 

import cv2
import numpy as np
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def apply_brightness_contrast(img, alpha=1.0, beta=0):
    out = img.astype(np.float32) * float(alpha) + float(beta)
    return np.clip(out, 0, 255).astype(np.uint8)
    
def apply_blur_gaussian(img, ksize, sigma):
    if ksize % 2 == 0:
        raise ValueError(f"ksize must be odd, got {ksize}")
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def apply_noise_gaussian(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def reduce_resolution(img,scale):
    width, height = img.shape[2:]
    small = cv2.resize(
        img,
        (int(width * scale), int(height * scale))
    )

    restored = cv2.resize(
        small,
        (width, height)
    )
    return restored

def apply_resolution_reduction(img, scale):
    if not (0 < scale < 1):
        raise ValueError(f"scale must be between 0 and 1 exclusive, got {scale}")
 
    height, width = img.shape[:2]           
    small = cv2.resize(
        img,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_LINEAR
    )
    restored = cv2.resize(
        small,
        (width, height),                    # back to original dimensions
        interpolation=cv2.INTER_NEAREST
    )
    return restored



CONDITION_FUNCTIONS = {
    "brightness_contrast":  apply_brightness_contrast,  
    "blur_gaussian":        apply_blur_gaussian,
    "noise_gaussian":       apply_noise_gaussian,
    "resolution_reduction": apply_resolution_reduction,
}

#This stores the function objects 
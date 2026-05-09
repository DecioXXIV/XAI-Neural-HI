DATASETS = ["Vat.lat.653", "Vat.lat.5951", "Vat.lat.4221", "Chelles"]
SCRIBES_TO_DATASET = {
    "Vat.lat.653": ["1", "2", "3", "4"],
    "Vat.lat.5951": ["1", "2", "3"],
    "Vat.lat.4221": ["1", "4", "6", "8"],
    "Chelles": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
}

LEGACY_TRANSFORMS = {
    "cjitter": {'brightness': (0.4, 1.3), 'contrast': 0.6, 'saturation': 0.6, 'hue': (-0.4, 0.4)},
    "cjitter_p": 1.0,
    "randaffine": {'degrees': (-10, 10), 'translate': (0.2, 0.2), 'scale': (1.3, 1.4), 'shear': 1},
    "randaffine_p": 1.0,
    "randpersp": {'distortion_scale': 0.1, 'p': 0.2},
    "gray_p": 0.2,
    "gaussian_blur": {'kernel_size': 3, 'sigma': [0.1, 0.5]},
    "blur_p": 1.0,
    "rand_eras": {'p': 0.5, 'scale': [0.02, 0.33], 'ratio': [0.3, 3.3]},
    "invert_p": 0.05,
    "gaussian_noise": {'mean': 0.0, 'std': 0.004},
    "gn_p": 0.05
}

BACKGROUND_ROBUST_TRANSFORMS = {
    "cjitter": {'brightness': (0.85, 1.15), 'contrast': (0.85, 1.18), 'saturation': (0.90, 1.10), 'hue': (-0.015, 0.015)},
    "cjitter_p": 1.0,
    "randaffine": {'degrees': [-2, 2], 'translate': [0.02, 0.02], 'scale': [0.98, 1.03], 'shear': [-1, 1]},
    "randaffine_p": 1.0,
    "randpersp": {'distortion_scale': 0.02, 'p': 0.05},
    "gray_p": 0.15,
    "gaussian_blur": {'kernel_size': 3, 'sigma': [0.1, 0.30]},
    "blur_p": 1.0,
    "rand_eras": {'p': 0.03, 'scale': [0.005, 0.02], 'ratio': [0.3, 3.3]},
    "invert_p": 0.0,
    "gaussian_noise": {'mean': 0., 'std': 0.004},
    "gn_p": 0.15
}

MODERATE_TRANSFORMS = {
    "cjitter": {'brightness': [0.85, 1.18], 'contrast': [0.85, 1.20], 'saturation': [0.90, 1.10], 'hue': (-0.02, 0.02)},
    "cjitter_p": 1.0,
    "randaffine": {'degrees': [-5, 5], 'translate': [0.05, 0.05], 'scale': [0.95, 1.08], 'shear': [-2, 2]},
    "randaffine_p": 1.0,
    "randpersp": {'distortion_scale': 0.04, 'p': 0.10},
    "gray_p": 0.10,
    "gaussian_blur": {'kernel_size': 3, 'sigma': [0.1, 0.35]},
    "blur_p": 1.0,
    "rand_eras": {'p': 0.10, 'scale': [0.005, 0.05],'ratio': [0.3, 3.3]},
    "invert_p": 0.0,
    "gaussian_noise": {'mean': 0., 'std': 0.004},
    "gn_p": 0.15
}

AGGRESSIVE_TRANSFORMS = {
    "cjitter": {'brightness': [0.85, 1.18], 'contrast': [0.85, 1.20], 'saturation': [0.90, 1.10], 'hue': (-0.02, 0.02)},
    "cjitter_p": 1.0,
    "randaffine": {'degrees': [-5, 5], 'translate': [0.05, 0.05], 'scale': [0.95, 1.08], 'shear': [-2, 2]},
    "randaffine_p": 1.0,
    "randpersp": {'distortion_scale': 0.04, 'p': 0.10},
    "gray_p": 0.10,
    "gaussian_blur": {'kernel_size': 3, 'sigma': [0.1, 0.35]},
    "blur_p": 1.0,
    "rand_eras": {'p': 0.10, 'scale': [0.005, 0.05], 'ratio': [0.3, 3.3]},
    "invert_p": 0.05,
    "gaussian_noise": {'mean': 0., 'std': 0.004},
    "gn_p": 0.15
}

XAGGRESSIVE_TRANSFORMS = {
    "cjitter": {'brightness': [0.55, 1.45], 'contrast': [0.55, 1.60], 'saturation': [0.35, 1.30],'hue': (-0.04, 0.04)},
    "cjitter_p": 0.85,
    "randaffine": {'degrees': [-8, 8], 'translate': [0.12, 0.12], 'scale': [0.82, 1.22], 'shear': [-3, 3]},
    "randaffine_p": 0.85,
    "randpersp": {'distortion_scale': 0.1, 'p': 0.25},
    "gray_p": 0.35,
    "gaussian_blur": {'kernel_size': 3, 'sigma': [0.1, 0.5]},
    "blur_p": 0.30,
    "rand_eras": {'p': 0.35, 'scale': [0.01, 0.12], 'ratio': [0.3, 3.3]},
    "invert_p": 0.05,
    "gaussian_noise": {'mean': 0., 'std': 0.004},
    "gn_p": 0.25}

__all__ = ["DATASETS", "SCRIBES_TO_DATASET", "LEGACY_TRANSFORMS", "MODERATE_TRANSFORMS", "BACKGROUND_ROBUST_TRANSFORMS", "AGGRESSIVE_TRANSFORMS", "XAGGRESSIVE_TRANSFORMS"]
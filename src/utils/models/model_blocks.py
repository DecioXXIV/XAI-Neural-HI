import torch.nn as nn
from typing import List

from src.utils.logger import Logger

logger = Logger()

_ACTIVATIONS = {
    "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU,
    "celu": nn.CELU, "silu": nn.SiLU, "swish": nn.SiLU, "mish": nn.Mish
}

def build_classification_head(input_dim: int, n_classes: int, layers: List[str]) -> nn.Sequential:
    modules, current_dim = [], input_dim
    
    for token in layers:
        layer, id, param = None, None, None
        if ':' in token: id, param = token.split(':')
        else: id = token
        
        if id == "fc":
            layer = nn.Linear(current_dim, int(param))
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            current_dim = int(param)
        
        elif id == "act": layer = _ACTIVATIONS[param]()
        elif id == "dropout": layer = nn.Dropout(float(param))
        elif id == "bn": layer = nn.BatchNorm1d(current_dim)
        elif id == "ln": layer = nn.LayerNorm(current_dim)
        
        else:
            logger.warning(f"Unrecognized layer token '{token}' in classification head configuration. Skipping this token.")
        
        if layer is not None: modules.append(layer)
    
    # Add final classification layer
    final_layer = nn.Linear(current_dim, n_classes)
    nn.init.xavier_normal_(final_layer.weight)
    nn.init.zeros_(final_layer.bias)
    modules.append(final_layer)
    
    return nn.Sequential(*modules)
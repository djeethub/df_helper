import torch
from safetensors.torch import load_file
from collections import defaultdict

LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"

current_pipeline = None
original_weights = None

def clear_lora(pipe):
    global current_pipeline, original_weights

    if current_pipeline is not None and pipe == current_pipeline:
        for layer, data in original_weights.items():
            curr_layer = find_layer(pipe, layer)
            curr_layer.weight.data = data.clone().detach()
    current_pipeline = None
    original_weights = None

def find_layer(pipe, layer):
    if "text" in layer:
        layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
        curr_layer = pipe.text_encoder
    else:
        layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
        curr_layer = pipe.unet

    # find the target layer
    temp_name = layer_infos.pop(0)
    while len(layer_infos) > -1:
        try:
            curr_layer = curr_layer.__getattr__(temp_name)
            if len(layer_infos) > 0:
                temp_name = layer_infos.pop(0)
            elif len(layer_infos) == 0:
                break
        except Exception:
            if len(temp_name) > 0:
                temp_name += "_" + layer_infos.pop(0)
            else:
                temp_name = layer_infos.pop(0)

    return curr_layer

def load_lora(pipe, path, alpha):
    global current_pipeline, original_weights
    
    if current_pipeline != pipe:
        current_pipeline = pipe
        original_weights = {}
    
    # load LoRA weight from .safetensors
    state_dict = load_file(path, pipe.device.type)
    updates = defaultdict(dict)
    dtype = pipe.dtype

    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():
        curr_layer = find_layer(layer)

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        item_alpha = elems['alpha']
        if item_alpha:
            item_alpha = item_alpha.item() / weight_up.shape[1]
        else:
            item_alpha = 1.0

        if layer not in original_weights:
            original_weights[layer] = curr_layer.weight.data.clone().detach()

        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += alpha * item_alpha * torch.mm(
                weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)
            ).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += alpha * item_alpha * torch.mm(weight_up, weight_down)
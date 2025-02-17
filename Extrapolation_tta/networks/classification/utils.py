
import torch


from .vit import AdaptSupCEVit
from .efficientnet import AdaptSupCEEfficientnet
from .resnet_simclr import AdaptSupCESimCLRResNet
from .resnet import AdaptSupCEResNet
from .shot_resnet import AdaptSupCEResNet2
from .resnet_adacontrast import AdaptSupCEResNet3
from .mobilenet import AdaptSupCEMobileNet, load_network_mobilenet


def load_network(net, path):

    if path is not None:
        ckpt = torch.load(path, map_location="cpu")
        if "model" in ckpt.keys():
            state_dict = ckpt['model']
        else:
            state_dict = ckpt

        net_dict = {}

        for k, v in state_dict.items():

            if "head" not in k:
                k = k.replace("module.", "")
                net_dict[k] = v
            else:
                print(f"Removing {k}...")

        model_keys = set(net.state_dict().keys())
        loaded_keys = set(net_dict.keys())
        missing_in_loaded = model_keys - loaded_keys
        extra_in_loaded = loaded_keys - model_keys
        print("Missing in loaded state dict:", missing_in_loaded)
        print("Extra in loaded state dict:", extra_in_loaded)

        assert all(("head" in key or "lora" in key or "soft_mask" in key) for key in missing_in_loaded)



        net.load_state_dict(net_dict, strict=False)


def build_network(
    network_name, 
    n_classes, 
    ckpt_path=None,
):

    if "shot" in network_name:
        net = AdaptSupCEResNet2(ckpt_path, network_name, n_classes)
    elif "adacontrast" in network_name:
        net = AdaptSupCEResNet3(ckpt_path, network_name, n_classes)
    else:
        if "resnet" in network_name:
            if "simclr" in network_name:
                net = AdaptSupCESimCLRResNet(network_name, n_classes)
            else:
                net = AdaptSupCEResNet(
                    network_name, 
                    n_classes,
                )
            load_network(net, ckpt_path)
        elif "vit" in network_name:
            net = AdaptSupCEVit(num_classes=n_classes)
            load_network(net, ckpt_path)
        elif "efficientnet" in network_name:
            net = AdaptSupCEEfficientnet(network_name, n_classes)
            load_network(net, ckpt_path)
        elif "mobilenet" in network_name:
            net = AdaptSupCEMobileNet(network_name, n_classes)
            load_network_mobilenet(net, ckpt_path)
        else:
            raise NotImplementedError("The requested network architecture is not implemented !")

    return net
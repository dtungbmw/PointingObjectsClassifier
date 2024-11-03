from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from model import build_pointing_network
import torch


class GesturePointingDirectionEstimator:
    
    def __init__(self):
        pass
    
    def predict(self, video):
        res_direction = None
        res_p = 0
        
        return res_direction, res_p


class DeepointointingEstimator(GesturePointingDirectionEstimator):

    def build_network(self, cfg, DEVICE):
        network = build_pointing_network(cfg, DEVICE)

        # Since the model trained using pytorch lightning contains `model.` as an prefix to the keys of state_dict, we should remove them before loading
        model_dict = torch.load(cfg.ckpt, map_location=torch.device('cpu'))["state_dict"]
        new_model_dict = dict()
        for k, v in model_dict.items():
            new_model_dict[k[6:]] = model_dict[k]
        model_dict = new_model_dict
        network.load_state_dict(model_dict)
        network.to(DEVICE)

        #Path("demo").mkdir(exist_ok=True)
        #fps = 15
        return network

    def predict(self, video):
        pass
#! C:\Users\alanc\OneDrive\Documents\Python Projects\DogWaterLevelDetector\ModelCreation\venv\Scripts\python.exe
import hydra
from hydra.utils import instantiate
import tensorflow as tf
#from tensorflow.keras import layers
#from tensorflow.keras.models import Sequential
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(version_base = None, config_path="./config", config_name="config")
def config(cfg):
    a = instantiate(cfg.modelconf.model)
    print(cfg.modelconf.layersconf.Conv2D_1.filters)
    print(cfg.modelconf.layersconf.Conv2D_2.filters)

if __name__ == "__main__":
    config()
    
#import torch
#from hydra import initialize, compose
#import hydra
#from hydra.utils import instantiate
#initialize("./config") # Assume the configuration file is in the current folder
#cfg = compose(config_name='config')
#net = instantiate(cfg.feature_extractor)

#@hydra.main(version_base = None, config_path="./config", config_name="config")
#def train(cfg):
#    net = instantiate(cfg.feature_extractor)
#    print(net)

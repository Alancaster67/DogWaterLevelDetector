#! C:\Users\alanc\OneDrive\Documents\Python Projects\DogWaterLevelDetector\ModelCreation\venv\Scripts\python.exe
import hydra
from hydra.utils import instantiate
import tensorflow as tf
from omegaconf import OmegaConf
import pathlib

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(version_base = None, config_path="./config", config_name="config")
def train_model(cfg):
    model = instantiate(cfg.modelconf.model)
    performance_metrics = instantiate(cfg.metrics_list)
    
    train_dir = pathlib.Path.cwd().parent / cfg.dataconf.train_dataset
    val_dir = pathlib.Path.cwd().parent / cfg.dataconf.validation_dataset
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode = 'binary',
        image_size=(cfg.dataconf.input_shape[0], cfg.dataconf.input_shape[1]),
        batch_size=cfg.batch_size)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode = 'binary',
        image_size=(cfg.dataconf.input_shape[0], cfg.dataconf.input_shape[1]),
        batch_size=cfg.batch_size)
    
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model.compile(optimizer= cfg.optimizer,
              loss=instantiate(cfg.loss),
              metrics=instantiate(cfg.metrics_list))
    
    
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=cfg.epochs
    )
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
if __name__ == "__main__":
    train_model()
    

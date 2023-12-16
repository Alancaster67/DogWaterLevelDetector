import hydra
from hydra.utils import instantiate
import tensorflow as tf
from omegaconf import OmegaConf
import pathlib
import logging

OmegaConf.register_new_resolver("eval", eval)
log = logging.getLogger(__name__)

@hydra.main(version_base = None, config_path="./config", config_name="config")
def train_model(cfg):
    
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
    
    
    model = instantiate(cfg.modelconf.model)
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(train_ds.cardinality()).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model.compile(optimizer= instantiate(cfg.optimizer),
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
    log.info(f"Train Accuracy: {acc}")
    log.info(f"Train Loss: {loss}")
    log.info(f"Validation Accuracy: {val_acc}")
    log.info(f"Validation Loss: {val_loss}")
    
    # Convert the model.
    #converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #tflite_model = converter.convert()

    # Save the model.
    #with open('model_sigmoid.tflite', 'wb') as f:
    #f.write(tflite_model)
    
if __name__ == "__main__":
    train_model()
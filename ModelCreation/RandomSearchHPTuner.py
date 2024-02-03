import hydra
from hydra.utils import instantiate
import tensorflow as tf
from omegaconf import OmegaConf
import pathlib
import logging
import multiprocessing
import datetime
from flatten_dict import flatten 
from flatten_dict import unflatten
from tensorboard.plugins.hparams import api as hp
from datetime import datetime
import copy

OmegaConf.register_new_resolver("eval", eval)
log = logging.getLogger(__name__)

@hydra.main(version_base = None, config_path="./RandomSearchConfig", config_name="config")
def main(cfg):
    
    def build_model(cfg):
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential()
        
        model_conf = cfg.modelconf.layersconf
        
        for layer in cfg.modelconf.layersconf.keys():
            l = instantiate(model_conf[layer])
            model.add(l)
        
        return model
        
    def configure_exp(cfg_input):
        cfg = copy.deepcopy(cfg_input)
        params = flatten(cfg.experiment.params)
        hparam_dict = {}
      
        for param_key in list(params.keys()):
            temp_dict = cfg.experiment.hparams
            for k in param_key:
                temp_dict = temp_dict[k]
            
            dom = instantiate(temp_dict.domain)
            dom = dom(**OmegaConf.to_container(temp_dict.domain_args))
            sampled_value = dom.sample_uniform()
            params[param_key] = sampled_value
            
            temp_dict = OmegaConf.to_container(temp_dict)
            temp_dict['domain'] = dom
            del temp_dict['domain_args']
            
            hparam_dict.update({hp.HParam(**temp_dict): sampled_value})
            
            cfg.experiment.params = unflatten(params)
            
        #if 'runid' in cfg.experiment.keys():
        #    cfg.experiment.runid = +1
        #else:
        #    OmegaConf.update(cfg, 'experiment', {'runid' : 0}, force_add=True)
        #    with tf.summary.create_file_writer(f'logs/{cfg.experiment.name}').as_default():
        #        hp.hparams_config(
        #            hparams=list(hparam_dict.keys()),
        #            metrics=[hp.Metric(metric) for metric in cfg.experiment.metrics],
        #           )
        del params
        return cfg, hparam_dict

    def override_cfg(new_cfg):
        params = flatten(new_cfg.experiment.params)
        new_cfg = flatten(new_cfg)
        new_cfg.update(params)
        new_cfg = unflatten(new_cfg)
        del new_cfg['experiment']
        return OmegaConf.create(new_cfg)
    
    def load_dataset(cfg):
        train_dir = pathlib.Path.cwd().parent / cfg.dataconf.train_dataset
        val_dir = pathlib.Path.cwd().parent / cfg.dataconf.validation_dataset
        
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            label_mode = 'binary',
            image_size=(cfg.dataconf.input_shape[0], cfg.dataconf.input_shape[1]),
            batch_size= cfg.dataconf.batch_size
            )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            label_mode = 'binary',
            image_size=(cfg.dataconf.input_shape[0], cfg.dataconf.input_shape[1]),
            batch_size= cfg.dataconf.batch_size
            )
        
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(train_ds.cardinality()).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        return train_ds, val_ds
    
    def train_test_model(cfg, hparams, train_ds, val_ds):
        
        train_ds = train_ds.rebatch(cfg.dataconf.batch_size)
        val_ds = val_ds.rebatch(cfg.dataconf.batch_size)
        
        cli_in = OmegaConf.from_cli()
        model_name = cli_in.modelconf
        if 'experiment' in cli_in.keys():
            experiment_name = 'experiments/' + cli_in.experiment
            
        else:
            experiment_name = 'default'       
        
        logdir = f"logs/{model_name}/{experiment_name}/"+ datetime.now().strftime("%Y%m%d-%H%M%S")

        model = build_model(cfg)
        model.compile(optimizer= instantiate(cfg.modelconf.optimizer),
                loss=instantiate(cfg.modelconf.loss),
                metrics= list(cfg.modelconf.metrics))
        
        history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.modelconf.epochs,
        callbacks = [
            tf.keras.callbacks.TensorBoard(logdir),
            hp.KerasCallback(logdir, hparams)
            ],
        )
        
        #acc = history.history['accuracy']
        #val_acc = history.history['val_accuracy']
        #loss = history.history['loss']
        #val_loss = history.history['val_loss']
        #log.info(f"Train Accuracy: {acc}")
        #log.info(f"Train Loss: {loss}")
        #log.info(f"Validation Accuracy: {val_acc}")
        #log.info(f"Validation Loss: {val_loss}")
        
        # Convert the model.
        #converter = tf.lite.TFLiteConverter.from_keras_model(model)
        #tflite_model = converter.convert()

        # Save the model.
        #with open('model_sigmoid.tflite', 'wb') as f:
        #f.write(tflite_model)

    def run_exp(cfg):
        cli_in = OmegaConf.from_cli()
        exp_cfg, hparams = configure_exp(cfg_input = cfg)
        exp_cfg = override_cfg(new_cfg = exp_cfg)
        train_ds, val_ds = load_dataset(cfg = exp_cfg)
        
        for __ in range(cfg.modelconf.replicates):
                try:
                    #p = multiprocessing.Process(target=train_test_model, args=[exp_cfg, hparams, train_ds, val_ds])
                    #p.start()
                    #p.join()
                    train_test_model(cfg=exp_cfg, hparams=hparams, train_ds=train_ds, val_ds=val_ds)
                except:
                    pass 
    
    
    
    cli_in = OmegaConf.from_cli()
       
    if 'experiment' in cli_in:
        for _ in range(cfg.experiment.samples):
            p = multiprocessing.Process(target=run_exp, args=[cfg])
            p.start()
            p.join()
    else:

        hparams = {'model': cli_in.modelconf}
        
        train_ds, val_ds = load_dataset(cfg)
        for __ in range(cfg.modelconf.replicates):
            train_test_model(cfg=cfg, hparams=hparams, train_ds=train_ds, val_ds=val_ds)
                    

if __name__ == "__main__":
    main()

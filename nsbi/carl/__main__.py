from lightning.pytorch.cli import LightningCLI
import yaml
from models.carl import CARL
from models.model_wrapper import monitored_model
from datasets.balanced import BalancedDataModule

import torch
torch.set_float32_matmul_precision('high')

class CarlCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.features", "model.n_features", compute_fn=lambda x: len(x))
        parser.link_arguments("seed_everything", "data.random_state")
        
    def before_instantiate_classes(self):
        config_str = self.parser.dump(self.config, skip_none=False)
        self._config_dict = yaml.safe_load(config_str)
        
    def after_instantiate_classes(self):
        model_kwargs = self._config_dict['fit']["model"]
        
        n_features = len(self._config_dict['fit']['data']['features'])
        features = self._config_dict['fit']['data']['features']
        
        model_kwargs['n_features'] = n_features
        model_kwargs['feature_names'] = features
        
        arg_order = ["n_features", "n_layers", "n_nodes", "learning_rate"]
        
        wrapped_model = monitored_model(CARL, arg_order=arg_order, **model_kwargs)

        self.model = wrapped_model
        


        
def main():
    logger_cfg = {
        "class_path": "lightning.pytorch.loggers.CSVLogger",
        "init_args": {"save_dir": "./"}
    }

    cli = CarlCLI(
        model_class=CARL,
        datamodule_class=BalancedDataModule,
        trainer_defaults={"logger": logger_cfg},
    )

if __name__ == "__main__":
    main()
    
    
    
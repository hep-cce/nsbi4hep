from lightning.pytorch.cli import LightningCLI
import yaml
import sys
from models.carl import CARL
from models.model_wrapper import monitored_model
from datasets.balanced import BalancedDataModule
import argparse
import torch
torch.set_float32_matmul_precision('high')

from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback


def _try_num(x: str):
    try:
        i = int(x); return i
    except ValueError:
        try: return float(x)
        except ValueError: return x  # keep as string

def _parse_dist(spec: str):
    kind, _, args = spec.partition(":")
    parts = [p.strip() for p in args.split(",") if p.strip() != ""]
    k = kind.lower()
    if k == "randint":     return tune.randint(int(parts[0]), int(parts[1]))
    if k == "qrandint":    return tune.qrandint(int(parts[0]), int(parts[1]), int(parts[2]))
    if k == "uniform":     return tune.uniform(float(parts[0]), float(parts[1]))
    if k == "loguniform":  return tune.loguniform(float(parts[0]), float(parts[1]))
    if k == "choice":      return tune.choice([_try_num(p) for p in parts])
    if k == "grid":        return tune.grid_search([_try_num(p) for p in parts])
    raise ValueError(f"Unsupported distribution spec: {spec}")

def _space_from_json(d: dict):
    out = {}
    for key, body in d.items():
        if not isinstance(body, dict) or len(body) != 1:
            raise ValueError(f"Bad spec for {key}: {body}")
        name, params = next(iter(body.items()))
        if name == "randint":     out[key] = tune.randint(*map(int, params))
        elif name == "qrandint":  out[key] = tune.qrandint(*map(int, params))
        elif name == "uniform":   out[key] = tune.uniform(*map(float, params))
        elif name == "loguniform":out[key] = tune.loguniform(*map(float, params))
        elif name == "choice":    out[key] = tune.choice([_try_num(x) for x in params])
        elif name == "grid":      out[key] = tune.grid_search([_try_num(x) for x in params])
        else:
            raise ValueError(f"Unsupported distribution: {name}")
    return out

def parse_tune_cli(argv):
    ap = argparse.ArgumentParser(add_help=False)

    # Per-trial resources
    ap.add_argument("--resources.gpu_per_trial", dest="gpu_per_trial", type=float, default=1.0)
    ap.add_argument("--resources.cpu_per_trial", dest="cpu_per_trial", type=float, default=4.0)

    # Search space
    ap.add_argument("--space.n_layers",        dest="space_n_layers",        default="randint:5,15")
    ap.add_argument("--space.n_nodes",         dest="space_n_nodes",         default="qrandint:100,1000,100")
    ap.add_argument("--space.learning_rate",   dest="space_learning_rate",   default="loguniform:1e-5,3e-2")

    # Optional full JSON override
    ap.add_argument("--space.json", dest="space_json", default=None,
                    help="JSON dict to override search space entirely")

    tune_args, remaining = ap.parse_known_args(argv)

    resources = {"gpu": tune_args.gpu_per_trial, "cpu": tune_args.cpu_per_trial}

    if tune_args.space_json:
        param_space = _space_from_json(json.loads(tune_args.space_json))
    else:
        param_space = {
            "n_layers":      _parse_dist(tune_args.space_n_layers),
            "n_nodes":       _parse_dist(tune_args.space_n_nodes),
            "learning_rate": _parse_dist(tune_args.space_learning_rate),
        }

    return resources, param_space, remaining


class CarlTuneCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.features", "model.n_features", compute_fn=lambda x: len(x))
        

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
        
        seed = self._config_dict.get("seed_everything", None)
        if isinstance(seed, int) and hasattr(self.datamodule, "random_state"):
            self.datamodule.random_state = seed


def build_cli(extra_args=None, ray_callback=None):
    logger_cfg = {
        "class_path": "lightning.pytorch.loggers.CSVLogger",
        "init_args": {"save_dir": "./"}
    }
    trainer_defaults = {"logger": logger_cfg,
                        "enable_progress_bar": False,}
    if ray_callback is not None:
        trainer_defaults["callbacks"] = [ray_callback]

    return CarlTuneCLI(
        model_class=CARL,
        datamodule_class=BalancedDataModule,
        trainer_defaults=trainer_defaults,
        args=extra_args or [],
    )


def ray_train(config, base_args):
    overrides = [
        f"--model.n_layers={config['n_layers']}",
        f"--model.n_nodes={config['n_nodes']}",
        f"--model.learning_rate={config['learning_rate']}",
    ]

    from ray.tune.integration.pytorch_lightning import TuneReportCallback
    tune_cb = TuneReportCallback(metrics={"val_loss": "val_loss",
                                         "l1_energy_ws": "l1_energy_ws"}, on="validation_end")

    #prevent ray worker flags to be passed to lightningCLI
    import sys
    _argv_backup = sys.argv
    try:
        sys.argv = [_argv_backup[0]] 
        build_cli(extra_args=['fit'] + base_args + overrides, ray_callback=tune_cb)
    finally:
        sys.argv = _argv_backup
        
def tune_main():
    
    resources, param_space, base_args = parse_tune_cli(sys.argv[1:])
    
    scheduler = ASHAScheduler(max_t=100, grace_period=5, reduction_factor=3)
    trainable = tune.with_parameters(ray_train, base_args=base_args)
    tuner = Tuner(
        tune.with_resources(trainable, resources=resources),  
        param_space=param_space,
        tune_config=TuneConfig(
            metric="val_loss",  #change to one of our custom metrics
            mode="min",
            num_samples=4,          # total trials
            scheduler=scheduler,
        ),
    )
    results = tuner.fit()

    best = results.get_best_result(metric="val_loss", mode="min")
    print("Best config:", best.config)
    print("Best val_loss:", best.metrics.get("val_loss"))


if __name__ == "__main__":
    tune_main()

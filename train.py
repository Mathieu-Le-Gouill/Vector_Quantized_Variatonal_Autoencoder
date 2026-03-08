from utils.training.trainer import Trainer
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    trainer = Trainer(config)

    trainer.train()

    if config.evaluation.active:
        trainer.evaluate()

    if config.visualization.active:
        trainer.visualize_recon()

if __name__ == "__main__":
    main()


import hydra
from omegaconf import DictConfig, OmegaConf
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils import initialize_ray
import ray
import weave
from .swe_generator import SweAgentGenerator,SWERolloutWorker
from skyrl_train.trainer import RayPPOTrainer,RayPPOAsynchTrainer
from skyrl_train.generators.base import GeneratorInterface


class MiniSWEPPOExp(BasePPOExp):
    """Customized PPO experiment for SWE-Agent."""

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initializes the generator."""
        generator = SweAgentGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=OmegaConf.create({"max_env_workers": 0}),
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=self.cfg.trainer.policy.model.path,
        )
        return generator
    def get_rollout_worker(self,cfg, tokenizer,sweagent_config):
        worker=SWERolloutWorker.remote(generator_cfg=cfg.generator,tokenizer=tokenizer,sweagent_config=sweagent_config)
        return worker
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator: GeneratorInterface,
        colocate_pg,
    ):
        """Initializes the trainer. By default, we use RayPPOTrainer.
        RayPPOAsynchTrainer overlap the environment setup and model training for better resource utilization.

        Returns:
            RayPPOTrainer: The trainer.
        """
        if cfg.trainer.asynch:
            sweworker=self.get_rollout_worker(cfg,tokenizer,generator.sweagent_config)
            return RayPPOAsynchTrainer(
                sweworker=sweworker,
                cfg=cfg,
                tracker=tracker,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                inference_engine_client=inference_engine_client,
                generator=generator,
                colocate_pg=colocate_pg,
            )
        else:

            return RayPPOTrainer(
                cfg=cfg,
                tracker=tracker,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                inference_engine_client=inference_engine_client,
                generator=generator,
                colocate_pg=colocate_pg,
            )
@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    # import hf tokenizer to debug 
  
    exp = MiniSWEPPOExp(cfg)
    
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:

    # validate the arguments
    print('validate')
    validate_cfg(cfg)
    print('init')
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()

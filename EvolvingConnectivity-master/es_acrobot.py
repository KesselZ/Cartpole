import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from functools import partial
from typing import Any, Dict, Tuple
import time

import jax
import jax.numpy as jnp
import flax
import optax
import gymnax
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb

from networks import NETWORKS
from utils.functions import rand_normal_like_tree, zeros_like_tree, finitemean, param_norm, save_obj_to_file

# Use RBG generator for less memory consumption
jax.config.update("jax_default_prng_impl", "rbg")


@flax.struct.dataclass
class ESConfig:
    # Network, optim & env class
    network_cls: Any = None
    optim_cls: Any = None
    env_cls: Any = None
    # [Hyperparameters] ES
    pop_size: int = 10240
    sigma: float = 0.3
    lr: float = 0.15
    weight_decay: float = 0.1

    # [Hyperparameters] Warmup
    warmup_steps: int = 0

    # [Hyperparameters] Eval
    eval_size: int = 128

    # [Computing] Data types
    action_dtype: Any = jnp.float32


@flax.struct.dataclass
class RunnerState:
    key: Any
    # Normalizer
    normalizer_state: Dict[str, jnp.ndarray]
    # Env reset state pool
    obs_array: Any
    state_object: Any
    # Network optimization
    params: Any
    opt_state: Any


@flax.struct.dataclass
class PopulationState:
    # Network
    network_params: Any
    network_states: Any
    # Env
    env_states: Any
    obs_array: Any
    # Fitness
    fitness_totrew: jnp.ndarray
    fitness_sum: jnp.ndarray
    fitness_n: jnp.ndarray


def _centered_rank_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Centered rank from: https://arxiv.org/pdf/1703.03864.pdf"""
    shape = x.shape
    x = x.ravel()
    x = jnp.argsort(jnp.argsort(x))
    x = x / (len(x) - 1) - .5
    return x.reshape(shape)


# Evaluate the population for a single step
def _evaluate_step(pop: PopulationState, runner: RunnerState, conf: ESConfig, env_params: Any) -> PopulationState:
    # Normalize observations
    obs_norm = (pop.obs_array - runner.normalizer_state["mean"]) / (runner.normalizer_state["var"] + 1e-8)

    # Step network
    new_network_states, act = jax.vmap(conf.network_cls.apply)(pop.network_params, pop.network_states, obs_norm)

    # Convert continuous output to discrete action (for CartPole: 0 or 1)
    act = jnp.argmax(act, axis=-1)
    act = act.astype(jnp.int32)
    # Step environment
    new_key, step_key = jax.random.split(runner.key)
    runner = runner.replace(key=new_key)
    vmapped_step = jax.vmap(conf.env_cls.step, in_axes=(0, 0, 0, None))
    step_keys = jax.random.split(step_key, conf.pop_size)
    n_obs, n_state, reward, done, info = vmapped_step(step_keys, pop.env_states, act, env_params)
    new_env_states = n_state

    # Calculate episodic rewards
    new_fitness_totrew = pop.fitness_totrew + reward
    new_fitness_sum = jnp.where(done, pop.fitness_sum + new_fitness_totrew, pop.fitness_sum)
    new_fitness_n = jnp.where(done, pop.fitness_n + 1, pop.fitness_n)
    new_fitness_totrew = jnp.where(done, 0, new_fitness_totrew)

    # Reset done environments
    def _where_done(x, y):
        done2 = done
        done2 = done2.reshape([-1] + [1] * (len(x.shape) - 1))
        return jnp.where(done2, x, y)

    new_env_states = jax.tree_map(_where_done, runner.state_object, new_env_states)

    return pop.replace(
        network_states=new_network_states,
        env_states=new_env_states,
        obs_array=n_obs,
        fitness_totrew=new_fitness_totrew,
        fitness_sum=new_fitness_sum,
        fitness_n=new_fitness_n
    )


@partial(jax.jit, static_argnums=(2, 4))
def _runner_init(key: Any, network_init_key: Any, env: Any, env_params: Any, conf: ESConfig) -> RunnerState:
    # Split keys for initializing env
    key, env_init_key = jax.random.split(key)

    # Initialize environment
    batched_keys = jax.random.split(env_init_key, conf.pop_size)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(batched_keys, env_params)

    # Initialize network params and optimizer state
    network_params = jax.jit(conf.network_cls.init, donate_argnums=(1,))(
        network_init_key,
        conf.network_cls.initial_carry(jax.random.PRNGKey(0), conf.pop_size),
        obs
    )
    optim_state = conf.optim_cls.init(network_params)

    # Runner state
    runner = RunnerState(
        key=key,
        normalizer_state={"mean": jnp.zeros((obs.shape[-1],)), "var": jnp.ones((obs.shape[-1],))},
        obs_array=obs,
        state_object=state,
        params=network_params,
        opt_state=optim_state
    )
    return runner


@partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
def _runnner_run(runner: RunnerState, conf: ESConfig, env_params: Any) -> Tuple[RunnerState, Dict]:
    metrics = {}

    # Split keys for this run
    new_key, run_key, carry_key = jax.random.split(runner.key, 3)
    runner = runner.replace(key=new_key)

    # Generate params with antithetic noise
    noise = rand_normal_like_tree(run_key, runner.params, std=conf.sigma,
                                  batch_shape=((conf.pop_size - conf.eval_size) // 2,))
    zeros = zeros_like_tree(runner.params, batch_shape=(conf.eval_size,))
    network_params = jax.tree_map(lambda x, n, z: x + jnp.concatenate([n, -n, z], axis=0), runner.params, noise, zeros)

    # Split the eval and train fitness
    def _split_fitness(x):
        return jnp.split(x, [conf.pop_size - conf.eval_size, ])

    # Initialize population
    pop = PopulationState(
        network_params=network_params,
        network_states=conf.network_cls.initial_carry(carry_key, conf.pop_size),
        env_states=runner.state_object,
        obs_array=runner.obs_array,
        fitness_totrew=jnp.zeros(conf.pop_size),
        fitness_sum=jnp.zeros(conf.pop_size),
        fitness_n=jnp.zeros(conf.pop_size, dtype=jnp.int32)
    )

    # Warmup steps
    if conf.warmup_steps > 0:
        pop, _ = jax.lax.scan(lambda p, x: (_evaluate_step(p, runner, conf, env_params), None), pop, None,
                              length=conf.warmup_steps)
        warmup_fitness, warmup_eval_fitness = _split_fitness(pop.fitness_sum / pop.fitness_n)
        metrics.update({
            "warmup_fitness": finitemean(warmup_fitness),
            "warmup_eval_fitness": finitemean(warmup_eval_fitness)
        })

        # Update normalizer
        new_mean = 0.9 * runner.normalizer_state["mean"] + 0.1 * jnp.mean(pop.obs_array, axis=0)
        new_var = 0.9 * runner.normalizer_state["var"] + 0.1 * jnp.var(pop.obs_array, axis=0)
        runner = runner.replace(normalizer_state={"mean": new_mean, "var": new_var})

        # Reset envs and clear fitness
        pop = pop.replace(
            env_states=runner.state_object,
            obs_array=runner.obs_array,
            fitness_totrew=jnp.zeros(conf.pop_size),
            fitness_sum=jnp.zeros(conf.pop_size),
            fitness_n=jnp.zeros(conf.pop_size, dtype=jnp.int32)
        )

    # Evaluate
    def _eval_stop_cond(p: PopulationState) -> jnp.ndarray:
        return ~jnp.all(p.fitness_n >= 1)

    pop = jax.lax.while_loop(_eval_stop_cond, (lambda p: _evaluate_step(p, runner, conf, env_params)), pop)

    # Update normalizer if no warmup
    if conf.warmup_steps <= 0:
        new_mean = 0.9 * runner.normalizer_state["mean"] + 0.1 * jnp.mean(pop.obs_array, axis=0)
        new_var = 0.9 * runner.normalizer_state["var"] + 0.1 * jnp.var(pop.obs_array, axis=0)
        runner = runner.replace(normalizer_state={"mean": new_mean, "var": new_var})

    # Calculate population metrics
    if hasattr(conf.network_cls, "carry_metrics"):
        metrics.update(conf.network_cls.carry_metrics(pop.network_states))

    # Calculate fitness
    fitness, eval_fitness = _split_fitness(pop.fitness_sum / pop.fitness_n)

    # Transform and split antithetic fitness --> weight
    weight = _centered_rank_transform(fitness)
    weight_pos, weight_neg = jnp.split(weight, 2, axis=-1)
    weight = weight_pos - weight_neg

    # Compute gradients
    grads = jax.tree_map(
        lambda p, op: -jnp.mean(
            weight.reshape([-1] + [1] * (p.ndim - 1)) * (p[:(conf.pop_size - conf.eval_size) // 2] - op), axis=0),
        pop.network_params, runner.params
    )

    # Gradient step
    updates, new_opt_state = conf.optim_cls.update(grads, runner.opt_state, runner.params)
    new_params = optax.apply_updates(runner.params, updates)

    runner = runner.replace(
        params=new_params,
        opt_state=new_opt_state
    )

    # Metrics
    metrics.update({
        "fitness": jnp.mean(fitness),
        "eval_fitness": jnp.mean(eval_fitness),
        "param_norm": param_norm(new_params)
    })
    return runner, metrics


def main(conf):
    conf = OmegaConf.merge({
        # Task
        "seed": 0,
        "task": "acrobot",
        "task_conf": {},
        "episode_conf": {
            "max_episode_length": 500,
            "action_repeat": 1
        },
        # Train & Checkpointing
        "total_generations": 100,
        "save_every": 50,
        # Network
        "network_type": "DenseSNN",
        "network_conf": {},
        # ES hyperparameter (see ESConfig)
        "es_conf": {}
    }, conf)
    # Naming
    conf = OmegaConf.merge({
        "project_name": f"ESDense-{conf.task}",
        "run_name": f"ES {conf.seed} {conf.network_type} {time.strftime('%H:%M %m-%d')}"
    }, conf)
    # ES Config
    es_conf = ESConfig(**conf.es_conf)

    print(OmegaConf.to_yaml(conf))
    print(es_conf)

    # Create environment
    env, env_params = gymnax.make("Acrobot-v1")

    # Create network class
    network_cls = NETWORKS[conf.network_type]
    network = network_cls(out_dims=3, **conf.network_conf)  # CartPole has 1 discrete action

    # Create optimizer
    optim = optax.adamw(learning_rate=es_conf.lr, weight_decay=es_conf.weight_decay)

    # Initialize ES config
    es_conf = es_conf.replace(
        network_cls=network,
        optim_cls=optim,
        env_cls=env
    )

    # Runner state
    key_run, key_network_init = jax.random.split(jax.random.PRNGKey(conf.seed))
    runner = _runner_init(key_run, key_network_init, env, env_params, es_conf)

    # Save model path
    conf.save_model_path = "models/{}/{}/".format(conf.project_name, conf.run_name)

    # Initialize wandb
    if "log_group" in conf:
        wandb.init(reinit=True, project=f"(G) ESDense-{conf.task}", group=conf.log_group, name=str(conf.seed),
                   config=OmegaConf.to_container(conf))
    else:
        wandb.init(reinit=True, project=conf.project_name, name=conf.run_name, config=OmegaConf.to_container(conf))

    # Run training
    for step in tqdm(range(1, conf.total_generations + 1)):
        runner, metrics = _runnner_run(runner, es_conf, env_params)
        metrics = jax.device_get(metrics)
        wandb.log(metrics, step=step)

        if not (step % conf.save_every):
            fn = conf.save_model_path + str(step)
            save_obj_to_file(fn, dict(
                conf=conf,
                state=dict(
                    normalizer_state=runner.normalizer_state,
                    params=runner.params
                )
            ))
            wandb.save(fn)


if __name__ == "__main__":
    main(OmegaConf.from_cli())
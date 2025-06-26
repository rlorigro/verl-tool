# DAPO support

we add DAPO support for training. 

- To enable dynamic batch, add following config in your training script:

```bash
    +algorithm.filter_groups.enable=True \
    +algorithm.filter_groups.metric='seq_final_reward' \
    +algorithm.filter_groups.max_num_gen_batches=0 \
```

- to mask the overlong trajectory (avoid training on it), add following config in your training script:

```bash
    +actor_rollout_ref.agent.mask_overlong_loss=True \
```


- to clip higher reward, add following config in your training script:

```bash
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
```

- to do token-level loss aggregation, add following config in your training script:

```bash
    actor_rollout_ref.actor.loss_agg_mode='token-mean' \
```

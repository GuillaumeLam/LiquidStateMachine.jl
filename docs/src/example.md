# Example

## Minimal usage:

``` julia
using LiquidStateMachine

params = LSM_Params(4,2)
lsm = LSM(params)
x = [1,2,5,3]

lsm(x)
```

## Ultra Minimal usage:
``` julia
using LiquidStateMachine

x = [1,2,5,3]

LSM(LSM_Params(4,2))(x)
```


## ReinforcementLearning.jl usage:

As LiquidStateMachine.jl is differentiable thanks to Flux and Zygote, anywhere a Flux model can be passed, a LiquidStateMachine.jl LSM can be passed:

``` julia
using LiquidStateMachine
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses

rng = StableRNG(123)
env = CartPoleEnv(; T = Float32, rng = rng)
ns, na = length(state(env)), length(action_space(env))

params = LSM_Params(ns,na,"cartpole")
lsm = LSM(params)

policy = Agent(
    policy = QBasedPolicy(
        learner = BasicDQNLearner(
            approximator = NeuralNetworkApproximator(
                model = lsm |> cpu,
                optimizer = ADAM(),
            ),
            batch_size = 32,
            min_replay_history = 100,
            loss_func = huber_loss,
            rng = rng,
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            ϵ_stable = 0.01,
            decay_steps = 500,
            rng = rng,
        ),
    ),
    trajectory = CircularArraySARTTrajectory(
        capacity = 1000,
        state = Vector{Float32} => (ns,),
    ),
)
stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
hook = TotalRewardPerEpisode()

run(policy, env, stop_condition, hook)
```

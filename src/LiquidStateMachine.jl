module LiquidStateMachine

using CUDA
using Distributions
using DifferentialEquations
using Flux
using LinearAlgebra
using Random
using StableRNGs
using WaspNet
using Zygote

include("util.jl")
include("params.jl")
include("spike_interpreter.jl")
include("lsm.jl")

export LSM, LSM_Params


using Plots

include("liquid_util.jl")

export SP,eigen_spectrum

end # module

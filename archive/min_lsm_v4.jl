# install: julia> Pkg.add("https://github.com/AStupidBear/SpikingNeuralNetworks.jl")

using Plots
using Random
using SpikingNeuralNetworks
SNN.@load_units

# N = 1000
# E1 = SNN.IF(;N = N)
# E2 = SNN.IF(;N = N)
# EE = SNN.SpikingSynapse(E1, E2, :ge)
# for n = 1:E1.N SNN.connect!(EE, n, n) end
# SNN.monitor([E1, E2], [:fire])
# SNN.monitor(EE, [:W])

# @time for t = 1:N
# 	E1.v[t] = 100
# 	E2.v[N - t + 1] = 100
# 	SNN.train!([E1, E2], [EE], 0.5ms, (t - 1) * 0.5ms)
# end
# SNN.raster([E1, E2])
# ΔW = EE.records[:W][end]
# plot(ΔW)

# spike_train = E1.:fire
# plot(spike_train)

E = SNN.IF(;N = 3200, param = SNN.IFParameter(;El = -49mV))
I = SNN.IF(;N = 800, param = SNN.IFParameter(;El = -49mV))
EE = SNN.SpikingSynapse(E, E, :ge; σ = 60*0.27/10, p = 0.02)
EI = SNN.SpikingSynapse(E, I, :ge; σ = 60*0.27/10, p = 0.02)
IE = SNN.SpikingSynapse(I, E, :gi; σ = -20*4.5/10, p = 0.02)
II = SNN.SpikingSynapse(I, I, :gi; σ = -20*4.5/10, p = 0.02)
P = [E, I]
C = [EE, EI, IE, II]

SNN.monitor([E, I], [:fire])
SNN.monitor([EE, EI, IE, II], [:W])

SNN.sim!(P, C; duration = 1second)
SNN.raster(P)
SNN.train!(P, C; duration = 1second)

ΔW = EE.records[:W][end]
plot(ΔW)
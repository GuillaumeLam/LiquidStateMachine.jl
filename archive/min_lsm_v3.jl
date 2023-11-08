# from: https://docs.sciml.ai/SciMLTutorialsOutput/html/models/08-spiking_neural_systems.html

using DifferentialEquations
# load error :/
using Plots
gr()

function lif(u,p,t);
	gL, EL, C, Vth, I = p
	(-gL*(u-EL)+I)/C
end

function thr(u,t,integrator)
	integrator.u > integrator.p[4]
end

function reset!(integrator)
	integrator.u = integrator.p[2]
end

threshold = DiscreteCallback(thr,reset!)
current_step= PresetTimeCallback([2,15],integrator -> integrator.p[5] += 210.0)
cb = CallbackSet(current_step,threshold)

u0 = -75
tspan = (0.0, 40.0)
# p = (gL, EL, C, Vth, I)
p = [10.0, -75.0, 5.0, -55.0, 0]

prob = ODEProblem(lif, u0, tspan, p, callback=cb)

sol = solve(prob)

plot(sol)


function eigen_spectrum(lsm::LSM)
	m = lsm.reservoir.layers[2].W[2]

	eigen_matrix = eigen(Matrix(m)).vectors

	eigen_imag = imag(eigen_matrix)
	eigen_real = real(eigen_matrix)

	function circShape(h,k,r)
	    θ=LinRange(0,2*π, 500)
	    h .+ r*sin.(θ), k .+ r*cos.(θ)
	end

	scatter(vec(x_m),vec(y_m), size=(500,500))
	plot!(circShape(0,0,1), seriestype=[:shape,], lw = 0.5,c=:green,legend=false,fillalpha=0.2,aspect_ration=1)
end

function SP(lsm::LSM)
	base = [0,0,0,0]
	v = [4,5,6,7]
	w = [100,30,50,25]


	function xᵐ(i)
		a = lsm.reservoir(lsm.st_gen(lsm.preprocessing(base)))
		b = lsm.reservoir(lsm.st_gen(lsm.preprocessing(i)))
		c = lsm.reservoir(lsm.st_gen(lsm.preprocessing(base)))

		return # append a,b,c length wise
	end

	xᵐᵤ = xᵐ(u)
	xᵐᵥ = xᵐ(v)
	xmw = xᵐ(u)

	x = [xᵐᵤ,xᵐᵥ,xmw]

	plot(1:length(x[1]),x)
end

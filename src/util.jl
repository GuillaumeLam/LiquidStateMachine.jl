struct LSM_Params{I<:Real, F<:Real}
    n_in::I
    res_in::I

    ne::I
    ni::I

    res_out::I
    n_out::I

    K::I
    C::I

    PE_UB::F
    EE_UB::F
    EI_UB::F
    IE_UB::F
    II_UB::F

    LSM_Params(
        n_in::I,res_in::I,ne::I,ni::I,res_out::I,n_out::I,K::I,C::I,
        PE_UB::F,EE_UB::F,EI_UB::F,IE_UB::F,II_UB::F
        ) where {I<:Real,F<:Real} = new{I,F}(n_in,res_in,ne,ni,res_out,n_out,K,C,PE_UB,EE_UB,EI_UB,IE_UB,II_UB)

    LSM_Params(
            n_in::I,res_in::I,ne::I,ni::I,res_out::I,n_out::I,K::I,C::I
            ) where {I<:Number} = LSM_Params(n_in,res_in,ne,ni,res_out,n_out,K,C,0.6,0.005,0.25,0.3,0.01)

    LSM_Params(n_in::I, n_out::I, env::String) where {I<:Real} = (
        if env == "cartpole"
            return LSM_Params(n_in,32,120,30,32,n_out,3,4)
        end
    )
end

function create_conn(val, avg_conn, ub, n)
    if val < avg_conn/n
        return rand()*ub
    else
        return 0.
    end
end

function genPositive(arr::AbstractVector)
	s = map(arr) do e
		if e < 0
			return [0, abs(e)]
		else
			return [e, 0]
		end
	end

	return vcat(s...)
end

function genPositive(m::AbstractMatrix)
	pm = mapslices(m, dims=1) do arr
		return genPositive(arr)
	end
	return pm
end

function genCapped(arr::AbstractVector, caps::AbstractVector)
	h = broadcast(arr, caps) do val, cap
		if val <= -cap
			return -cap
		elseif val >= cap
			return cap
		else
			return val
		end
	end

	return h
end

function genCapped(m::AbstractMatrix, caps::AbstractVector)
	cm = mapslices(m, dims=1) do arr
		return genCapped(arr, caps)
	end
	return cm
end

function discretize(arr::AbstractVector, caps::AbstractVector, n)
	s = broadcast(arr,caps) do a, c
		val = begin
			if a <= -c
				-c
			elseif a >= c
				c
			else
				a
			end
		end

		rng = collect(-c:(2*c/n):c)

		idx = searchsorted(rng,val).stop

		v = zeros(n+1)
		v[idx] = abs(rng[idx])

		return v
	end

	return vcat(s...)
end

function discretize(m::AbstractMatrix, caps::AbstractVector, n)
	dm = mapslices(m, dims=1) do arr
		return discretize(arr, caps, n)
	end
	return dm
end

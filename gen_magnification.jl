using CSV, DataFrames
using Random
using DelimitedFiles
using LinearAlgebra: searchsortedlast # Used for efficient index lookup in sorted arrays
using Base: size, length, time, println, reverse, zeros, ones

# Helper function to perform 1D linear interpolation (simulating np.interp behavior for CDF sampling)
# Since the input xp_sorted (Psupmu column, usually a CDF) is sorted, we can use searchsortedlast.
function linear_interp_sampling(x_target, xp_sorted::AbstractVector, fp_values::AbstractVector)
    
    # 1. Find the index 'i' such that xp_sorted[i] <= x_target < xp_sorted[i+1]
    i = searchsortedlast(xp_sorted, x_target)
    N = length(xp_sorted)
    y_first = @allowscalar fp_values[1]
    y_last = @allowscalar fp_values[N]

    # Build interpolation indices without traced control-flow branches.
    i_lo = ifelse(i <= 0, 1, ifelse(i >= N, N - 1, i))
    i_hi = i_lo + 1

    x1 = @allowscalar xp_sorted[i_lo]
    y1 = @allowscalar fp_values[i_lo]
    x2 = @allowscalar xp_sorted[i_hi]
    y2 = @allowscalar fp_values[i_hi]

    denom = x2 - x1
    y_interp = ifelse(denom == 0, y1, y1 + ((y2 - y1) / denom) * (x_target - x1))

    return ifelse(i <= 0, y_first, ifelse(i >= N, y_last, y_interp))
end

function update_mu_for_bin(mu, indz_gal_1based, k_val, Psupmu, mu_grid)
    mask = indz_gal_1based .== k_val

    Psupmu_col = @allowscalar Psupmu[:, k_val]
    u = rand(length(mu))
    mu_new = similar(u)
    @trace for I in eachindex(u)
        @allowscalar mu_new[I] =
            linear_interp_sampling(@allowscalar(u[I]), Psupmu_col, mu_grid)
    end

    return ifelse.(mask, mu_new, mu)
end

function gen_magnification(cat, mag_params, magnify = true)

    tstart = time()

    println("Generate magnification...")

    if magnify == true
        (z_grid, mu_grid, Psupmu) = mag_params
        # Calculate redshift grid index (emulating np.interp/round/fix)
        # We find the 1-based index in z_grid where the redshift falls, clamping to 1:N_z
        # searchsortedlast gives the 1-based index (i) such that z_grid[i] <= value.
        # Use a scalar loop instead of broadcast(Ref(...), ...) to avoid the traced Ref broadcast path.
        N_z = length(z_grid)
        redshift = cat.redshift
        indz_gal_1based = map(z -> searchsortedlast(z_grid,z), cat.redshift)
        println("Type of indz_gal_1based: ", typeof(indz_gal_1based))
        
        Ngal = size(cat, 1) # np.shape(cat)
        mu = zero.(redshift)
        
        println("Looping over ", N_z, " redshift indices...")
        # Loop over all redshift bins; execute work only when bin is populated.
        @trace for k_val = 1:N_z
            mu = update_mu_for_bin(mu, indz_gal_1based, k_val, Psupmu, mu_grid)
        end
    else
        println("The magnify keyword is set to False. mu = 1 for all the sources.")
        mu = one.(cat.redshift)
    end

    # Assign the new column to the DataFrame (cat = cat.assign(mu = mu))
    cat[!, :mu] = mu
    tstop = time()
    println(size(cat, 1), " magnifications generated in ", tstop - tstart, "s")
    
    return cat
end

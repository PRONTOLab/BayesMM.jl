using CSV, DataFrames
using Random
using DelimitedFiles
using LinearAlgebra: searchsortedlast # Used for efficient index lookup in sorted arrays
using Base: size, length, time, println, reverse, zeros, ones

# Helper function to perform 1D linear interpolation (simulating np.interp behavior for CDF sampling)
# Since the input xp_sorted (Psupmu column, usually a CDF) is sorted, we can use searchsortedlast.
function linear_interp_sampling(x_target::Float64, xp_sorted::AbstractVector{Float64}, fp_values::AbstractVector{Float64})
    
    # 1. Find the index 'i' such that xp_sorted[i] <= x_target < xp_sorted[i+1]
    i = searchsortedlast(xp_sorted, x_target)
    N = length(xp_sorted)
    
    # Handle extrapolation (flat boundaries, matching numpy.interp default behavior)
    if i == 0
        return fp_values[1]
    elseif i == N
        return fp_values[N]
    end
    
    # If i points to the last element but x_target > xp_sorted[N]
    if i == N - 1 && x_target > xp_sorted[N]
        return fp_values[N]
    end
    
    # 2. Linear interpolation between point i and i+1
    x1, y1 = xp_sorted[i], fp_values[i]
    x2, y2 = xp_sorted[i+1], fp_values[i+1]
    
    # Guard against division by zero if x1 == x2 (should be rare/impossible for proper CDF data)
    if x1 == x2
        return y1
    end
    
    slope = (y2 - y1) / (x2 - x1)
    return y1 + slope * (x_target - x1)
end


function gen_magnification(cat, params, magnify = true)

    tstart = time()

    println("Generate magnification...")

    if magnify == true

        # Load the magnification grid (np.loadtxt -> DelimitedFiles.readdlm)
        #data = readdlm(params["path_mu_file"], ',')
        #data_mat = parse.(Float64, data) # Ensure matrix of Floats
        df = CSV.read(params["path_mu_file"], DataFrame, comment="#")
        data_mat = Matrix{Float64}(df) 


        # 1. Slice data: Python [0, 1:] -> Julia [1, 2:end]
        z_grid = data_mat[1, 2:end] 
        
        # 2. Reverse (flip): Python axis=0 -> Julia dims=1 (rows)
        mu_grid = reverse(data_mat[2:end, 1])
        Psupmu = reverse(data_mat[2:end, 2:end]; dims=1) 
        
        # 3. Calculate redshift grid index (emulating np.interp/round/fix)
        # We find the 1-based index in z_grid where the redshift falls, clamping to 1:N_z
        
        # searchsortedlast gives the 1-based index (i) such that z_grid[i] <= value
        indz_gal_1based = searchsortedlast.(Ref(z_grid), cat.redshift)
        
        N_z = length(z_grid)
        # Clamp ensures indices are within bounds [1, N_z]
        indz_gal_1based = clamp.(indz_gal_1based, 1, N_z)
        
        # To match Python's internal 0-based index list for iteration, convert the value:
        indz_gal_0based = indz_gal_1based .- 1 
        
        # Python: indz_set = list(set(indz_gal)) (set of unique 0-based indices)
        indz_set_0based = unique(indz_gal_0based)

        Ngal = size(cat, 1) # np.shape(cat)
    
        mu = zeros(Float64, Ngal)
        
        println("Looping over ", length(indz_set_0based), " unique redshift indices...")
        
        # Loop over the unique 0-based index values present in the catalog (`k_val`)
        # NOTE: ProgressBar replaced by standard Julia 'for' loop
        for k_val in indz_set_0based
            
            # Find 1-based row indices where the galaxy redshift index matches k_val
            mask = (indz_gal_0based .== k_val)
            gal_indices = findall(mask)
            
            Xuni = rand(length(gal_indices))
            
            # 1-based column index for Psupmu:
            col_idx_1based = Int(k_val) + 1 
            
            Psupmu_col = Psupmu[:, col_idx_1based]
            
            # Perform interpolation/sampling and broadcast over all galaxies in this index bin
            mu[gal_indices] .= linear_interp_sampling.(Xuni, Ref(Psupmu_col), Ref(mu_grid))
        end

    else

        println("The magnify keyword is set to False. mu = 1 for all the sources.")
        Ngal = size(cat, 1)
        mu = ones(Float64, Ngal) # np.ones(len(cat))

    end

    # Assign the new column to the DataFrame (cat = cat.assign(mu = mu))
    cat[!, :mu] = mu
        
    tstop = time()

    println(size(cat, 1), " magnifications generated in ", tstop - tstart, "s")
    
    return cat
end
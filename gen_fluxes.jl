using Distributed # For parallel processing (replaces Python's multiprocessing)
using Unitful, UnitfulAstro # For unit handling (replaces astropy.units)
using PhysicalConstants # For constants (replaces cst.c)
# Assuming Base.Threads.cpu_count() is available or defined 
const CPU_COUNT = Sys.CPU_THREADS # Or manually set addprocs(N)
using Statistics
using Cosmology
#using ASDF
using FITSIO
using HDF5
using Reactant

import PhysicalConstants.CODATA2022: c_0 as c # Speed of light constant

# --- Setup Assumptions ---
# Assuming these are defined globally or within the scope of gen_fluxes.
#const cosmo_model = Cosmology.FlatLCDM(h=0.677, OmegaM=0.307, OmegaK=0.0) 

function grouper(N::Int, n::Int)
    if n <= 0
        throw(ArgumentError("Number of chunks (n) must be positive"))
    end
    N_per_chunk = ceil(Int, N / n)
    chunks = []
    start_idx = 1
    while start_idx <= N
        end_idx = min(N, start_idx + N_per_chunk - 1)
        push!(chunks, start_idx:end_idx)
        start_idx = end_idx + 1
    end
    return [chunk for chunk in chunks if !isempty(chunk)]
end

# The worker function needs to be available on all processes if pmap is used.
# If running locally, you might need to wrap the definition in @everywhere
# or ensure it's defined globally before starting workers.

function interp_sorted_flux(x_target, xp_sorted::AbstractVector, fp_values::AbstractVector)
    i = searchsortedlast(xp_sorted, x_target)
    n = length(xp_sorted)

    i_lo = ifelse(i <= 0, 1, ifelse(i >= n, n - 1, i))
    i_hi = i_lo + 1

    x1 = @allowscalar xp_sorted[i_lo]
    x2 = @allowscalar xp_sorted[i_hi]
    y1 = @allowscalar fp_values[i_lo]
    y2 = @allowscalar fp_values[i_hi]

    d = x2 - x1
    y_interp = ifelse(d == 0, y1, y1 + ((y2 - y1) / d) * (x_target - x1))

    return ifelse(i <= 0, @allowscalar(fp_values[1]), ifelse(i >= n, @allowscalar(fp_values[n]), y_interp))
end

function worker(lambda_list, issb, Uindex, sed_tables, redshift)
    N_gal = length(redshift)
    N_lambda = length(lambda_list)
    
    # 2. Calculate rest-frame wavelengths and rest-frame frequencies (unitless).
    redshift_col = reshape(redshift, N_gal, 1) # Force column vector for broadcasting

    # lambda_list is in micron. Convert to meter for nu = c / lambda.
    lambda_rest_um = reshape(lambda_list, 1, N_lambda) ./ (1.0 .+ redshift_col)
    lambda_rest_m = lambda_rest_um .* 1.0e-6
    c_m_per_s = 299792458.0
    nu_rest_Hz = c_m_per_s ./ lambda_rest_m

    # Hold interpolated SED values in traced storage.
    nuLnu = zero.(lambda_rest_um)

    # 4. Interpolation Loop (Replacing np.interp with Interpolations.jl)
    @trace for i in 1:N_gal
        # Data points for interpolation (X-axis)
        lambda_interp_x = sed_tables.lambda

        # Pick the MS/SB table directly from the pre-extracted SED arrays.
        is_sb = @allowscalar issb[i]
        u_idx = @allowscalar Uindex[i]
        sed_data_ms = sed_tables.nuLnu_MS[:, u_idx]
        sed_data_sb = sed_tables.nuLnu_SB[:, u_idx]
        sed_data = ifelse.(is_sb, sed_data_sb, sed_data_ms)
        
        # Apply interpolation element-wise using a traced-safe kernel.
        @trace for j in 1:N_lambda
            x = @allowscalar lambda_rest_um[i, j]
            y = interp_sorted_flux(x, lambda_interp_x, sed_data)
            @allowscalar setindex!(nuLnu, y, i, j)
        end
    end
    

    # 5. Return nuLnu / nu as a unitless numeric array.
    return nuLnu ./ nu_rest_Hz
end

function gen_Snu_arr(lambda_list, sed_tables, redshift, LIR, Umean, Dlum, issb)
    N_total = length(redshift) 

    # 2. Uindex Calculation (Broadcasted operations replacing NumPy)
    
    # sed_tables.Umean[3] is the first element (replaces Python's  due to 1-indexing)
    Umean_min = @allowscalar sed_tables.Umean[3] # Accesses the first element (Julia is 1-indexed)
    dU = sed_tables.dU
    
    # Uindex calculation: round, division, and subtraction are all broadcasted
    Uindex = round.((Umean .- Umean_min) ./ dU)
    
    # Cast to integer (NumPy astype(int) -> Julia Int.())
    Uindex = Int.(Uindex)
    
    # Clamping (NumPy np.maximum/minimum -> Julia max./min. using 1-based bounds)
    Umax_index = length(sed_tables.Umean)
    
    # Clamp minimum index to 1 (replaces Python's 0)
    Uindex = max.(Uindex, 1) 
    
    # Clamp maximum index (replaces Python's np.size(arr) - 1)
    Uindex = min.(Uindex, Umax_index)
    
    # 3. Serial execution setup (avoid pmap during tracing).
    concatenated_worker_output = worker(
        lambda_list,
        issb,
        Uindex,
        sed_tables,
        redshift
    )

    # 4. Luminosity Calculation (Lnu, in W/Hz as plain Float64)
    L_sun_W = 3.828e26

    # Reshape LIR to a column vector (replaces np.array(LIR)[:, np.newaxis])
    LIR_col = reshape(LIR, N_total, 1) 
    
    # concatenated_worker_output encodes the 1/Hz term numerically.
    Lnu = L_sun_W .* LIR_col .* concatenated_worker_output

    # 5. Flux Density Calculation (Snu_arr)

    # Numerator: Lnu * ( 1 + redshift) * (1/ (4 * pi))
    redshift_col = reshape(redshift, N_total, 1)
    Numerator = Lnu .* (1.0 .+ redshift_col) .* (1.0 / (pi * 4.0))

    # Dlum is expected in meters as plain Float64.
    Dlum_m_squared = Dlum .^ 2
    Denominator = reshape(Dlum_m_squared, N_total, 1) # Reshape for division broadcasting

    # Final Flux Density Calculation in Jy (1 Jy = 1e-26 W m^-2 Hz^-1).
    Snu_arr = (Numerator ./ Denominator) .* 1.0e26

    return Snu_arr
end



function gen_LFIR_vec(ratio_tables, redshift, LIR, Umean, issb)
    # 1. Initialize LFIR array (using zeros_like is equivalent to zeros(size(redshift)))
    # We assume redshift is a 1D vector of numerical type (e.g., Float64)
    LFIR = zeros(eltype(redshift), size(redshift)) 

    # 2. Identify Selection Indices (replacing np.where and == True/False)
    
    # In Julia, issb is a Vector{Bool}. We use findall for 1-based indices.
    # Note: issb == true is typically written simply as `issb` in Julia.
    selSB = findall(issb)      # Indices where issb is true (Starburst)
    selMS = findall(.!issb)    # Indices where issb is false (Main Sequence)
    
    # 3. Uindex Calculation (Vectorized operations replacing np.round, np.astype)

    # Note: Array indexing in Julia starts at 1, so the first element is [6], not  [2].
    # We assume ratio_tables.Umean is an array/vector.
    Umean_min = ratio_tables.Umean[6]#[6] # Accessing the first element (index 1)
    
    # Python: Uindex = np.round((Umean - ratio_tables["Umean"]) / ratio_tables["dU"])
    # Julia uses broadcasting (dot notation) for element-wise operations [7].
    Uindex = round.((Umean .- Umean_min) ./ ratio_tables.dU)
    
    # Python: Uindex.astype(int) is replaced by broadcasting Int.()
    Uindex = Int.(Uindex) 

    # 4. Clamping Indices (Replacing np.maximum, np.minimum, and array size calculation)

    # Maximum valid index in Julia is length(array) (replaces Python's np.size(arr) - 1) [8].
    Umax_index = length(ratio_tables.Umean) 
    
    # Clamp minimum index to 1 (replaces Python's 0)
    Uindex = max.(Uindex, 1) 
    
    # Clamp maximum index
    Uindex = min.(Uindex, Umax_index)
    
    # 5. Luminosity Calculation (Replacing array indexing and multiplication)
    
    # Python slices selectSB (the tuple containing the index array) are not needed in Julia.
    # We use the index vectors selSB and selMS to access and update elements simultaneously.
    
    # Starburst (SB) calculation: LFIR[SB indices] = LIR[SB indices] * ratio[Uindex[SB indices]]
    @views LFIR[selSB] = LIR[selSB] .* ratio_tables.LFIR_LIR_ratio_SB[Uindex[selSB]]
    
    # Main Sequence (MS) calculation: LFIR[MS indices] = LIR[MS indices] * ratio[Uindex[MS indices]]
    @views LFIR[selMS] = LIR[selMS] .* ratio_tables.LFIR_LIR_ratio_MS[Uindex[selMS]]

    return LFIR
end


function load_sed_pickle_equivalent(file_path::String)
    
    println("Loading data from hdf5 file: $file_path")

    # 2. Define the file path
    hdf5_file_path = file_path
    
    # 3. Create a Julia dictionary to hold the data
    SEDData = Dict{String, Any}()
    
    # 4. Read the HDF5 file and populate the dictionary
    try
        h5open(hdf5_file_path, "r") do f
            for key in keys(f)
                # read() loads the HDF5 dataset into memory (e.g., as a Julia Array)
                SEDData[key] = read(f[key])
            end
        end
    
        println("Successfully loaded SEDData from HDF5!")
        println("Type: ", typeof(SEDData))
        #println("Data: ", SEDData)
    catch e
        println("Error loading HDF5 file: ", e)

    end

    return SEDData

end

#List of parameters used to compute flux and preprocess stuff
function parse_flux_params(cosmo_model,params)
    _p(k) =
        let v = params[k]
            v isa AbstractString ? parse(Float64, v) : Float64(v)
        end

    sed_dict = load_sed_pickle_equivalent(params["SED_file"])
    sed_tables = (
        lambda=sed_dict["lambda"],
        dU=sed_dict["dU"],
        Umean=sed_dict["Umean"],
        nuLnu_MS=sed_dict["nuLnu_MS_arr"],
        nuLnu_SB=sed_dict["nuLnu_SB_arr"],
    )
    ratio_dict = load_sed_pickle_equivalent(params["ratios_file"])
    ratio_tables = (
        Umean=ratio_dict["Umean"],
        dU=ratio_dict["dU"],
        LFIR_LIR_ratio_MS=ratio_dict["LFIR_LIR_ratio_MS"],
        LFIR_LIR_ratio_SB=ratio_dict["LFIR_LIR_ratio_SB"],
    )

    return (
        UmeanSB=_p("UmeanSB"),
        UmeanMSz0=_p("UmeanMSz0"),
        alphaMS=_p("alphaMS"),
        zlimMS=_p("zlimMS"),
        sigma_logUmean=_p("sigma_logUmean"),
        SFR2LIR=_p("SFR2LIR"),
        lambda_list=params["lambda_list"],
        sed_tables=sed_tables,
        ratio_tables=ratio_tables,
        cosmo_model = cosmo_model
    )
end

"""
Translated function to generate SED properties and fluxes.
"""
function gen_fluxes(cat::DataFrame, p)
    # Read parameters
    (; UmeanSB, UmeanMSz0, alphaMS, zlimMS,
        sigma_logUmean,
        SFR2LIR,
        lambda_list,
        sed_tables,
        ratio_tables,
        cosmo_model
    ) = p 

    tstart = time() # Start timer [5]

    println("Generate SED properties and fluxes...")

    # --- 1. Compute zlimSB (Scalar Operations) ---
    zlimSB = (log10(UmeanSB) - log10(UmeanMSz0)) / alphaMS
    zlimMS = zlimMS

    @trace if zlimSB > zlimMS
        # Julia string interpolation [2]
        println("zlim SB (when UMS = USB)= $(zlimMS)")
        zlimSB = 9999.0 
    end

    Ngal = nrow(cat) # Equivalent to len(cat)

    # --- 3. Draw <U> parameters (Vectorized Operations) ---
    println("Draw <U> parameters...")

    Umean = zeros(Ngal) # Equivalent to np.zeros

    # Reactant-friendly mask updates avoid findall(::TracedRArray{Bool}).
    mask_MS_SBhighz = (.!(cat.issb)) .| (cat.redshift .>= zlimSB)
    Umean_MS_SBhighz = 10.0 .^ (log10(UmeanMSz0) .+ alphaMS .* min.(cat.redshift, zlimMS))
    Umean = ifelse.(mask_MS_SBhighz, Umean_MS_SBhighz, Umean)

    mask_SBlowz = cat.issb .& (cat.redshift .< zlimSB)
    Umean = ifelse.(mask_SBlowz, UmeanSB, Umean)

    # Add log-normal scatter (np.random.normal -> randn)
    scatter_factor = 10.0 .^ (sigma_logUmean .* randn(Ngal))
    Umean .*= scatter_factor 

    cat[!, :Umean] = Umean


    # --- 5. Generate LIR ---
    println("Generate LIR...")
    cat[!, :LIR] = SFR2LIR .* cat[!, :SFR]

    # --- 6. Generate Flux Array (Assumed helper function call) ---
    println("Generate Flux Array ...")    
    Snu_arr = gen_Snu_arr(
        lambda_list, 
        sed_tables,
        cat.redshift, 
        cat.mu .* cat.LIR, 
        cat.Umean, 
        cat.Dlum, 
        cat.issb
    )

    for (i,v) in enumerate(lambda_list)
        colname = "S$(v)"
        colsym = Symbol(colname)
        # Assign the flux vector (Snu_arr is assumed to be Ngal x N_lambda matrix)
        # Note: We use in-place column creation
        cat[!, colsym] =  Snu_arr[:, i]
    end

    # generate LFIR (40-400 microns)
    println("Generate LFIR...")
    
    # Assign the new column derived from the vector function
    cat[!, :LFIR] = gen_LFIR_vec(
        ratio_tables, 
        cat.redshift, 
        cat.LIR, 
        cat.Umean, 
        cat.issb
    )

    tstop = time() # Stop timer

    # Print final timing and results using Julia's println and interpolation
    # Equivalent to Python's len(cat) is nrow(cat)
    println("SED properties of $(nrow(cat)) generated in $(tstop - tstart)s")

    return cat
end

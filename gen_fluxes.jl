using Distributed # For parallel processing (replaces Python's multiprocessing)
using Unitful, UnitfulAstro # For unit handling (replaces astropy.units)
using PhysicalConstants # For constants (replaces cst.c)
using Interpolations 
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

function worker(ks, lambda_list, issb, Uindex, sed_tables, redshift)
    # 1. Filter None indices (ks are 1-based indices here)
    ks = filter(!isnothing, ks) # Literal translation of filtering None [1, 2]

    N_gal = length(ks)
    N_lambda = length(lambda_list)
    
    # Initialize nuLnu (note: starting with dimensionless array, units are added later)
    nuLnu = zeros(Float64, N_gal, N_lambda) 

    # 2. Calculate lambda_rest (vectorized operations replacing np.newaxis and broadcasting)
    # np.array(redshift)[ks, np.newaxis] is replaced by extracting the subset and reshaping to N_gal x 1.
    redshift_subset = redshift[ks]
    redshift_col = reshape(redshift_subset, N_gal, 1) # Force column vector for broadcasting

    # Apply broadcasting (./, .+) and add units (u"μm")
    # Python: lambda_list / (1 + np.array(redshift)[ks, np.newaxis]) * u.um
    # size(lambda_list) = (10, )
    
    # size(redshift_col) = (N)
    # size(lambda_rest) = (N, 10)
    lambda_rest = ( reshape(lambda_list, 1, 10) ./ (1.0 .+ reshape(redshift_col, length(redshift_col), 1))).*u"μm"

    #println(length(lambda_rest))
    #println(lambda_rest)

    # 3. Calculate nu_rest_Hz 
    # Python: (cst.c * u.m/u.s) / lambda_rest.to(u.m)
    # Use Julia's unit conversion (`|> u"m"`) and constant access (`c`)
    c_speed = uconvert(u"m/s", c) # Ensure c is in m/s for clean calculation
    
    # nu_rest_Hz is calculated as a matrix of Quantities
    nu_rest_Hz = (c_speed ./ (lambda_rest .|> u"m")) .|> u"Hz"

    # 4. Interpolation Loop (Replacing np.interp with Interpolations.jl)
    for i in 1:N_gal
        k = ks[i] # Current 1-based index in the full catalog
        
        # NOTE ON INDEXING: Julia is 1-indexed. Uindex is already 1-based and clamped.
        
        # Data points for interpolation (X-axis)
        lambda_interp_x = sed_tables.lambda

        # Pick the MS/SB table directly from the pre-extracted SED arrays.
        sed_table = ifelse(issb[k], sed_tables.nuLnu_SB, sed_tables.nuLnu_MS)
        sed_data = sed_table[:, Uindex[k]]
        
        # Create interpolation object (Linear is standard for np.interp; Line() handles extrapolation)
        # Assuming SED_dict[stype[k]] is Matrix{Float64}, hence sed_data is Vector{Float64}
        interp_itp = LinearInterpolation(lambda_interp_x, sed_data, extrapolation_bc=Line())

        
        # Values to interpolate (rest-frame lambdas for this galaxy)
        lambda_rest_row = ustrip(lambda_rest[i, :])

        
        # Apply interpolation element-wise (using broadcasting .() on the interpolation object)
        nuLnu_row = interp_itp.(lambda_rest_row) 
        
        # Store result (now just Float64, we will add units back later)
        nuLnu[i, :] = nuLnu_row
    end
    

    # Re-add assumed units for nuLnu output to allow the final division to work
    nuLnu_quantified = nuLnu .* u"W"

    # 5. Final Calculation and Return Value
    # Python: (nuLnu / nu_rest_Hz).value
    # Element-wise division (./). The result is dimensionless because units cancel to Hz^-1
    # .|> Unitful.NoUnits converts the Quantity to a pure Float (replaces .value in Python)
    return ustrip(nuLnu_quantified ./ nu_rest_Hz) 
end

function gen_Snu_arr(lambda_list, sed_tables, redshift, LIR, Umean, Dlum, issb)
    N_total = length(redshift) 

    # 2. Uindex Calculation (Broadcasted operations replacing NumPy)
    
    # sed_tables.Umean[3] is the first element (replaces Python's  due to 1-indexing)
    Umean_min = sed_tables.Umean[3] # Accesses the first element (Julia is 1-indexed)
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
    
    # 3. Parallel Execution Setup (Replacing Python Pool/map)

    # Prepare input arguments (range(len(redshift)) -> 1:N_total)
    index_range = 1:N_total
    
    # Determine chunk size (// cpu_count() -> div(N_total, CPU_COUNT))
    chunk_size = div(N_total, CPU_COUNT) 
    
    # Create index chunks using the Julia grouper
    index_chunks = grouper(N_total, CPU_COUNT)
    #println(size(lambda_list))
    #println(size(Uindex))
    #println(size(redshift))
    
    Worker_partial(ks) = worker(
        ks, 
        lambda_list, 
        issb, 
        Uindex, 
        sed_tables, 
        redshift
    )
    
    # Execute the worker function in parallel (pmap replaces pool.map)
    L_nu_over_nu_chunks = pmap(Worker_partial, index_chunks) 
    
    # Concatenate the results (np.concatenate -> vcat)
    # The result is a Matrix{Float64} of dimensionless L_nu/nu values
    concatenated_worker_output = vcat(L_nu_over_nu_chunks...) 

    # 4. Luminosity Calculation (Lnu)

    # L_sun_W = 3.828e26 * u.W 
    L_sun_W = 3.828e26 * u"W" 

    # Reshape LIR to a column vector (replaces np.array(LIR)[:, np.newaxis])
    LIR_col = reshape(LIR, N_total, 1) 
    
    # Lnu calculation in W/Hz (Julia units are applied directly)
    Lnu = L_sun_W .* LIR_col .* concatenated_worker_output ./ u"Hz"

    # 5. Flux Density Calculation (Snu_arr)

    # Numerator: Lnu * ( 1 + redshift) * (1/ (4 * pi))
    redshift_col = reshape(redshift, N_total, 1)
    Numerator = Lnu .* (1.0 .+ redshift_col) .* (1.0 / (pi * 4.0))

    # Denominator: ((np.asarray(Dlum) * u.Mpc).to(u.m)) ** 2
    # Dlum is assumed to be a vector of Quantity{Float64, L, ...} (e.g., Mpc units)
    
    # Convert Dlum to meters (using .|> u"m") and square it (using .^ 2)
    Dlum_m_squared = (Dlum .|> u"m") .^ 2
    Denominator = reshape(Dlum_m_squared, N_total, 1) # Reshape for division broadcasting

    # Final Flux Density Calculation
    # Snu_arr = ( Numerator / Denominator ).to(u.Jy)
    Snu_arr = (Numerator ./ Denominator) .|> u"Jy" # Element-wise division and conversion to Jansky

    return Snu_arr
end



function gen_LFIR_vec(LIR_LFIR_ratio_dict, redshift, LIR, Umean, issb)
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
    # We assume LIR_LFIR_ratio_dict["Umean"] is an array/vector.
    Umean_min = LIR_LFIR_ratio_dict[:"Umean"][6]#[6] # Accessing the first element (index 1)
    
    # Python: Uindex = np.round((Umean - LIR_LFIR_ratio_dict["Umean"]) / LIR_LFIR_ratio_dict["dU"])
    # Julia uses broadcasting (dot notation) for element-wise operations [7].
    Uindex = round.((Umean .- Umean_min) ./ LIR_LFIR_ratio_dict["dU"])
    
    # Python: Uindex.astype(int) is replaced by broadcasting Int.()
    Uindex = Int.(Uindex) 

    # 4. Clamping Indices (Replacing np.maximum, np.minimum, and array size calculation)

    # Maximum valid index in Julia is length(array) (replaces Python's np.size(arr) - 1) [8].
    Umax_index = length(LIR_LFIR_ratio_dict["Umean"]) 
    
    # Clamp minimum index to 1 (replaces Python's 0)
    Uindex = max.(Uindex, 1) 
    
    # Clamp maximum index
    Uindex = min.(Uindex, Umax_index)
    
    # 5. Luminosity Calculation (Replacing array indexing and multiplication)
    
    # Python slices selectSB (the tuple containing the index array) are not needed in Julia.
    # We use the index vectors selSB and selMS to access and update elements simultaneously.
    
    # Starburst (SB) calculation: LFIR[SB indices] = LIR[SB indices] * ratio[Uindex[SB indices]]
    @views LFIR[selSB] = LIR[selSB] .* LIR_LFIR_ratio_dict["LFIR_LIR_ratio_SB"][Uindex[selSB]]
    
    # Main Sequence (MS) calculation: LFIR[MS indices] = LIR[MS indices] * ratio[Uindex[MS indices]]
    @views LFIR[selMS] = LIR[selMS] .* LIR_LFIR_ratio_dict["LFIR_LIR_ratio_MS"][Uindex[selMS]]

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


# function add_fluxes(cat::DataFrame, params::Dict, new_lambda::AbstractVector)
#
#     tstart = time()
#
#     SED_dict = load_sed_pickle_equivalent(params["SED_file"])
#
#     println("Add new monochromatic fluxes...")
#
#     # Calculate Snu_arr. Column access uses dot syntax or bracket indexing (cat.redshift) 
#     # and element-wise multiplication requires the dot operator (.*) [3, 4].
#     Snu_arr = gen_Snu_arr(
#         new_lambda, 
#         SED_dict, 
#         cat.redshift, 
#         cat.mu .* cat.LIR, 
#         cat.Umean, 
#         cat.Dlum, 
#         cat.issb
#     )
#
#     # Since the original Python uses `cat = cat.assign(...)` which returns a new DataFrame, 
#     # we copy the input to maintain non-mutating semantics [5].
#     new_cat = copy(cat) 
#
#     # Iterate over the indices of new_lambda. Julia uses 1-based indexing [6, 7].
#     for i in eachindex(new_lambda) 
#         # Dynamically generate column name as a Symbol (idiomatic for DataFrame column names) [8].
#         col_name = Symbol("S$(new_lambda[i])") 
#
#         # Assign the calculated flux array (Snu_arr column i) to the new DataFrame.
#         # The `df[!, :column] = data` syntax is used for efficient column assignment in DataFrames.jl [9].
#         new_cat[!, col_name] = Snu_arr[:, i]
#     end
#
#     tstop = time()
#
#     # Use string interpolation for printing variables within strings [10].
#     println("New fluxes of $(length(new_cat)) galaxies generated in $(tstop - tstart)s")
#
#     return new_cat
# end
#

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

    return (
        UmeanSB=_p("UmeanSB"),
        UmeanMSz0=_p("UmeanMSz0"),
        alphaMS=_p("alphaMS"),
        zlimMS=_p("zlimMS"),
        sigma_logUmean=_p("sigma_logUmean"),
        SFR2LIR=_p("SFR2LIR"),
        lambda_list=params["lambda_list"],
        sed_tables=sed_tables,
        LIR_LFIR_ratio_dict=load_sed_pickle_equivalent(params["ratios_file"]),
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
        LIR_LFIR_ratio_dict,
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

    # --- 2. Compute Luminosity Distance ---
    println("Compute luminosity distances since they have not been computed before...") 
    # Use vector comprehension for cosmological calculation
    Dlum_values = [Cosmology.luminosity_dist(cosmo_model, z) for z in cat.redshift]  
    cat[!, :Dlum] = Dlum_values

    Ngal = nrow(cat) # Equivalent to len(cat)

    # --- 3. Draw <U> parameters (Vectorized Operations) ---
    println("Draw <U> parameters...")

    Umean = zeros(Ngal) # Equivalent to np.zeros

    # Find indices for MS or high-z SB: uses Julia's 1-based indexing and broadcasting operators
    index_MS_SBhighz = findall((.!(cat.issb)) .| (cat.redshift .>= zlimSB))

    # Calculate Umean for MS and high-z SB (requires broadcasting dot `.` for vectorized operations)
    Umean[index_MS_SBhighz] = 10.0 .^ (
        log10(UmeanMSz0) .+ alphaMS .* min.(cat.redshift[index_MS_SBhighz], zlimMS)
    )

    # Find indices for low-z SB
    index_SBlowz = findall(cat.issb .& (cat.redshift .< zlimSB))

    # Assign Umean value for low-z SB
    Umean[index_SBlowz] .= UmeanSB # Uses broadcast assignment

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
        LIR_LFIR_ratio_dict, 
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

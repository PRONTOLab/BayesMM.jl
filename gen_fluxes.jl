using DataFrames
using Cosmology 
using Random   
using Unitful, UnitfulAstro 
using PhysicalConstants 

# --- Setup Assumptions ---
# Assuming these are defined globally or within the scope of gen_fluxes.
const cosmo_model = Cosmology.cosmology() 
#const cosmo_model = Cosmology.FlatLCDM(h=0.677, OmegaM=0.307, OmegaK=0.0) 

function load_pickle(file_path)
    # Placeholder for reading pickle files, normally replaced by ASDF2.jl or specific I/O
    println("WARNING: Using placeholder load_pickle for $file_path")
    return Dict("grid" => rand(10, 10)) 
end

function gen_Snu_arr(args...)
    # Placeholder for the array generation function
    Ngal = length(args[8]) 
    N_lambda = length(args[6])
    # Returns a matrix where columns are lambda values (Snu_arr[:, i])
    return rand(Ngal, N_lambda) 
end

function gen_LFIR_vec(args...)
    # Placeholder for the LFIR vector generation function
    return rand(length(args[9])) 
end

"""
Translated function to generate SED properties and fluxes.
Note: DataFrames are modified in-place using cat[!, :col] = values.
"""
function gen_fluxes(cat::DataFrame, params::Dict)
    tstart = time() # Start timer [5]

    println("Generate SED properties and fluxes...")

    # --- 1. Compute zlimSB (Scalar Operations) ---
    zlimSB = (log10(params["UmeanSB"]) - log10(params["UmeanMSz0"])) / params["alphaMS"]
    zlimMS = params["zlimMS"]

    if zlimSB > zlimMS
        # Julia string interpolation [2]
        println("zlim SB (when UMS = USB)= $(params["zlimSB"])")
        zlimSB = 9999.0 
    end

    # --- 2. Compute Luminosity Distance ---
    if !("Dlum" in names(cat))
        println("Compute luminosity distances since they have not been computed before...")
        
        redshifts = cat[!, :redshift]
        # Use vector comprehension for cosmological calculation
        Dlum_values = [luminosity_distance(cosmo_model, z) for z in redshifts] 
        
        cat[!, :Dlum] = Dlum_values
    end

    Ngal = nrow(cat) # Equivalent to len(cat)

    # --- 3. Draw <U> parameters (Vectorized Operations) ---
    println("Draw <U> parameters...")

    Umean = zeros(Ngal) # Equivalent to np.zeros

    redshifts = cat[!, :redshift]
    issb = cat[!, :issb]

    # Find indices for MS or high-z SB: uses Julia's 1-based indexing and broadcasting operators
    index_MS_SBhighz = findall((.!(issb)) .| (redshifts .>= zlimSB))

    # Calculate Umean for MS and high-z SB (requires broadcasting dot `.` for vectorized operations)
    Umean[index_MS_SBhighz] = 10.0 .^ (
        log10(params["UmeanMSz0"]) .+ params["alphaMS"] .* min.(redshifts[index_MS_SBhighz], zlimMS)
    )

    # Find indices for low-z SB
    index_SBlowz = findall(issb .& (redshifts .< zlimSB))

    # Assign Umean value for low-z SB
    Umean[index_SBlowz] .= params["UmeanSB"] # Uses broadcast assignment

    # Add log-normal scatter (np.random.normal -> randn)
    scatter_factor = 10.0 .^ (params["sigma_logUmean"] .* randn(Ngal))
    Umean .*= scatter_factor 

    cat[!, :Umean] = Umean

    # --- 4. Load Grids (Serialization) ---
    println("Load SED and LIR grids...")
    SED_dict = load_pickle(params["SED_file"])
    LIR_LFIR_ratio_dict = load_pickle(params["ratios_file"])

    # --- 5. Generate LIR ---
    println("Generate LIR...")
    cat[!, :LIR] = params["SFR2LIR"] .* cat[!, :SFR]

    # --- 6. Generate Flux Array (Assumed helper function call) ---
    Snu_arr = gen_Snu_arr(
        params["lambda_list"], 
        SED_dict, 
        cat[!, :redshift], 
        cat[!, :mu] .* cat[!, :LIR], 
        cat[!, :Umean], 
        cat[!, :Dlum], 
        cat[!, :issb]
    )


    # --- FINAL SECTION TRANSLATION ---
    # Python: for i in range(0,len(params['lambda_list'])):
    lambda_list = params["lambda_list"]
    
    # Iterate using Julia's 1-based indexing [1]
    for i in 1:length(lambda_list) 
        lambda_val = lambda_list[i]

        # Dynamic column naming and assignment
        # Python: kwargs = {'S{:d}'.format(...) : Snu_arr[:,i]}
        col_name_str = "S$(lambda_val)"
        col_name_sym = Symbol(col_name_str) # Convert string name to Symbol for DataFrame column indexing
        
        # Assign the flux vector (Snu_arr is assumed to be Ngal x N_lambda matrix)
        # Note: We use in-place column creation
        cat[!, col_name_sym] = Snu_arr[:, i]
    end

    # generate LFIR (40-400 microns)
    println("Generate LFIR...")
    
    # Assign the new column derived from the vector function
    cat[!, :LFIR] = gen_LFIR_vec(
        LIR_LFIR_ratio_dict, 
        cat[!, :redshift], 
        cat[!, :LIR], 
        cat[!, :Umean], 
        cat[!, :issb]
    )

    tstop = time() # Stop timer

    # Print final timing and results using Julia's println and interpolation
    # Equivalent to Python's len(cat) is nrow(cat)
    println("SED properties of $(nrow(cat)) generated in $(tstop - tstart)s")

    return cat
end
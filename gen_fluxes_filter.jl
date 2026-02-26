using DataFrames
using Dates # For time() if needed, but it's often in Base in modern Julia
using LinearAlgebra # For potential array operations, though dot ops handle most of it
# Assume a utility function for loading the grid data exists, 
# analogous to `load_sed_pickle_equivalent` shown in the context materials [4].
# For demonstration purposes, we assume `load_filter_grid` is defined elsewhere
# and returns a Dict structured like the Python grid.

"""
    gen_fluxes_filter(cat::DataFrame, params::Dict)

Computes the flux of each galaxy in a set of filters using pre-computed grids.
"""
function gen_fluxes_filter(cat::DataFrame, params::Dict)
    tstart = time()

    println("Compute the flux of each galaxy in a set of filters using pre-computed grids...")

    # Data validation: check for necessary columns
    if !("LIR" in names(cat)) || !("Umean" in names(cat))
        println("The catalog must have been processed with gen_fluxes first (it needs LIR and Umean columns).")
        return false
    end

    grid_filter_path = params["grid_filter_path"]
    filter_list = params["filter_list"]

    first_grid = true
    ref_filter_grid_dict = nothing
    
    # Pre-define indices for Z lookups (these will be overwritten in the loop if needed)
    # Initializing them outside the 'if first_grid' block allows using the same 
    # indices if the grid doesn't change, matching the Python optimization [query].
    zidx_below_1based = Int[]
    zidx_above_1based = Int[]
    weight_below = Float64[]
    weight_above = Float64[]
    Uindex = Int[]

    for filter_name in filter_list
        filename = grid_filter_path * filter_name * ".h5" # String concatenation
        println("Load the grid for the ", filter_name, " filter (", filename, ")...")

        # Assume load_filter_grid loads the dictionary structure (replacing Python's pickle)
        filter_grid_dict = load_filter_grid(filename) # Placeholder function

        if first_grid == true
            ref_filter_grid_dict = filter_grid_dict
        end

        # Only check grid equality if this is not the first grid
        if first_grid == false
            same_dU = (filter_grid_dict["dU"] == ref_filter_grid_dict["dU"])
            same_dlog1plusz = (filter_grid_dict["dlog1plusz"] == ref_filter_grid_dict["dlog1plusz"])
            same_Nlog1plusz = (filter_grid_dict["Nlog1plusz"] == ref_filter_grid_dict["Nlog1plusz"])
            same_NUmean = (length(filter_grid_dict["Umean"]) == length(ref_filter_grid_dict["Umean"]))

            # Recompute indices/weights if it's the first grid OR if the current grid differs
            if !(same_dU && same_dlog1plusz && same_Nlog1plusz && same_NUmean)
                # The condition from the end of the first snippet: check if grid is the same
                # (Note: Python used 'not' for negation, which is '!' in Julia [5])
                # We enter the recomputation block below if this condition is met.
            else
                # Skip index/weight computation if grid is the same
                # Jump directly to flux computation below `end # End of index/weight computation block`
                @goto skip_recompute
            end
        end


        # --- Index and Weight Computation (Executed if first_grid or grid changed) ---

        println("Compute the index and weight to perform a fast interpolation... If it is ran several times, it means that the grids are not all the same, which is not optimal because indexes and weights for the interpolation have to be recomputed each time.")

        # --- UMEAN INDEX CALCULATION ---

        # Python: Uindex = np.int_(np.array(np.round(cat['Umean'] / filter_grid_dict['dU'] - 1)))
        # Julia uses broadcasting (`.`), integer casting (`Int.`), and 1-based indexing for clamping.
        Uindex = Int.(round.(cat.Umean ./ filter_grid_dict["dU"] .- 1.0))

        # Solve rare cases with too low Umean (clamping to minimum index: 1) [6]
        lowUmean_indices = findall(Uindex .< 1)
        if !isempty(lowUmean_indices)
            println("WARNING!!!!! ", length(lowUmean_indices), " objects have a Umean below the lowest value of the grid. They were set by default to this lowest value.")
            Uindex[lowUmean_indices] .= 1
        end

        # Solve rare cases with too high Umean (clamping to maximum index: length(Umean)) [6]
        Umean_len = length(filter_grid_dict["Umean"])
        highUmean_indices = findall(Uindex .> Umean_len)
        if !isempty(highUmean_indices)
            println("WARNING!!!!! ", length(highUmean_indices), " objects have a Umean above the highest value of the grid. They were set by default to this highest value.")
            Uindex[highUmean_indices] .= Umean_len # Clamped to maximum valid 1-based index
        end

        # --- REDSHIFT INDEX & WEIGHT CALCULATION ---

        Nlog1plusz = filter_grid_dict["Nlog1plusz"]
        dlog1plusz = filter_grid_dict["dlog1plusz"]

        println("Redshift Grid Key (Nlog1plusz): ", filter_grid_dict["Nlog1plusz"])
        println("Umean Array Length: ", length(filter_grid_dict["Umean"]))

        # 1. Generate grid values
        log1pluszgrid = dlog1plusz .* (1:Nlog1plusz)
        z_grid = 10.0 .^ log1pluszgrid .- 1.0 # Exponentiation uses .^ [7]

        # 2. Calculate float index (0-based equivalent, necessary for calculating integer bounds)
        zindex_float = (log10.(1.0 .+ cat.redshift) ./ dlog1plusz) .- 1.0
        
        # 3. Calculate integer bounds (0-based)
        zindex_below = Int.(floor.(zindex_float))
        zindex_above = Int.(ceil.(zindex_float))
        
        # 4. Convert to 1-based indices for z_grid lookup (needs to happen before boundary checks)
        zidx_below_1based = zindex_below .+ 1
        zidx_above_1based = zindex_above .+ 1

        # Lookup grid values (using these 1-based indices)
        z_above_val = z_grid[zidx_above_1based]
        z_below_val = z_grid[zidx_below_1based]

        # Denominator calculation
        denominator = z_above_val .- z_below_val

        # Initialize weights arrays (required for broadcasting)
        weight_below = (z_above_val .- cat.redshift) ./ denominator
        weight_above = (cat.redshift .- z_below_val) ./ denominator

        # Handle points exactly on the grid (denominator == 0 requires setting weights manually)
        ongrid_indices = findall(zindex_below .== zindex_above)
        if !isempty(ongrid_indices)
            # Julia uses .= for broadcasting assignment [1]
            weight_above[ongrid_indices] .= 0.5 
            weight_below[ongrid_indices] .= 0.5
        end

        # Solve rare cases with too high z (clamping indices to maximum valid index)
        # Check if index exceeds Nlog1plusz (the size of the grid)
        toohighz_indices = findall(zidx_above_1based .> Nlog1plusz)
        if !isempty(toohighz_indices)
            println("WARNING!!!!! A source is at higher z than the highest z of the grid. Use the value of this last element.")
            
            clamped_idx = Nlog1plusz 
            
            zidx_above_1based[toohighz_indices] .= clamped_idx
            zidx_below_1based[toohighz_indices] .= clamped_idx
            
            weight_above[toohighz_indices] .= 0.0
            weight_below[toohighz_indices] .= 1.0
        end

        # Solve rare cases with too low z (clamping to minimum index: 1)
        # Check if the raw integer index (0-based) is below 0, meaning 1-based is below 1
        toolowz_indices = findall(zindex_below .< 0)
        if !isempty(toolowz_indices)
            println("WARNING!!!!! A source is at lower z than the lower z of the grid. Use a rescaling in (zmin_grid/z)^2 for the interpolation weight. It should be an excellent approximation.")
            
            min_idx = 1
            
            zidx_above_1based[toolowz_indices] .= min_idx
            zidx_below_1based[toolowz_indices] .= min_idx
            
            # Calculate scaling factor: (zmin_grid / z)^2
            z_grid_min = z_grid[8]
            redshift_at_low_z = cat.redshift[toolowz_indices]
            scaling_factor = (z_grid_min ./ redshift_at_low_z) .^ 2
            
            weight_above[toolowz_indices] .= scaling_factor
            weight_below[toolowz_indices] .= 0.0
        end

        @label skip_recompute # Jump target if grid structure matches first_grid

        first_grid = false
        N_cat = nrow(cat)
        S_LIR_filter = zeros(Float64, N_cat) # Initialize output vector
        
        # --- COMPUTE MS FLUXES ---
        println("Compute the fluxes for the MS templates...")
        println("Redshift Grid Key (Nlog1plusz): ", filter_grid_dict["Nlog1plusz"])
        println("Umean Array Length: ", length(filter_grid_dict["Umean"]))
        ## Select MS indices (MS = !issb && !qflag)
       MSindex = findall((.!cat.issb) .&& (.!cat.qflag)) 
       
       println("Row range: ", extrema(zidx_below_1based[MSindex]))
       println("Col range: ", extrema(Uindex[MSindex]))
#
       # Interpolation/Lookup for MS (using 1-based Z indices and U index)
       # @views macro allows efficient slicing/assignment without copying large arrays
      #@views S_LIR_filter[MSindex] = (
      #    filter_grid_dict["Snu_LIR_MS"][zidx_below_1based[MSindex], Uindex[MSindex]] .* weight_below[MSindex]
      #    .+ 
      #    filter_grid_dict["Snu_LIR_MS"][zidx_above_1based[MSindex], Uindex[MSindex]] .* weight_above[MSindex]
      #)
       #@views S_LIR_filter[MSindex] = (
       #    filter_grid_dict["Snu_LIR_MS"][Uindex[MSindex],zidx_below_1based[MSindex]] .* weight_below[MSindex]
       #    .+ 
       #    filter_grid_dict["Snu_LIR_MS"][Uindex[MSindex],zidx_above_1based[MSindex]] .* weight_above[MSindex]
       #)
        ## Identify MS galaxies
        MSindex = findall((.!cat.issb) .&& (.!cat.qflag))
#
        # Ensure strict clamping to the correct dimensions
        z_max = filter_grid_dict["Nlog1plusz"] # 1200
        U_max = length(filter_grid_dict["Umean"]) # 3001
#
        println("z_max = ", z_max)
        println("U_max = ", U_max)
        
        # Clamp redshift indices (Rows: 1 to 3001)
        zidx_below_clamped = clamp.(zidx_below_1based[MSindex], 1, z_max)
        zidx_above_clamped = clamp.(zidx_above_1based[MSindex], 1, z_max)
        
        # Clamp U indices (Columns: 1 to 1200)
        Uindex_clamped = clamp.(Uindex[MSindex], 1, U_max)
        
        #println("Matrix Size: ", size(filter_grid_dict["Snu_LIR_MS"]))
        #println("Max Row Index used: ", maximum(zidx_above_1based[MSindex]))
        #println("Max Col Index used: ", maximum(Uindex[MSindex]))
        #println("Min Row Index used: ", minimum(zidx_below_1based[MSindex]))
        #println("Min Col Index used: ", minimum(Uindex[MSindex]))
        
        # Perform element-wise lookup using CartesianIndex
        # Format: Matrix[CartesianIndex.(row_vec, col_vec)]
        @views flux_below = filter_grid_dict["Snu_LIR_MS"][
            CartesianIndex.(Uindex_clamped, zidx_below_clamped)
        ]
        
        @views flux_above = filter_grid_dict["Snu_LIR_MS"][
            CartesianIndex.(Uindex_clamped, zidx_above_clamped)
        ]
        
        # Apply weights
        S_LIR_filter[MSindex] = flux_below .* weight_below[MSindex] .+ flux_above .* weight_above[MSindex]

        
        ## Correct order: CartesianIndex(ROW_INDEX, COL_INDEX)
        ## ROW = Redshift (1-3001), COL = Umean (1-1200)
        #indices_below = CartesianIndex.(zidx_below_1based[MSindex], Uindex[MSindex])
        #indices_above = CartesianIndex.(zidx_above_1based[MSindex], Uindex[MSindex])

        
        ## Perform the lookup
        #@views flux_below = filter_grid_dict["Snu_LIR_MS"][indices_below]
        #@views flux_above = filter_grid_dict["Snu_LIR_MS"][indices_above]
#
        #
        ## Apply weights and assign to the main filter array
        #S_LIR_filter[MSindex] = flux_below .* weight_below[MSindex] .+ flux_above .* weight_above[MSindex]

        # --- COMPUTE SB FLUXES ---
        println("Compute the fluxes for the SB templates...")
        
        # Select SB indices (SB = issb)
        SBindex = findall(cat.issb)
        
        ## Interpolation/Lookup for SB
        #@views S_LIR_filter[SBindex] = (
        #    filter_grid_dict["Snu_LIR_SB"][Uindex[SBindex], zidx_below_1based[SBindex]] .* weight_below[SBindex]
        #    .+ 
        #    filter_grid_dict["Snu_LIR_SB"][Uindex[SBindex], zidx_above_1based[SBindex]] .* weight_above[SBindex]
        #)
        
        sb_matrix = filter_grid_dict["Snu_LIR_SB"]
        phys_rows, phys_cols = size(sb_matrix) # Expected: (3001, 1200)
        
        # 3. Assign limits based on physical shape
        # Per your confirmation: U is the first dimension (3001), z is the second (1200)
        U_limit = phys_rows  # 3001
        z_limit = phys_cols  # 1200
        
        # 4. Clamp indices to physical 1-based bounds 
        Uindex_clamped = clamp.(Uindex[SBindex], 1, U_limit)
        zidx_below_clamped = clamp.(zidx_below_1based[SBindex], 1, z_limit)
        zidx_above_clamped = clamp.(zidx_above_1based[SBindex], 1, z_limit)
        
        # 5. Perform element-wise lookup using CartesianIndex and broadcasting [conversation history]
        # Order: Matrix[CartesianIndex.(Row_Index, Col_Index)]
        @views flux_below = sb_matrix[CartesianIndex.(Uindex_clamped, zidx_below_clamped)]
        @views flux_above = sb_matrix[CartesianIndex.(Uindex_clamped, zidx_above_clamped)]
        
        # 6. Apply weights and assign to the main filter array using broadcasting [4, 5]
        @views S_LIR_filter[SBindex] = (
            flux_below .* weight_below[SBindex] .+ flux_above .* weight_above[SBindex]
        )

        # --- UPDATE CATALOG ---
        
        # Final Flux calculation: S_LIR_filter * cat.LIR * cat.mu (all element-wise operations)
        new_flux = S_LIR_filter .* cat.LIR .* cat.mu
        
        # Dynamic column naming and conversion to Symbol [11]
        col_name = Symbol("S" * filter_name)
        
        # Add new column to the existing DataFrame (mutating operation, common in DataFrames.jl) [11]
        cat[!, col_name] = new_flux
    end

    tstop = time()
    
    # Use string interpolation for printing results [11]
    println("Fluxes through filters of ", nrow(cat), " galaxies generated in ", round(tstop - tstart, digits=3), "s")
    
    return cat
end


function load_filter_grid(filename::String)::Dict{String, Any}
    
    println("Loading data from hdf5 file: $filename")

    # 2. Define the file path
    hdf5_file_path = filename
    
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
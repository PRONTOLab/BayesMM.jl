using DataFrames
using FITSIO
using Unitful
using UnitfulAstro
using Serialization # Equivalent for Python's pickle

function gen_outputs(cat::DataFrame, params::Dict)
    # 1. Handle directory creation
    output_path = params["output_path"]
    if !isdir(output_path)
        println("Create $output_path")
        mkpath(output_path) # Equivalent to os.makedirs
    end

    # 2. Export to Pickle (Serialization)
    # In Julia, serialize is the standard way to dump objects to a binary file
    #if get(params, "gen_pickle", false) == true
    #    file_p = joinpath(output_path, params["run_name"] * ".p")
    #    println("Export the catalog to pickle... ($file_p)")
    #    open(file_p, "w") do f
    #        serialize(f, cat)
    #    end
    #end

    # 3. Export to FITS
    if get(params, "gen_fits", false) == true
        file_fits = joinpath(output_path, params["run_name"] * ".fits")
        println("Export the catalog to FITS... ($file_fits)")

        # Create a copy to add units without mutating the original catalog
        export_cat = copy(cat)
        col_names = names(export_cat)

        # Create a compatible dictionary for FITSIO
        data_dict = Dict{String, AbstractVector}()
        
        for name in col_names
            # 1. Extract the column and strip any physical units 
            col = ustrip.(export_cat[!, name])
            
            # 2. Check for BitVector and convert to standard Vector{Bool}
            # Standard Arrays are recognized by FITSIO's Array{T} method 
            if col isa BitVector
                col = Vector{Bool}(col)
            end
            
            data_dict[name] = col
        end

        # 1. Prepare three vectors to define the header records
        # FITS records consist of a Key, a Value, and a Comment
        h_keys = String[]
        h_vals = Any[]
        h_comms = String[]
        
        # 2. Populate the vectors with your simulation parameters
        for (key, val) in params
            # We use "COMMENT" as the keyword for every parameter
            push!(h_keys, "COMMENT")
            # COMMENT cards have no value in the standard FITS format
            push!(h_vals, nothing) 
            # Store the "Key = Value" string in the comment field of the record
            push!(h_comms, "$key = $val")
        end
        header = FITSHeader(h_keys, h_vals, h_comms)

        # 2. Add your parameters as "COMMENT" cards to the header object
        # FITS supports multiple entries under the "COMMENT" keyword
        #for (key, val) in params
        #    # set_comment! adds or modifies metadata in a header object [1]
        #    # We use "COMMENT" as the key to create standard FITS comment lines
        #    set_comment!(header, "COMMENT", "$key = $val")
        #end
        
        # 3. Write to FITS using the function-block syntax to ensure the file closes
        FITS(file_fits, "w") do f
            # This will now succeed because all types are standard arrays
            write(f, data_dict;header=header)
            
            ## 4. Add simulation parameters as comments
            #for (key, val) in params
            #    FITSIO.write_key(f[1], "$key = $val")
            #end
        end
    end

    return true
end
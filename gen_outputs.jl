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
        mkpath(output_path) # Equivalent to os.makedirs [source: standard library]
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

        # Create a copy to add units without mutating the original catalog [5]
        export_cat = copy(cat)
        col_names = names(export_cat)

        # Create a compatible dictionary for FITSIO
        data_dict = Dict{String, AbstractVector}()
        
        for name in col_names
            # 1. Extract the column and strip any physical units [conversation history]
            col = ustrip.(export_cat[!, name])
            
            # 2. Check for BitVector and convert to standard Vector{Bool}
            # Standard Arrays are recognized by FITSIO's Array{T} method [stacktrace]
            if col isa BitVector
                col = Vector{Bool}(col)
            end
            
            data_dict[name] = col
        end
        
        # 3. Write to FITS using the function-block syntax to ensure the file closes
        FITS(file_fits, "w") do f
            # This will now succeed because all types are standard arrays
            write(f, data_dict)
            
            # 4. Add simulation parameters as comments [1, 2]
            for (key, val) in params
                write_comment(f, "$key = $val")
            end
        end
    end

    return true
end
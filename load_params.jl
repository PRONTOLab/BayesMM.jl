using Base.Meta  # Needed for Meta.parse (optional, as Base.Meta exports Meta.parse)
# Note: Since numpy and IPython.embed are not used in the parsing logic, 
# they are omitted, but numpy functionality is generally handled by Julia's built-in arrays [5, 6].

"""
    load_params(path::AbstractString, force_pysides_path::AbstractString="")

Reads a configuration file, ignoring comments starting with '#', and parses 
the values dynamically using Meta.parse and Base.eval, similar to Python's eval.
Returns a Dict{String, Any}.
"""
function load_params(path::AbstractString, force_pysides_path::AbstractString = "")
    # Initialize a dictionary to hold heterogeneous values
    params = Dict{String, Any}() 

    # 1. Read and Parse Key-Value Pairs
    # Use the 'do' block syntax for safe file handling (ensures 'file' is closed)
    open(path, "r") do file
        # Iterate over lines, similar to Python's 'for line in file:'
        for line in eachline(file) 
            
            line_stripped = strip(line) # Equivalent to line.strip()

            # Skip lines starting with '#' (comments)
            if !startswith(line_stripped, "#") 
                
                # Split line by the first occurrence of '#', keeping only the key/value part
                no_comment = split(line_stripped, '#', limit=2)[1] 
                
                # Split key and value by the first occurrence of '='
                key_value = split(no_comment, '=', limit=2)
                
                if length(key_value) == 2
                    key = strip(key_value[1])
                    value_str = strip(key_value[2])
                    
                    # Store raw string value initially
                    params[key] = value_str
                end
            end
        end
    end

    # 2. Evaluate and Convert Types (Equivalent to Python's eval loop)
    # This dynamic evaluation is necessary to convert strings like "3.14" 
    # or "[1-3]" into their corresponding Julia types.
    for (key, value_str) in params
        try
            # Convert the string into a Julia expression object
            ex = Meta.parse(value_str) 
            
            # Execute the expression within the main module scope (Main)
            # This is the dynamic execution equivalent of Python's eval()
            params[key] = Base.eval(Main, ex)
            
        catch e
            # If the value cannot be parsed and evaluated (e.g., non-valid Julia syntax), 
            # keep it as the raw string and issue a warning.
            @warn "Could not evaluate parameter '$key'. Keeping value as raw string: $value_str" exception=(e, catch_backtrace())
        end
    end
    
    return params
end

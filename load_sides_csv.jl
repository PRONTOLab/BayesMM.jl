using DataFrames # Equivalent to pandas [1, 3, 4]
using CSV        # Used for reading CSV files efficiently [7, 8]
# Note: NumPy array functionality is built into Julia [9, 10].
# IPython.embed is replaced by interactive REPL use or the debugger.

"""
    load_sides_csv(catfile::AbstractString, nrows::Union{Nothing, Int} = nothing)

Loads a catalog CSV, selecting the primary columns required for analysis.
nrows: Limits the number of rows read from the file (excluding headers).
"""
function load_sides_csv(catfile::AbstractString, nrows::Union{Nothing, Int} = nothing)

    # Equivalent to Python's print()
    println("Load the catalog CSV generated from the original IDL code to get RA, Dec, z, Mhalo, and Mstar...") # [9]

    # Equivalent to pd.read_csv, reading directly into a DataFrame [8].
    # The 'limit' keyword argument handles the Python 'nrows' parameter [11].
    # We explicitly specify the delimiter as ',' using 'delim' (equivalent to 'sep=', assuming a standard CSV).
    cat_IDL = CSV.read(
        catfile, 
        DataFrame; 
        delim=',', 
        limit=nrows # Reads only up to `nrows`
    )

    # --- Column Selection and Renaming (Equivalent to pd.DataFrame(..., columns=...)) ---
    
    # Define the list of required columns (as Symbols in Julia, which represent column names)
    required_cols = [:redshift, :ra, :dec, :Mhalo, :Mstar]
    
    # Select only the required columns and return a new DataFrame [2, 12].
    # This assumes the CSV file already contains columns with these exact names.
    cat = cat_IDL[!, required_cols] 
    
    return cat
end
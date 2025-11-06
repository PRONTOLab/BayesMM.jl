# BayesMMfwd.jl
Forward model for mm-sources


gen_pysides_from_original.jl -> main script to run

The following include commands load up the helper functions.
include("load_params.jl")
include("load_sides_csv.jl")
include("gen_sfr_props.jl")
include("gen_magnification.jl")
include("gen_fluxes.jl")

params = load_params("SIDES_from_original.par") command loads the parameters

cat = load_sides_csv(csv_idl_path) loads up the big table which has basic halo and redshift information.

The commands following these modify and edit the catalog '[cat]' and add columns.


include("load_params.jl")
include("load_sides_csv.jl")
include("gen_sfr_props.jl")
include("gen_magnification.jl")
include("gen_fluxes.jl")
include("gen_fluxes_filter.jl")
include("gen_lines.jl")
include("gen_outputs.jl")


csv_idl_path = "/Volumes/T7 Shield/SIDES/test/SIDES_Bethermin2017_short.csv"

params = load_params("SIDES_from_original.par")

cat = load_sides_csv(csv_idl_path)

cat = gen_sfr_props(cat, params)

cat = gen_magnification(cat, params)

cat = gen_fluxes(cat, params)

cat = gen_fluxes_filter(cat, params)

cat = gen_CO(cat, params)

cat = gen_CII(cat, params)

cat = gen_CI(cat, params)

gen_outputs(cat, params)

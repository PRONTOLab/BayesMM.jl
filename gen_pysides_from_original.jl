include("load_params.jl")
include("load_sides_csv.jl")
include("gen_sfr_props.jl")
include("gen_magnification.jl")
include("gen_fluxes.jl")


csv_idl_path = "data/SIDES_Bethermin2017_short.csv"

params = load_params("SIDES_from_original.par")

cat = load_sides_csv(csv_idl_path)

function process(cat, params)
	cat = gen_sfr_props(cat, params)

	# cat = gen_magnification(cat, params)

	# cat = gen_fluxes(cat, params)
	return cat
end

using Reactant
Reactant.set_default_backend("cpu")
params = Reactant.to_rarray(params)
cat_nt = Tables.columntable(cat)
cat2 = @jit process(cat_nt, params)


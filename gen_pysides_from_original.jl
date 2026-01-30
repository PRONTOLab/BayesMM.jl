ENV["JULIA_DEBUG"] = "Reactant,Reactant_jll"
using Reactant
Reactant.set_default_backend("cpu")
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true

include("load_params.jl")
include("load_sides_csv.jl")
include("gen_sfr_props.jl")
include("gen_magnification.jl")
include("gen_fluxes.jl")


csv_idl_path = "data/SIDES_Bethermin2017_short.csv"

params = load_params("SIDES_from_original.par")

cat = load_sides_csv(csv_idl_path)

sfr_params = parse_sfr_params(params)

function process(cat)
    return gen_sfr_props_traced(cat, sfr_params)

	# cat = gen_sfr_props(cat, params)

	# cat = gen_magnification(cat, params)

	# cat = gen_fluxes(cat, params)
end

cat_ra = (
    redshift = Reactant.ConcreteRArray(cat.redshift),
    ra = Reactant.ConcreteRArray(cat.ra),
    dec = Reactant.ConcreteRArray(cat.dec),
    Mhalo = Reactant.ConcreteRArray(cat.Mhalo),
    Mstar = Reactant.ConcreteRArray(cat.Mstar),
)
(qflag, SFR, issb) = @jit process(cat_ra)

println("\n=== Results ===")
println("qflag: ", typeof(qflag), " size=", size(qflag))
println("  quenched count: ", sum(Array(qflag)))
println("SFR: ", typeof(SFR), " size=", size(SFR))
println("  min=", minimum(Array(SFR)), " max=", maximum(Array(SFR)), " mean=", sum(Array(SFR))/length(SFR))
println("issb: ", typeof(issb), " size=", size(issb))
println("  starburst count: ", sum(Array(issb)))


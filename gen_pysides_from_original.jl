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

function gen_sfr_props2(cat, p)
    (; Chab2Salp_num, Mt0, alpha1, alpha2, sigma0, beta1, beta2, qfrac0, gamma,
        m1, a2, m0, a0, a1, corr_zmean_lowzcorr, zmax_lowzcorr, zmean_lowzcorr,
        Psb_hz, slope_Psb, z_Psb_knee, sigma_MS, logx0, logBsb, SFR_max) = p

    Ngal = length(cat.redshift)

    Mtz = Mt0 .+ alpha1 .* cat.redshift .+ alpha2 .* cat.redshift .^ 2
    sigmaz = sigma0 .+ beta1 .* cat.redshift .+ beta2 .* cat.redshift .^ 2
    qfrac0z = qfrac0 .* (1.0 .+ cat.redshift) .^ gamma
    Prob_SF = (1.0 .- qfrac0z) .* 0.5 .* (1.0 .- erf.((log10.(cat.Mstar) .- Mtz) ./ sigmaz))
    Xuni = rand(Ngal)
    qflag = Xuni .> Prob_SF

    m_all = log10.(cat.Mstar .* Chab2Salp_num ./ 1.0e9)
    r_all = log10.(1.0 .+ cat.redshift)
    expr_all = max.(m_all .- m1 .- a2 .* r_all, 0.0)
    logSFRms_all = m_all .- m0 .+ a0 .* r_all .- a1 .* expr_all .^ 2 .- log10(Chab2Salp_num)
    logSFRms_all = logSFRms_all .+ corr_zmean_lowzcorr .* (zmax_lowzcorr .- min.(cat.redshift, zmax_lowzcorr)) ./ (zmax_lowzcorr - zmean_lowzcorr)

    Psb_all = Psb_hz .+ slope_Psb .* (z_Psb_knee .- min.(cat.redshift, z_Psb_knee))
    Xuni_sb = rand(Ngal)
    issb_all = Xuni_sb .< Psb_all

    noise_all = randn(Ngal)
    issb_term_all = issb_all .* (logBsb - logx0)
    SFR_all = 10.0 .^ (logSFRms_all .+ sigma_MS .* noise_all .+ logx0 .+ issb_term_all)

    SFR_all = min.(SFR_all, SFR_max)

    mask_SF = .!qflag
    SFR_final = ifelse.(mask_SF, SFR_all, zero(eltype(SFR_all)))
    issb_final = ifelse.(mask_SF, issb_all, false)

    return (qflag, SFR_final, issb_final)
end

function process(cat)
    return gen_sfr_props2(cat, sfr_params)

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

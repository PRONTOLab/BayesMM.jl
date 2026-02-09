ENV["JULIA_DEBUG"] = "Reactant,Reactant_jll"
using Reactant
using DataFrames
using Random
using CSV
Reactant.set_default_backend("cpu")
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = false
const BENCHMARK_SEED = 1234
Random.seed!(BENCHMARK_SEED)

include("load_params.jl")
include("load_sides_csv.jl")
include("gen_sfr_props.jl")
include("gen_magnification.jl")
include("gen_fluxes.jl")

## Initialize stuff
csv_idl_path = "data/SIDES_Bethermin2017_short2.csv" #use truncated csv for
params = load_params("SIDES_from_original.par")
cat_og = load_sides_csv(csv_idl_path)
sfr_params = parse_sfr_params(params)

function gen_sfr_props2(cat::DataFrame, p::NamedTuple)
    # Read parameters
    (; Chab2Salp_num, Mt0, alpha1, alpha2, sigma0, beta1, beta2, qfrac0, gamma,
        m1, a2, m0, a0, a1, corr_zmean_lowzcorr, zmax_lowzcorr, zmean_lowzcorr,
        Psb_hz, slope_Psb, z_Psb_knee, sigma_MS, logx0, logBsb, SFR_max) = p

    println("Generate the star-formation properties...")
    println("Draw quenched galaxies...")
    Ngal = length(cat.redshift)

    # Draw quenched galaxies using parsed parameters
    Mtz = Mt0 .+ alpha1 .* cat.redshift .+ alpha2 .* cat.redshift .^ 2
    sigmaz = sigma0 .+ beta1 .* cat.redshift .+ beta2 .* cat.redshift .^ 2
    qfrac0z = qfrac0 .* (1.0 .+ cat.redshift) .^ gamma
    Prob_SF = (1.0 .- qfrac0z) .* 0.5 .* (1.0 .- erf.((log10.(cat.Mstar) .- Mtz) ./ sigmaz))
    Xuni = rand(Ngal)
    qflag = Xuni .> Prob_SF

    m_all = log10.(cat.Mstar .* Chab2Salp_num ./ 1.0e9)
    r_all = log10.(1.0 .+ cat.redshift)
    expr_all = max.(m_all .- m1 .- a2 .* r_all, 0.0)
    # Use parsed parameters here
    logSFRms_all = m_all .- m0 .+ a0 .* r_all .- a1 .* expr_all .^ 2 .- log10(Chab2Salp_num)
    # Crucial Fix applies here, ensuring zmax/zmean are numbers
    logSFRms_all = logSFRms_all .+ corr_zmean_lowzcorr .* (zmax_lowzcorr .- min.(cat.redshift, zmax_lowzcorr)) ./ (zmax_lowzcorr - zmean_lowzcorr)

    # Use parsed parameters here
    Psb_all = Psb_hz .+ slope_Psb .* (z_Psb_knee .- min.(cat.redshift, z_Psb_knee))
    Xuni_sb = rand(Ngal)
    issb_all = Xuni_sb .< Psb_all

    noise_all = randn(Ngal)

    # Use parsed parameters here
    issb_term_all = issb_all .* (logBsb - logx0)
    SFR_all = 10.0 .^ (logSFRms_all .+ sigma_MS .* noise_all .+ logx0 .+ issb_term_all)
    mask_SF = .!qflag
    
    println("Deal with SFR drawn initially above the SFR limit...")
    too_high_mask = (SFR_all .> SFR_max) .& mask_SF
    
    # Redraw only the initially invalid SF entries, until they are valid.
    @trace track_numbers=false for i in 1:Ngal
        needs_redraw_i = @allowscalar too_high_mask[i]
        @trace track_numbers=false while needs_redraw_i
            logSFRms_i = @allowscalar logSFRms_all[i]
            issb_i = @allowscalar issb_all[i]
            redraw_noise_i = randn()
            issb_term_i = issb_i * (logBsb - logx0)
            redraw_i = 10.0 ^ (logSFRms_i + sigma_MS * redraw_noise_i + logx0 + issb_term_i)
            @allowscalar setindex!(SFR_all, redraw_i, i)
            needs_redraw_i = redraw_i > SFR_max
        end
    end
    SFR_final = ifelse.(mask_SF, SFR_all, zero(eltype(SFR_all)))
    issb_final = ifelse.(mask_SF, issb_all, false)
    println("Typeof SFR_final is: ", typeof(SFR_final))
    println("Typeof issb_final is: ",typeof(issb_final))
    cat[!,:SFR] = SFR_final
    cat[!,:issb] = issb_final
    return cat
    # return (qflag, SFR_final, issb_final)
end

function process!(cat,sfr_params, params)
    cat = gen_sfr_props2(cat, sfr_params)
	# cat = gen_magnification(cat, params)
    return cat
	# cat = gen_fluxes(cat, params)
end

cat_ra = Reactant.to_rarray(cat_og)
cat_f = @jit process!(cat_ra, sfr_params, params)
println("\n=== Finished Computations ===")

# println("qflag: ", typeof(qflag), " size=", size(qflag))
# println("  quenched count: ", sum(ifelse.(qflag, 1, 0)))
# println("SFR: ", typeof(SFR), " size=", size(SFR))
# println("  min=", minimum(SFR), " max=", maximum(SFR), " mean=", sum(SFR) / length(SFR))
# println("issb: ", typeof(issb), " size=", size(issb))
# println("  starburst count: ", sum(ifelse.(issb, 1, 0)))

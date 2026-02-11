ENV["JULIA_DEBUG"] = "Reactant,Reactant_jll"
using Reactant
using DataFrames
using Random
using Cosmology
using Unitful
const cosmo_model = Cosmology.cosmology() 
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

function process!(cat,
                  sfr_params,
                  mag_params,
                  flux_params)
    cat = gen_sfr_props2(cat, sfr_params)
	cat = gen_magnification(cat, mag_params)
    cat = gen_fluxes(cat, flux_params)
    return cat
end

function preallocate_output_columns!(cat::DataFrame, flux_params)
    Ngal = nrow(cat)
    cat[!, :SFR] = zeros(Float64, Ngal)
    cat[!, :issb] = falses(Ngal)
    cat[!, :mu] = ones(Float64, Ngal)
    cat[!, :Dlum] = [ustrip(u"m", Cosmology.luminosity_dist(cosmo_model, z)) for z in cat.redshift]
    cat[!, :Umean] = zeros(Float64, Ngal)
    cat[!, :LIR] = zeros(Float64, Ngal)
    cat[!, :LFIR] = zeros(Float64, Ngal)
    for v in flux_params.lambda_list
        colname = "S$(v)"
        colsym = Symbol(colname)
        cat[!, colsym] = zeros(Float64, Ngal)
    end
    return cat
end

function build_runtime_inputs(csv_idl_path::String, param_path::String)
    params = load_params(param_path)
    cat_template = load_sides_csv(csv_idl_path)
    sfr_params = parse_sfr_params(params)
    mag_params = process_magnification_grid(params["path_mu_file"])
    flux_params = parse_flux_params(cosmo_model, params)
    preallocate_output_columns!(cat_template, flux_params)
    return (
        cat_template = cat_template,
        sfr_params = sfr_params,
        mag_params = mag_params,
        flux_params = flux_params
    )
end

function to_reactant_inputs(mag_params, flux_params)
    return (
        mag_r_params = Reactant.to_rarray(mag_params),
        flux_r_params = Reactant.to_rarray(flux_params)
    )
end

function benchmark_process!(runtime_inputs)
    reactant_inputs = to_reactant_inputs(runtime_inputs.mag_params, runtime_inputs.flux_params)

    println("\n=== Benchmark process! ===")

    cat_nonjit = copy(runtime_inputs.cat_template)
    Random.seed!(BENCHMARK_SEED)
    process!(
        cat_nonjit,
        runtime_inputs.sfr_params,
        runtime_inputs.mag_params,
        runtime_inputs.flux_params
    )

    t_nonjit = @elapsed process!(
        cat_nonjit,
        runtime_inputs.sfr_params,
        runtime_inputs.mag_params,
        runtime_inputs.flux_params
    )

    cat_jit_cold = Reactant.to_rarray(copy(runtime_inputs.cat_template))
    # Random.seed!(BENCHMARK_SEED)
    # t_jit_cold = @elapsed @jit process!(
    #     cat_jit_cold,
    #     runtime_inputs.sfr_params,
    #     reactant_inputs.mag_r_params,
    #     reactant_inputs.flux_r_params
    # )
    
    cat_jit_warm = Reactant.to_rarray(copy(runtime_inputs.cat_template))
    v = @compile sync=true process!(
        cat_jit_warm,
        runtime_inputs.sfr_params,
        reactant_inputs.mag_r_params,
        reactant_inputs.flux_r_params
    )

    Random.seed!(BENCHMARK_SEED)
    t_jit_cold = @elapsed v(
        cat_jit_warm,
        runtime_inputs.sfr_params,
        reactant_inputs.mag_r_params,
        reactant_inputs.flux_r_params)

    
    t_jit_warm = @elapsed v(
        cat_jit_warm,
        runtime_inputs.sfr_params,
        reactant_inputs.mag_r_params,
        reactant_inputs.flux_r_params)

    println("process! (default-Julia): ", t_nonjit, "s")
    println("process! (@jit cold): ", t_jit_cold, "s")
    println("process! (@jit warm): ", t_jit_warm, "s")

    return (
        nonjit = cat_nonjit,
        jit_cold = cat_jit_cold,
        jit_warm = cat_jit_warm,
        timings = (t_nonjit = t_nonjit, t_jit_cold = t_jit_cold, t_jit_warm = t_jit_warm)
    )
end

csv_idl_path = "data/SIDES_Bethermin2017_short2.csv"
runtime_inputs = build_runtime_inputs(csv_idl_path, "SIDES_from_original.par")
benchmark_results = benchmark_process!(runtime_inputs)
println("\n=== Finished Computations ===")

# println("qflag: ", typeof(qflag), " size=", size(qflag))
# println("  quenched count: ", sum(ifelse.(qflag, 1, 0)))
# println("SFR: ", typeof(SFR), " size=", size(SFR))
# println("  min=", minimum(SFR), " max=", maximum(SFR), " mean=", sum(SFR) / length(SFR))
# println("issb: ", typeof(issb), " size=", size(issb))
# println("  starburst count: ", sum(ifelse.(issb, 1, 0)))

ENV["JULIA_DEBUG"] = "Reactant,Reactant_jll"
using Reactant
using DataFrames
using Random
using Cosmology
using Unitful
using Dates
using Printf
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

function format_benchmark_report(dataset_path::String, timings::NamedTuple)
    t_julia_cold = timings.t_julia_cold
    t_julia_warm = timings.t_julia_warm
    t_reactant_cold = timings.t_reactant_cold
    t_reactant_warm = timings.t_reactant_warm
    io = IOBuffer()
    println(io, "BayesMMfwd Benchmark Results")
    println(io, "Generated: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println(io, "Dataset: ", dataset_path)
    println(io, "Backend: cpu")
    println(io, "Seed: ", BENCHMARK_SEED)
    println(io, "")
    println(io, rpad("Metric", 36), "Time (s)")
    println(io, repeat("-", 52))
    println(io, rpad("process! (Julia cold JIT)", 36), @sprintf("%.6f", t_julia_cold))
    println(io, rpad("process! (Julia warm JIT)", 36), @sprintf("%.6f", t_julia_warm))
    println(io, rpad("process! (Reactant cold run)", 36), @sprintf("%.6f", t_reactant_cold))
    println(io, rpad("process! (Reactant warm run)", 36), @sprintf("%.6f", t_reactant_warm))
    println(io, "")
    println(io, "Derived ratios")
    println(io, repeat("-", 52))
    println(io, "Julia cold/warm: ", @sprintf("%.3fx", t_julia_cold / t_julia_warm))
    println(io, "Reactant cold/warm: ", @sprintf("%.3fx", t_reactant_cold / t_reactant_warm))
    println(io, "Warm Reactant/Julia: ", @sprintf("%.3fx", t_reactant_warm / t_julia_warm))
    return String(take!(io))
end

function benchmark_process!(runtime_inputs, dataset_path::String; results_path::String = "benchmark_results_small.txt")
    reactant_inputs = to_reactant_inputs(runtime_inputs.mag_params, runtime_inputs.flux_params)

    println("\n=== Benchmark process! ===")

    cat_julia = copy(runtime_inputs.cat_template)
    Random.seed!(BENCHMARK_SEED)
    t_julia_cold = @elapsed process!(
        cat_julia,
        runtime_inputs.sfr_params,
        runtime_inputs.mag_params,
        runtime_inputs.flux_params
    )

    t_julia_warm = @elapsed process!(
        cat_julia,
        runtime_inputs.sfr_params,
        runtime_inputs.mag_params,
        runtime_inputs.flux_params
    )

    cat_reactant = Reactant.to_rarray(copy(runtime_inputs.cat_template))
    compiled_process = @compile sync=true process!(
        cat_reactant,
        runtime_inputs.sfr_params,
        reactant_inputs.mag_r_params,
        reactant_inputs.flux_r_params
    )

    Random.seed!(BENCHMARK_SEED)
    t_reactant_cold = @elapsed compiled_process(
        cat_reactant,
        runtime_inputs.sfr_params,
        reactant_inputs.mag_r_params,
        reactant_inputs.flux_r_params)

    
    t_reactant_warm = @elapsed compiled_process(
        cat_reactant,
        runtime_inputs.sfr_params,
        reactant_inputs.mag_r_params,
        reactant_inputs.flux_r_params)

    timings = (
        t_julia_cold = t_julia_cold,
        t_julia_warm = t_julia_warm,
        t_reactant_cold = t_reactant_cold,
        t_reactant_warm = t_reactant_warm
    )

    println("process! (Julia cold JIT): ", t_julia_cold, "s")
    println("process! (Julia warm JIT): ", t_julia_warm, "s")
    println("process! (Reactant cold run): ", t_reactant_cold, "s")
    println("process! (Reactant warm run): ", t_reactant_warm, "s")
    println("writing formatted benchmark report to ", results_path)

    report = format_benchmark_report(dataset_path, timings)
    write(results_path, report)

    return (
        julia = cat_julia,
        reactant = cat_reactant,
        timings = timings,
        report_path = results_path
    )
end

csv_idl_path = "data/SIDES_Bethermin2017_short.csv"
runtime_inputs = build_runtime_inputs(csv_idl_path, "SIDES_from_original.par")
benchmark_results = benchmark_process!(
    runtime_inputs,
    csv_idl_path;
    results_path = "benchmark_results_large.txt"
)
println("\n=== Finished Computations ===")

# println("qflag: ", typeof(qflag), " size=", size(qflag))
# println("  quenched count: ", sum(ifelse.(qflag, 1, 0)))
# println("SFR: ", typeof(SFR), " size=", size(SFR))
# println("  min=", minimum(SFR), " max=", maximum(SFR), " mean=", sum(SFR) / length(SFR))
# println("issb: ", typeof(issb), " size=", size(issb))
# println("  starburst count: ", sum(ifelse.(issb, 1, 0)))

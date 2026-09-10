import SkyMatch
import Reactant
using Printf: @sprintf

function configure_backend(backend)
    group = uppercase(String(backend))
    group == "GPU" && (group = "CUDA")
    group in ("CPU", "CUDA", "TPU") || throw(ArgumentError(
        "backend must be CPU, CUDA, or TPU; got $backend",
    ))
    Reactant.set_default_backend(group == "CUDA" ? "gpu" : lowercase(group))
    return group
end

function materialize_output(value::NamedTuple)
    names = keys(value)
    values = map(materialize_output, Tuple(value))
    return NamedTuple{names}(values)
end

materialize_output(value::Tuple) = map(materialize_output, value)
materialize_output(value::AbstractArray) = Array(value)
materialize_output(value) = value

seconds_since(start_ns) = (time_ns() - start_ns) / 1.0e9

function run_reactant_forward_model(inputs, params, noise)
    total_start = time_ns()

    transfer_start = time_ns()
    inputs_reactant = Reactant.to_rarray(inputs)
    params_reactant = Reactant.to_rarray(params)
    noise_reactant = Reactant.to_rarray(noise)
    host_to_device = seconds_since(transfer_start)

    compile_start = time_ns()
    compiled_forward = Reactant.@compile sync=true SkyMatch.forward_model(
        inputs_reactant,
        params_reactant,
        noise_reactant,
    )
    compilation = seconds_since(compile_start)

    execution_start = time_ns()
    reactant_output = compiled_forward(
        inputs_reactant,
        params_reactant,
        noise_reactant,
    )
    Reactant.synchronize(reactant_output)
    execution = seconds_since(execution_start)

    materialization_start = time_ns()
    host_output = materialize_output(reactant_output)
    device_to_host = seconds_since(materialization_start)
    total = seconds_since(total_start)

    timings = (; host_to_device, compilation, execution, device_to_host, total)
    return (; output=host_output, timings)
end

function print_usage(io=stdout)
    println(
        io,
        """
Usage:
  julia --project=. gen_pysides_from_original.jl [options]

Options:
  --dataset PATH     SIDES CSV input (default: data/SIDES_Bethermin2017_short2.csv)
  --rows N           Read only the first N rows
  --large            Use the large CSV (all rows unless followed by --rows)
  --filters          Include filter-grid fluxes (requires every FILTER_NAME.h5)
  --backend NAME     Reactant backend: CPU, CUDA, or TPU (default: CPU)
  --write-output     Write the final FITS catalog
  --help             Show this message

This is the Reactant-only production driver. For Julia-versus-Reactant
comparisons, use benchmark/benchmark_reactant.jl.
"""
    )
end

function parse_command_line(args)
    options = (
        dataset="data/SIDES_Bethermin2017_short2.csv",
        nrows=nothing,
        filters=false,
        backend="CPU",
        write_output=false,
    )
    values = Dict{Symbol,Any}(pairs(options))

    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--help"
            print_usage()
            return nothing
        elseif argument == "--large"
            values[:dataset] = "data/SIDES_Bethermin2017_short.csv"
            values[:nrows] = nothing
        elseif argument == "--filters"
            values[:filters] = true
        elseif argument == "--write-output"
            values[:write_output] = true
        elseif argument in ("--dataset", "--rows", "--backend")
            index == length(args) &&
                throw(ArgumentError("$argument requires a value"))
            value = args[index+1]
            if argument == "--dataset"
                values[:dataset] = value
            elseif argument == "--rows"
                values[:nrows] = parse(Int, value)
                values[:nrows] > 0 ||
                    throw(ArgumentError("--rows must be positive"))
            else
                values[:backend] = value
            end
            index += 1
        else
            throw(ArgumentError("unknown argument: $argument"))
        end
        index += 1
    end
    return (; values...)
end

function run_reactant_pipeline(;
    dataset="data/SIDES_Bethermin2017_short2.csv",
    param_path="data/SIDES_from_original.par",
    nrows=nothing,
    filters=false,
    backend="CPU",
    write_output=false,
)
    backend_group = configure_backend(backend)
    devices = join(string.(Reactant.devices()), ", ")
    params = SkyMatch.load_par_file(param_path)
    catalog_template = SkyMatch.load_sides_csv(dataset, nrows)
    inputs = SkyMatch.build_forward_inputs(catalog_template)
    parameter_data = SkyMatch.build_forward_parameters(params; filters)
    noise = SkyMatch.make_simulation_noise(length(inputs.redshift))

    n_gal = length(inputs.redshift)
    println(
        "Running the Reactant forward model for $n_gal rows on ",
        "$backend_group ($devices)...",
    )
    reactant_run = run_reactant_forward_model(
        inputs,
        parameter_data.numeric,
        noise,
    )
    timings = reactant_run.timings
    println("Reactant timings:")
    println("  host-to-device: ", @sprintf("%.6f s", timings.host_to_device))
    println("  compilation:    ", @sprintf("%.6f s", timings.compilation))
    println("  execution:      ", @sprintf("%.6f s", timings.execution))
    println("  device-to-host: ", @sprintf("%.6f s", timings.device_to_host))
    println("  total:          ", @sprintf("%.6f s", timings.total))

    output_catalog = SkyMatch.add_output_columns!(
        copy(catalog_template),
        reactant_run.output,
        parameter_data.numeric.dust.lambda_list,
        parameter_data.filter_names,
    )
    if write_output
        SkyMatch.gen_outputs(output_catalog, params)
    end

    return (; output_catalog, reactant_run, params)
end

function main(args=ARGS)
    options = parse_command_line(args)
    isnothing(options) && return nothing
    return run_reactant_pipeline(; options...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

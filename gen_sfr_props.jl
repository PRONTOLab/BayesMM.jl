using DataFrames
using Random
using SpecialFunctions: erf 
using Base: parse # Ensure parse is available

function gen_sfr_props(cat, params)

    tstart = time() 

    # --- FIX: Explicitly parse all string numerical parameters immediately ---
    # Assuming all necessary parameters read from the dictionary might be strings:
    python_expr_str = params["Chab2Salp"]
    #julia_expr_str = replace(python_expr_str, "**" => "^")
    Chab2Salp_num = eval(python_expr_str)
    Mt0 = params["Mt0"] isa AbstractString ? parse(Float64, params["Mt0"]) : params["Mt0"]
    alpha1 = params["alpha1"] isa AbstractString ? parse(Float64, params["alpha1"]) : params["alpha1"]
    alpha2 = params["alpha2"] isa AbstractString ? parse(Float64, params["alpha2"]) : params["alpha2"]
    sigma0 = params["sigma0"] isa AbstractString ? parse(Float64, params["sigma0"]) : params["sigma0"]
    beta1 = params["beta1"] isa AbstractString ? parse(Float64, params["beta1"]) : params["beta1"]
    beta2 = params["beta2"] isa AbstractString ? parse(Float64, params["beta2"]) : params["beta2"]
    qfrac0 = params["qfrac0"] isa AbstractString ? parse(Float64, params["qfrac0"]) : params["qfrac0"]
    gamma = params["gamma"] isa AbstractString ? parse(Float64, params["gamma"]) : params["gamma"]

    m1 = params["m1"] isa AbstractString ? parse(Float64, params["m1"]) : params["m1"]
    a2 = params["a2"] isa AbstractString ? parse(Float64, params["a2"]) : params["a2"]
    m0 = params["m0"] isa AbstractString ? parse(Float64, params["m0"]) : params["m0"]
    a0 = params["a0"] isa AbstractString ? parse(Float64, params["a0"]) : params["a0"]
    a1 = params["a1"] isa AbstractString ? parse(Float64, params["a1"]) : params["a1"]

    corr_zmean_lowzcorr = params["corr_zmean_lowzcorr"] isa AbstractString ? parse(Float64, params["corr_zmean_lowzcorr"]) : params["corr_zmean_lowzcorr"]
    zmax_lowzcorr = params["zmax_lowzcorr"] isa AbstractString ? parse(Float64, params["zmax_lowzcorr"]) : params["zmax_lowzcorr"]
    zmean_lowzcorr = params["zmean_lowzcorr"] isa AbstractString ? parse(Float64, params["zmean_lowzcorr"]) : params["zmean_lowzcorr"]

    Psb_hz = params["Psb_hz"] isa AbstractString ? parse(Float64, params["Psb_hz"]) : params["Psb_hz"]
    slope_Psb = params["slope_Psb"] isa AbstractString ? parse(Float64, params["slope_Psb"]) : params["slope_Psb"]
    z_Psb_knee = params["z_Psb_knee"] isa AbstractString ? parse(Float64, params["z_Psb_knee"]) : params["z_Psb_knee"]
    
    sigma_MS = params["sigma_MS"] isa AbstractString ? parse(Float64, params["sigma_MS"]) : params["sigma_MS"]
    logx0 = params["logx0"] isa AbstractString ? parse(Float64, params["logx0"]) : params["logx0"]
    logBsb = params["logBsb"] isa AbstractString ? parse(Float64, params["logBsb"]) : params["logBsb"]
    
    SFR_max = params["SFR_max"] isa AbstractString ? parse(Float64, params["SFR_max"]) : params["SFR_max"]
    # --- END FIX ---


    println("Generate the star-formation properties...")
    println("Draw quenched galaxies...")

    Ngal = size(cat, 1) 

    # Draw quenched galaxies using parsed parameters
    Mtz = Mt0 .+ alpha1 .* cat.redshift .+ alpha2 .* cat.redshift.^2 
    sigmaz =  sigma0 .+  beta1 .* cat.redshift .+  beta2 .* cat.redshift.^2
    qfrac0z = qfrac0 .* (1.0 .+ cat.redshift) .^ gamma
    
    Prob_SF = (1.0 .- qfrac0z) .* 0.5 .* (1.0 .- erf.((log10.(cat.Mstar) .- Mtz) ./ sigmaz))
    
    Xuni = rand(Ngal) 

    qflag = Xuni .> Prob_SF 
    
    cat[!, :qflag] = qflag

    # Generate SFR for non-quenched objects

    println("Generate SFRs...")

    index_SF = findall(.!qflag)
    mask_SF = .!qflag 
    N_SF = length(index_SF)

    Mstar_SF = cat[mask_SF, :Mstar]
    z_SF = cat[mask_SF, :redshift]
    
    # Use parsed Chab2Salp_num
    m_SF = log10.(Mstar_SF .* Chab2Salp_num ./ 1.0e9)
    z_SF = cat[mask_SF, :redshift]
    
    r = log10.(1.0 .+ z_SF)
    
    expr = max.(m_SF .- m1 .- a2 .* r, 0.0)

    # Use parsed parameters here
    logSFRms_SF = m_SF .- m0 .+ a0 .* r .- a1 .* expr.^2 .- log10(Chab2Salp_num)

    # Crucial Fix applies here, ensuring zmax/zmean are numbers
    logSFRms_SF .+= corr_zmean_lowzcorr .* (zmax_lowzcorr .- min.(z_SF, zmax_lowzcorr)) ./ (zmax_lowzcorr .- zmean_lowzcorr)

    # Use parsed parameters here
    Psb = Psb_hz .+ slope_Psb .* (z_Psb_knee .- min.(z_SF, z_Psb_knee))

    Xuni_sb = rand(N_SF)

    issb = (Xuni_sb .< Psb) 

    noise = randn(N_SF)

    # Use parsed parameters here
    issb_term = issb .* (logBsb - logx0)

    SFR_SF = 10.0 .^ ( logSFRms_SF .+ sigma_MS .* noise
                      .+ logx0 .+ issb_term )


    println("Deal with SFR drawn initially above the SFR limit...")

    too_high_SFRs_indices = findall(x -> x > SFR_max, SFR_SF)
    
    while !isempty(too_high_SFRs_indices)
        
        N_redraw = length(too_high_SFRs_indices)
        
        logSFRms_subset = logSFRms_SF[too_high_SFRs_indices]
        issb_subset = issb[too_high_SFRs_indices]
        
        # Redraw using parsed parameters (sigma_MS, logx0, logBsb)
        SFR_SF[too_high_SFRs_indices] = 10.0 .^ ( 
            logSFRms_subset .+ sigma_MS .* randn(N_redraw)
            .+ logx0 .+ issb_subset .* (logBsb - logx0) 
        )
        
        # Check against SFR_max
        too_high_SFRs_indices = findall(x -> x > SFR_max, SFR_SF)
    end
    

    println("Store the results...")

    cat[!, :SFR] = zeros(Float64, Ngal) 
    cat[!, :issb] = fill(false, Ngal) 

    cat[index_SF, :SFR] = SFR_SF
    cat[index_SF, :issb] = issb

    tstop = time()

    println(Ngal, " galaxy SFRs generated in ", tstop - tstart, "s")
    
    return cat
end
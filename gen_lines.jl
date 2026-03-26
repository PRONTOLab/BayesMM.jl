using DataFrames
using Statistics
using Random
using Cosmology
using DelimitedFiles

# Define cosmology (Planck15 equivalent) 
const cosmo_model = Cosmology.cosmology(h=0.6774, OmegaM=0.3089)

function gen_CO(cat::DataFrame, params::Dict)
    tstart = time() # [7]

    # 1. Compute luminosity distances if missing [8]
    if !("Dlum" in names(cat))
        println("Compute luminosity distances because it was not done previously...")
        cat[!, :Dlum] = [Cosmology.luminosity_dist(cosmo_model, z) for z in cat.redshift]
    end

    println("Compute the CO(1-0) fluxes...")

    # 2. Recipe based on Sargent et al. 2014 to generate CO(1-0) [9]
    Ngal = nrow(cat) # Using nrow instead of length to avoid MethodError [8]
    LprimCO10 = zeros(Ngal)
    
    # Selection for Main Sequence (qflag == false) [9, 10]
    sel = findall(.!cat.qflag)
    # the +10. after SFR is for Kennicutt conversion (Chabrier IMF) [11, 12]
    LprimCO10[sel] = 10.0 .^ (0.81 .* (log10.(cat.SFR[sel]) .+ 10.0) .+ 0.54)

    # Apply Starburst correction [13, 14]
    sb_idx = findall(cat.issb)
    LprimCO10[sb_idx] .*= 10.0 ^ (-0.46)

    cat[!, :LprimCO10] = LprimCO10 # In-place assignment [15]

    # 3. Add scatter on CO1-0 luminosity [16, 17]
    cat.LprimCO10 .*= 10.0 .^ (params["sigma_dex_CO10"] .* randn(Ngal))

    # 4. Calculate observed CO frequency and flux [18, 19]
    nu_CO_obs = params["nu_CO"] ./ (1.0 .+ cat.redshift)
    
    # FIX: Use ustrip.() to discard the units instead of .|> Unitful.NoUnits
    cat[!, :ICO10] = ustrip.(
        cat.mu .* cat.LprimCO10 .* (1.0 .+ cat.redshift).^3 .* 
        nu_CO_obs .^ 2 ./ cat.Dlum.^2 ./ 3.25e7
    )
    
    # 5. Load and Normalize SLEDs (Daddi et al. 2015) [20]
    # readdlm serves as the equivalent to np.loadtxt [21]
    sled_data = readdlm(params["SLED_filename"], comments=true)  
    Jup = sled_data[:, 1]
    Idiffuse = sled_data[:, 2]
    Iclump = sled_data[:, 3]

    # Normalize to 1-0 transition (Julia is 1-indexed) [22, 23]
    Idiffuse ./= Idiffuse[5]
    Iclump ./= Iclump[5]

    # Daddi et al. 2015: log(ICO54/ICO21) = 0.6 * log(<U>) - 0.38
    R54_21 = 10.0 .^ (0.6 .* log10.(cat.Umean) .- 0.38)

    # Note: Python index [25] becomes Julia index [26]; index [24] becomes [27] [28, 29]
    fclump = (Idiffuse[6] .- R54_21 .* Idiffuse[7]) ./ 
             (R54_21 .* (Iclump[7] .- Idiffuse[7]) .- (Iclump[6] .- Idiffuse[6]))

    # Clamping fclump between 0 and 1 [12, 30]
    fclump = clamp.(fclump, 0.0, 1.0)

    # 6. Alternative Starburst SLED (Birkin et al. 2020) [31, 32]
    rJup1_Birkin = [0.9, 0.6, 0.32, 0.35, 0.3, 0.22, 0.22 * (7/8)^2]

    Ngal = nrow(cat)
    ms_sel = findall((.!cat.issb) .&& (.!cat.qflag)) # Main Sequence
    sb_sel = findall(cat.issb)                      # Starburst

    for k in 1:7 
        println("Work on the ICO$(k+1)$k lines")
        col_name = Symbol("ICO$(k+1)$k")
        Ivec = zeros(nrow(cat)) # Initialized as Vector{Float64}

        if params["SLED_SB_Birkin"] == true
            Ivec = zeros(Ngal)
            
            # index k in Python becomes k+1 in Julia for SLED lookup
            Ivec[ms_sel] = (fclump[ms_sel] .* Iclump[k+1] .+ 
                           (1.0 .- fclump[ms_sel]) .* Idiffuse[k+1])
            Ivec[sb_sel] .= rJup1_Birkin[k] * (k + 1)^2 
            
            # Now cat.ICO10 is a pure Float64, so this multiplication works
            Ivec .*= cat.ICO10
        else
            # Unified recipe
            Ivec = cat.ICO10 .* (fclump .* Iclump[k+1] .+ (1.0 .- fclump) .* Idiffuse[k+1])
        end
            
        cat[!, col_name] = Ivec 
    end

    tstop = time()
    println("CO line fluxes of $(nrow(cat)) galaxies generated in $(tstop-tstart)s") # 

    return cat
end


function gen_CII(cat::DataFrame, params::Dict)
    tstart = time() # 

    # 1. Compute luminosity distances if missing
    if !("Dlum" in names(cat))
        println("Compute luminosity distances because it was not done previously...")
        # Use vector comprehension for cosmological calculation [3]
        cat[!, :Dlum] = [Cosmology.luminosity_dist(cosmo_model, z) for z in cat.redshift]
    end

    nu_CII_obs = params["nu_CII"] ./ (1.0 .+ cat.redshift)

    # 2. Lagache & Cousin relation
    if get(params, "generate_Lagache", false) == true
        println("Compute the [CII] fluxes using the Lagache relation....")
        
        # Power and scaling calculations using broadcasting [2]
        cat[!, :LCII_Lagache] = cat.SFR .^ (1.4 .- 0.07 .* cat.redshift) .* 
                                10.0 .^ (7.1 .- 0.07 .* cat.redshift)
        
        # Add scatter using randn (equivalent to np.random.normal)
        cat.LCII_Lagache .*= 10.0 .^ (params["sigma_dex_CII"] .* randn(nrow(cat)))

        # Flux conversion
        cat[!, :ICII_Lagache] = cat.mu .* cat.LCII_Lagache ./ 1.04e-3 ./ 
                                cat.Dlum.^2 ./ nu_CII_obs
    end

    # 3. De Looze relation (HII/starburst galaxies)
    if get(params, "generate_de_Looze", false) == true
        println("Compute the [CII] fluxes using the de Looze relation....")
        
        cat[!, :LCII_de_Looze] = 10.0^7.06 .* cat.SFR
        
        # Add scatter
        cat.LCII_de_Looze .*= 10.0 .^ (params["sigma_dex_CII"] .* randn(nrow(cat)))

        cat[!, :ICII_de_Looze] = cat.mu .* cat.LCII_de_Looze ./ 1.04e-3 ./ 
                                 cat.Dlum.^2 ./ nu_CII_obs
    end

    tstop = time()
    # Using nrow(cat) instead of length(cat) to avoid MethodError [query, 938]
    println("[CII] line fluxes of $(nrow(cat)) galaxies generated in $(tstop-tstart)s") 

    return cat
end


function gen_CI(cat::DataFrame, params::Dict)
    tstart = time()

    if !("Dlum" in names(cat))
        println("Compute luminosity distances because it was not done previously...")
        cat[!, :Dlum] = [Cosmology.luminosity_dist(cosmo_model, z) for z in cat.redshift]
    end

    # Column existence check using names()
    if !("ICO43" in names(cat)) && !("ICO76" in names(cat))
        println("WARNING!!!!! CO fluxes must be generated before the [CI] fluxes! No CI flux generated!!!!!!")
        return cat
    end

    # 1. Setup indices and constants
    nu_CO43 = 4 * params["nu_CO"]
    Ngal = nrow(cat)
    ICI10 = zeros(Ngal)
    ICI21 = zeros(Ngal)

    # Compute only for sources with LIR > 0 (equivalent to np.where)
    sel = findall(cat.LIR .> 0)
    
    if !isempty(sel)
        # 2. CI(1-0) computation
        # Accessing subsets and using broadcasting for element-wise log10 and arithmetic
        logLCO43_LIR_sf = log10.(ustrip.(
            1.04e-3 .* cat.ICO43[sel] .* cat.Dlum[sel].^2 .* 
            nu_CO43 ./ (1.0 .+ cat.redshift[sel]) ./ cat.LIR[sel]
        ))
        
        LCI10_sf = 10.0 .^ (params["a_CI10"] .* logLCO43_LIR_sf .+ params["b_CI10"]) .* 
                   cat.LIR[sel] .* 10.0 .^ (params["sigma_CI10"] .* randn(length(sel)))
        
        ICI10[sel] = ustrip.(
            LCI10_sf .* (1.0 .+ cat.redshift[sel]) ./ 
            (1.04e-3 .* cat.Dlum[sel].^2 .* params["nu_CI10"])
        )

        # 3. CI(2-1) computation
        # the second term is coming from the ratios of nu_obs when going from Lsun to Jy km/s
        logLCO76_LCO43_sf = log10.(cat.ICO76[sel] ./ cat.ICO43[sel]) .+ log10(7.0 / 4.0)
        
        LCI21_sf = 10.0 .^ (params["a_CI21"] .* logLCO76_LCO43_sf .+ params["b_CI21"]) .* 
                   LCI10_sf .* 10.0 .^ (params["sigma_CI21"] .* randn(length(sel)))

        ICI21[sel] = ustrip.(
            LCI21_sf .* (1.0 .+ cat.redshift[sel]) ./ 
            (1.04e-3 .* cat.Dlum[sel].^2 .* params["nu_CI21"])
        )
    end

    cat[!, :ICI10] = ICI10
    cat[!, :ICI21] = ICI21

    tstop = time()
    println("[CI] line fluxes of $(nrow(cat)) galaxies generated in $(tstop-tstart)s")

    return cat
end
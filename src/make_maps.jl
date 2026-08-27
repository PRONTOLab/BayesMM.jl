using WCS
using FITSIO
using Statistics
using Unitful
using UnitfulAstro
using Dates
using DataFrames
using FFTW


"""
Computes the World Coordinate System (WCS) parameters for a 2D celestial map.
"""
function set_wcs_map(cat::DataFrame, pixel_size::Real, params::Dict)
    # 1. Coordinates acquisition and unit handling
    if !("ra" in names(cat)) || !("dec" in names(cat))
        println("generating the coordinates of the sources")
        # gen_radec must be defined in your scope
        ra, dec = gen_radec(cat, params)
    else
        # Match python: np.asarray(cat["ra"]) * u.deg
        ra = cat.ra .* u"°"
        dec = cat.dec .* u"°"
    end
    
    # Safe unit stripping to standard Float64 degrees
    ra_val = ustrip.(u"°", ra)
    dec_val = ustrip.(u"°", dec)
    
    ra_mean = mean(ra_val)
    dec_mean = mean(dec_val)
    
    # 2. Coordinate increments and boundaries
    pix_resol = pixel_size / 3600. # arcseconds to degrees
    
    ra_cen = 0.5 * (maximum(ra_val) + minimum(ra_val))
    dec_cen = 0.5 * (maximum(dec_val) + minimum(dec_val))
    delta_ra = maximum(ra_val) - minimum(ra_val)
    delta_dec = maximum(dec_val) - minimum(dec_val)
    
    # 3. Create WCSTransform (corresponds to wcs.WCS(naxis=2))
    w = WCSTransform(2)
    w.crval = [ra_cen, dec_cen]
    w.crpix = [0.5 * delta_ra / pix_resol, 0.5 * delta_dec / pix_resol]
    w.cdelt = [pix_resol, pix_resol]
    w.ctype = ["RA---TAN", "DEC--TAN"]
    w.cunit = ["deg", "deg"]
    
    # 4. Compute pixel positions (0-based indexing)
    # worldcoords is a 2 x N matrix (RA in row 1, Dec in row 2)
    worldcoords = vcat(ra_val', dec_val')
    pixcoords = world_to_pix(w, worldcoords)
    x = pixcoords[1, :]
    y = pixcoords[2, :]
    
    # Offset the central pixel to ensure all x >= 0 and y >= 0
    w.crpix = [
        0.5 * delta_ra / pix_resol - minimum(x),
        0.5 * delta_dec / pix_resol - minimum(y)
    ]
    
    # Recompute coordinates in the newly centered WCS
    pixcoords = world_to_pix(w, worldcoords)
    x = pixcoords[1, :]
    y = pixcoords[2, :]
    
    pos = [y, x]
    
    # 5. Compute grid dimensions
    # y corresponds to shape[1], x to shape[2] in Julia (1-based indices)
    shape_val = [Int(ceil(maximum(y))), Int(ceil(maximum(x)))]
    shape = [div(i, 2) * 2 + 1 for i in shape_val] # Force odd dimensions
    
    x_edges = collect(-0.5:1.0:(shape[2]-0.5))
    y_edges = collect(-0.5:1.0:(shape[1]-0.5))
    
    wcs_dict = Dict{String, Any}(
        "w" => w,
        "shape" => shape,
        "pos" => pos,
        "x_edges" => x_edges,
        "y_edges" => y_edges
    )
    
    return wcs_dict
end

"""
Writes a 2D map array alongside its WCS headers and comments to a FITS file.
"""
function save_map(filename::String, map_array::AbstractArray, map_prop_dict::Dict, filter_name::String, unit::String, beam_fwhm::Real, input_cat::String)
    println("Write $filename...")
    
    # 1. Build the FITS Header records manually
    h_keys = String[]
    h_vals = Any[]
    h_comms = String[]
    
    # Helper closure to push keywords
    function add_key!(k::String, v::Any, c::String="")
        push!(h_keys, k)
        push!(h_vals, v)
        push!(h_comms, c)
    end
    
    # Extract WCS parameters from the WCSTransform in map_prop_dict
    w = map_prop_dict["w"]
    add_key!("CRVAL1", w.crval[1])
    add_key!("CRVAL2", w.crval[2])
    add_key!("CRPIX1", w.crpix[1])
    add_key!("CRPIX2", w.crpix[2])
    add_key!("CDELT1", w.cdelt[1])
    add_key!("CDELT2", w.cdelt[2])
    add_key!("CTYPE1", w.ctype[1])
    add_key!("CTYPE2", w.ctype[2])
    add_key!("CUNIT1", w.cunit[1])
    add_key!("CUNIT2", w.cunit[2])
    
    # Add other metadata records
    add_key!("COMMENT", nothing, "map")
    add_key!("COMMENT", nothing, "Datas")
    add_key!("BUNIT", unit, "Physical unit of the map")
    add_key!("COMMENT", nothing, "Filter name = $filter_name")
    add_key!("COMMENT", nothing, "beam FWHM = $beam_fwhm arcsec")
    add_key!("COMMENT", nothing, "Input catalog = $input_cat")
    add_key!("DATE", string(Dates.now()), "Date of creation")
    
    # Construct the FITSIO.FITSHeader using the 3-vector constructor
    header = FITSHeader(h_keys, h_vals, h_comms)
    
    # 2. Write the array and header to the FITS file
    FITS(filename, "w") do f
        # Writes map_array into the primary HDU and attaches our custom WCS header
        write(f, map_array; header=header)
    end
end

# Define the constant for converting Gaussian FWHM to Sigma [source: 350]
const gaussian_fwhm_to_sigma = 1.0 / (2.0 * sqrt(2.0 * log(2.0)))

"""
Helper function to compute a 2D weighted spatial histogram.
Replaces `numpy.histogram2d` with a native Julia version optimized for integer-aligned bins.
"""
function compute_histogram2d(y::AbstractVector, x::AbstractVector, y_edges::AbstractVector, x_edges::AbstractVector, weights::AbstractVector)
    ny = length(y_edges) - 1
    nx = length(x_edges) - 1
    histo = zeros(Float64, ny, nx)
    
    for i in eachindex(x)
        # Faster than general search because bins are spaced exactly by 1.0 starting at -0.5:
        # Bin 'idx' contains elements in [idx - 1.5, idx - 0.5)
        iy = floor(Int, y[i] + 0.5) + 1
        ix = floor(Int, x[i] + 0.5) + 1
        
        if 1 <= iy <= ny && 1 <= ix <= nx
            histo[iy, ix] += weights[i]
        end
    end
    return histo
end

"""
Generates celestial maps for multiple filters, performing spatial binning 
and optional beam-smoothing convolution via FFT.
"""
function make_maps(cat::DataFrame, params_maps::Dict, params_sides::Dict)
    flux_filter_list = String[]
    output_path = params_maps["output_path"]
    
    # 1. Create the output directory if missing
    if !isdir(output_path)
        println("Create $output_path")
        mkpath(output_path)
    end

    # Iterate over specified filters, resolutions, and beam sizes [conversation history]
    for (filter_name, pixel_size, beam_fwhm) in zip(params_maps["filter_list"], params_maps["pixel_size"], params_maps["beam_fwhm_list"])
        println("Generate the map for $filter_name...")

        Sname = "S" * filter_name
        push!(flux_filter_list, Sname)

        # 2. Compute missing fluxes if not already in the catalog
        if !(Sname in names(cat))
            println("$filter_name fluxes are not included in the catalog. They are computed now...")
            params_temp = copy(params_sides)
            params_temp["filter_list"] = [filter_name]
            # Call filter flux generator (assumed defined in scope)
            cat = gen_fluxes_filter(cat, params_temp)
        end

        # 3. Setup Coordinate System
        println("Set World Coordinates System...")
        # Pass params_sides to handle coordinate generation internally
        map_prop_dict = set_wcs_map(cat, pixel_size, params_sides)

        # 4. Perform weighted spatial 2D binning of the galaxy fluxes
        histo = compute_histogram2d(
            map_prop_dict["pos"][1], # y pixel positions
            map_prop_dict["pos"][2], # x pixel positions
            map_prop_dict["y_edges"],
            map_prop_dict["x_edges"],
            cat[!, Symbol(Sname)]    # Broadcast selection on Flux column
        )

        # 5. Save raw maps without beam convolution
        if get(params_maps, "gen_map_nobeam_Jy_pix", false) == true
            filename = joinpath(output_path, params_maps["run_name"] * "_" * filter_name * "_nobeam_Jy_pix.fits")
            save_map(filename, histo, map_prop_dict, filter_name, "Jy/pix", 0.0, params_maps["sides_cat_path"])
        end

        if get(params_maps, "gen_map_nobeam_MJy_sr", false) == true
            pixel_sr = (pixel_size * pi / 180.0 / 3600.0)^2 # solid angle of pixel in steradians
            map_temp = (histo ./ pixel_sr) .* 1e-6          # Convert Jy to MJy/sr
            filename = joinpath(output_path, params_maps["run_name"] * "_" * filter_name * "_nobeam_MJy_sr.fits")
            save_map(filename, map_temp, map_prop_dict, filter_name, "MJy/sr", 0.0, params_maps["sides_cat_path"])
        end

        # 6. Smooth the map with the beam if requested
        if get(params_maps, "gen_map_smoothed_Jy_beam", false) == true || 
           get(params_maps, "gen_map_smoothed_MJy_sr", false) == true

            println("Convolve the map by the beam...")

            # Set the convolution kernel stddev in pixels
            sigma_pix = beam_fwhm * gaussian_fwhm_to_sigma / pixel_size

            # Create a 2D Gaussian kernel matching the exact shape of the binned map
            # Dimensions are odd numbers (enforced in set_wcs_map)
            ny, nx = map_prop_dict["shape"][1], map_prop_dict["shape"][2]
            cy = div(ny + 1, 2)
            cx = div(nx + 1, 2)
            kernel = zeros(Float64, ny, nx)
            
            for j in 1:nx, i in 1:ny
                dy = i - cy
                dx = j - cx
                kernel[i, j] = exp(-(dx^2 + dy^2) / (2.0 * sigma_pix^2))
            end

            # Convolve using fast FFT circular convolution (replaces Astropy `convolve_fft`)
            # `fftshift(kernel)` centers the peak at the frequency origin (1,1) to prevent spatial shifts
            histo_conv = real(ifft(fft(histo) .* fft(fftshift(kernel))))

            if get(params_maps, "gen_map_smoothed_Jy_beam", false) == true
                filename = joinpath(output_path, params_maps["run_name"] * "_" * filter_name * "_smoothed_Jy_beam.fits")
                save_map(filename, histo_conv, map_prop_dict, filter_name, "Jy/beam", beam_fwhm, params_maps["sides_cat_path"])
            end

            if get(params_maps, "gen_map_smoothed_MJy_sr", false) == true
                # Sum of peak-normalized kernel * solid angle of pixel * 1e-6
                conv_factor = sum(kernel) * (pixel_size * pi / 180.0 / 3600.0)^2 * 1e-6
                map_temp = histo_conv ./ conv_factor
                filename = joinpath(output_path, params_maps["run_name"] * "_" * filter_name * "_smoothed_MJy_sr.fits")
                save_map(filename, map_temp, map_prop_dict, filter_name, "MJy/sr", beam_fwhm, params_maps["sides_cat_path"])
            end
        end
    end

    return cat
end
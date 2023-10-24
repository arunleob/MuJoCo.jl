module MuJoCo
using UnsafeArrays

include("LibMuJoCo/LibMuJoCo.jl")
include("utils.jl")

using .LibMuJoCo

# Re-export non polluting symbols of LibMuJoCo
begin
    exported_prefixes = ["mj_", "mju_", "mjv_", "mjr_", "mjui_", "mjd_", "mjp_", "mjr", "mjt"]
    for name in names(LibMuJoCo; all = true)
        name_str = string(name)
        if any(startswith(name_str, prefix) for prefix in exported_prefixes) && !endswith(name_str, "_")
            @eval export $name
        end
    end
end

include("carray.jl")
export mj_array, mj_zeros

module Wrappers
    using ..LibMuJoCo
    import ..Utils
    include("wrappers.jl")
    include("visualiser_wrappers.jl")
    include("named_access.jl")
    
    using .NamedAccess
end
using .Wrappers
using .Wrappers.NamedAccess

# Rexport Model and Data
export Model, Data

include("visualiser.jl")

const MODEL_TYPES = Union{Model, NamedAccess.NamedModel}
const DATA_TYPES = Union{Data, NamedAccess.NamedData}


import .Visualiser: visualise!


export init_data, step!, forward!, timestep, reset!, resetkey!, getstate, setstate!
export init_visualiser, install_visualiser, visualise!

include("io.jl")

"""
    init_data(model::Model)

Creates an instance of the data required for the [`step!`](@ref) simulation.

Returns a `Data` object, wrapping the underlying `mjData` object.
"""
function init_data(model::Model)
    data_ptr = mj_makeData(model)
    return Data(data_ptr, model) # Requires a reference to the model to get array sizes
end

function init_data(model::NamedAccess.NamedModel)
    data_ptr = mj_makeData(model)
    return NamedAccess.NamedData(Data(data_ptr, getfield(model, :model)), model) # Requires a reference to the model to get array sizes
end

"""
    step!(model::Model, data::Data)

Runs the simulation forward one time step, modifying the underlying `data` object.
"""
function step!(model::MODEL_TYPES, data::DATA_TYPES)
    mj_step(model, data)
end

"""
    forward!(model::MODEL_TYPES, data::DATA_TYPES)

Same as [`step!`](@ref) but without integrating in time.
"""
function forward!(model::MODEL_TYPES, data::DATA_TYPES)
    mj_forward(model, data)
end

"""
    reset!(m::Model, d::Data)

Resets the data values to their default states. You may equivalently use [`mj_resetData`](@ref).
"""
reset!(m::Model, d::Data) = mj_resetData(m, d)

"""
Resets the data struct to values in the first key frame.
"""
resetkey!(m::Model, d::Data) = mj_resetDataKeyframe(m, d, 0)

"""
    resetkey!(m::Model, d::Data, [keyframe = 1])

Resets the data struct to values in the supplied keyframe. 
    
If no keyframe is specified, the first keyframe is used. The keyframe is a 1-based index into the list supplied by the model's specification.
"""
resetkey!(m::Model, d::Data, keyframe::Integer) = mj_resetDataKeyframe(m, d, keyframe-1)

"""
    timestep(model::MODEL_TYPES)

Extract the solver time-step for the given model.
"""
timestep(model::MODEL_TYPES) = model.opt.timestep

"""

    getstate(m::Model, d::Data)

Extract the state vector of a MuJoCo model.

The state vector is [joint positions, joint velocities, actuator states] with dimension (nq, nv, na). 

See also [`mj_getState`](@ref) and [`setstate!`](@ref).
"""
function getstate(m::Model, d::Data)
    x = mj_zeros(m.nq+m.nv+m.na)
    mj_getState(m, d, x, LibMuJoCo.mjSTATE_PHYSICS)
    return transpose(x)
end

"""

    setstate!(m::Model, d::Data, x::AbstractVector)

Set the state vector of a MuJoCo model.

The state vector is [joint positions, joint velocities, actuator states] with dimension (nq, nv, na). 

See also [`mj_setState`](@ref) and [`getstate`](@ref).
"""
function setstate!(m::Model, d::Data, x::AbstractVector)
    mj_setState(m, d, transpose(x), LibMuJoCo.mjSTATE_PHYSICS)
end

# Handle backwards compatibility
if !isdefined(Base, :get_extension)
    using Requires
end

"""
    init_visualiser()

Loads the packages necessary for the running the visualiser.

Add the following packages to your project to be able to use
the visualisation features, or run "install_visualiser()".

# Packages:
- FFMPEG
- GLFW
- Observables
- PrettyTables
- Printf
- StaticArrays
"""
function init_visualiser()
    @eval Main using FFMPEG, GLFW, Observables, PrettyTables, Printf, StaticArrays
    if isdefined(Base, :get_extension)
        @eval Main Base.retry_load_extensions()
    end
end

"""
    install_visualiser()

Installs the necessary packages for the running the visualiser
into the current running environment.

# Packages:
- FFMPEG
- GLFW
- Observables
- PrettyTables
- Printf
- StaticArrays
"""
function install_visualiser()
    @eval Main import Pkg
    @eval Main Pkg.add(["FFMPEG", "GLFW", "Observables", "PrettyTables", "Printf", "StaticArrays"])
end

@static if !isdefined(Base, :get_extension)
function __init__()
    @static if !isdefined(Base, :get_extension)
        @require FFMPEG = "c87230d0-a227-11e9-1b43-d7ebe4e7570a" begin
            @require GLFW = "f7f18e0c-5ee9-5ccd-a5bf-e8befd85ed98" begin
                @require Observables = "510215fc-4207-5dde-b226-833fc4488ee2" begin
                    @require PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d" begin
                        @require Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7" begin
                            @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" begin
                                include("../ext/VisualiserExt/VisualiserExt.jl")
                            end
                        end
                    end
                end
            end
        end
    end
end
end

end
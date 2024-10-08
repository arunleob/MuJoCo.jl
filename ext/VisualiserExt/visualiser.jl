# Adapted from https://github.com/Lyceum/LyceumMuJoCoViz.jl

function MuJoCo.Visualiser.visualise!(
    m::Model, d::Data; 
    controller = nothing, 
    trajectories = nothing,
    channel::Channel = nothing, 
    preferred_monitor = nothing,
    resolution::Union{Tuple{Integer, Integer}, Nothing} = nothing
)
    modes = EngineMode[]
    !isnothing(controller) && push!(modes, Controller(controller))
    !isnothing(trajectories) && push!(modes, Trajectory(trajectories))
    push!(modes, PassiveDynamics())

    window_size = if isnothing(resolution)
        default_windowsize()
    else
        @assert resolution[1] > 0 "Width of resolution should be greater than 0."
        @assert resolution[2] > 0 "Height of resolution should be greater than 0."
        resolution
    end

    phys = PhysicsState(m, d)
    rendertask = Threads.@spawn run!(Engine(window_size, m, d, Tuple(modes), phys), channel = channel, preferred_monitor=preferred_monitor)
    return phys, rendertask
end

"""
Run the visualiser engine
"""
function run!(e::Engine; channel::Channel = nothing, preferred_monitor = nothing)
    # Have shadows and reflection off by default
    _toggle!(e.ui.scn.flags, 1) #Shadow
    _toggle!(e.ui.scn.flags, 3) #Reflections

    # Show and label contact forces by default
    e.ui.vopt.label = 14 # Label contact force
    _toggle!(e.ui.vopt.flags, 16) # Show contact force
    _toggle!(e.ui.vopt.flags, 17) # Show contact force

    # Render the first frame before opening window
    prepare!(e)
    e.ui.refreshrate = GetRefreshRate()
    e.ui.lastrender = time()

    # Set window position
    if preferred_monitor !== nothing
        preferred_monitor = clamp(preferred_monitor, 1, length(GLFW.GetMonitors()))
        monitor_pos = GLFW.GetMonitorPos(GLFW.GetMonitors()[preferred_monitor])
        monitor_data = GLFW.GetVideoMode(GLFW.GetMonitors()[preferred_monitor])
        GLFW.SetWindowSize(e.manager.state.window, 1440, 810)
        GLFW.SetWindowPos(e.manager.state.window, Int(floor(monitor_pos.x + monitor_data.width/2 - 1440/2)), Int(floor(monitor_pos.y + monitor_data.height/2 - 810/2)))
    end

    GLFW.ShowWindow(e.manager.state.window)

    # Run the simulation/mode in a different thread
    modetask = Threads.@spawn runphysics!(e)

    # Print help info
    println(ASCII)
    println("Press \"F1\" to show the help message.")

    # Run the visuals
    runui!(e, channel = channel)
    wait(modetask)
    return nothing
end

"""
Run the UI
"""
function runui!(e::Engine; channel::Channel = nothing)
    shouldexit = false
    trecord = 0.0
    try
        while !shouldexit

            # Check for window interaction and prepare visualisation
            @lock e.phys.lock begin
                GLFW.PollEvents()
                prepare!(e)
            end

            # Render
            render!(e)
            trender = time()

            # Match the rendering rate to desired rates
            rt = 1 / (trender - e.ui.lastrender)
            @lock e.ui.lock begin
                e.ui.refreshrate = RNDGAMMA * e.ui.refreshrate + (1 - RNDGAMMA) * rt
                e.ui.lastrender = trender
                shouldexit = e.ui.shouldexit | GLFW.WindowShouldClose(e.manager.state.window)
            end

            # Handle frame recording
            tnow = time()
            if e.ffmpeghandle !== nothing && tnow - trecord > 1 / e.min_refreshrate
                trecord = tnow
                recordframe(e)
            end

            # Check for external signals
            if channel != nothing && isready(channel)
                shouldexit = take!(channel)
            end

            yield() # Visualisation should give way to running the physics model
        end
    finally
        @lock e.ui.lock begin
            e.ui.shouldexit = true
        end
        GLFW.DestroyWindow(e.manager.state.window)
    end
    return
end

"""
Prepare the visualisation engine for rendering
"""
function prepare!(e::Engine)
    ui, p = e.ui, e.phys
    m, d = p.model, p.data
    LibMuJoCo.mjv_updateScene(
        m, 
        d, 
        ui.vopt, 
        p.pert, 
        ui.cam, 
        LibMuJoCo.mjCAT_ALL,
        ui.scn
    )
    prepare!(ui, p, mode(e))
    return e
end

"""
Render a frame
"""
function render!(e::Engine)
    w, h = GLFW.GetFramebufferSize(e.manager.state.window)
    rect = mjrRect(Cint(0), Cint(0), Cint(w), Cint(h))
    LibMuJoCo.mjr_render(rect, e.ui.scn, e.ui.con)
    e.ui.showinfo && overlay_info(rect, e)
    GLFW.SwapBuffers(e.manager.state.window)
    return nothing
end

"""
Run the MuJoCo model.

This function handles simulating the model in pause, forward, and reverse mode.
Note that reverse mode is only implemented for the `Trajectory` EngineMode.
"""
function runphysics!(e::Engine)
    p = e.phys
    ui = e.ui
    resettime!(p) # reset sim and world clocks to 0

    try
        while true
            shouldexit, lastrender, reversed, paused, refrate, = @lock_nofail ui.lock begin
                ui.shouldexit, ui.lastrender, ui.reversed, ui.paused, ui.refreshrate
            end

            if shouldexit
                break
            elseif (time() - lastrender) > 1 / e.min_refreshrate
                yield()
                continue
            else
                @lock p.lock begin
                    elapsedworld = time(p.timer)

                    # advance sim
                    if ui.paused
                        pausestep!(p, mode(e))
                    elseif ui.reversed && p.elapsedsim > elapsedworld
                        reversestep!(p, mode(e))
                        p.elapsedsim -= timestep(p.model)
                    elseif !ui.reversed && p.elapsedsim < elapsedworld
                        forwardstep!(p, mode(e))
                        p.elapsedsim += timestep(p.model)
                    end
                end
            end
        end
    finally
        @lock ui.lock begin
            ui.shouldexit = true
        end
    end
    return nothing
end

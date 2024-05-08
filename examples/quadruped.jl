using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using MuJoCo
init_visualiser()

# Get the model
model = load_model(joinpath(@__DIR__, "Go2_model/go2_scene.xml"))
data = init_data(model)

# Start visualiser (non-blocking)
ch = Channel()
phys, rendertask = visualise!(model, data, channel = ch)

# Since the renderer is in a different task, lock it when changing data to prevent a race condition
@lock phys.lock begin
    # Set initial state
    data.qpos[1:7] = [0; 0; 0; 1; 0; 0; 0]
    data.qpos[7 .+ (1:12)] = [0.0; 0.8; -1.6; 0.0; 0.8; -1.6; 0.0; 0.8; -1.6; 0.0; 0.8; -1.6]

    # Get foot position and use it to correct height
    mj_forward(model, data)
    data.qpos[3] = -data.geom_xpos[MuJoCo.mj_name2id(model, MuJoCo.mjOBJ_GEOM, "FL"), 3] + 0.011
end

# Compute discrete jacobians
nx, nu = 2*model.nv, model.nu
A = mj_zeros(nx, nx)
B = mj_zeros(nx, nu)
@lock phys.lock begin
    mjd_transitionFD(model, data, 1e-6, true, A, B, nothing, nothing)
end
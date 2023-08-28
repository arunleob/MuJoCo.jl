using CairoMakie
using LinearAlgebra
using MatrixEquations: ared
using MuJoCo
using MuJoCo.LibMuJoCo

init_visualiser()
isplot = false

# Note: follow along with the DeepMind notebook: 
# https://colab.research.google.com/github/deepmind/mujoco/blob/main/python/LQR.ipynb

# Useful functions
reset!(m::Model, d::Data) = LibMuJoCo.mj_resetData(m, d)
resetkey!(m::Model, d::Data) = LibMuJoCo.mj_resetDataKeyframe(m, d, 1)

# Load humanoid in specific keyframe
model, data = MuJoCo.sample_model_and_data()
resetkey!(model, data)


################## Get control set-point ##################

# We want to torque required to hold the humanoid in this position
# Just using inverse dynamics won't work yet - there's an unphysical vertical 
# contact force with the ground we want to remove first. Here's a hacky approach.
heights = LinRange(-0.001, 0.001, 2001)
u_vert = zeros(length(heights))
for k in eachindex(heights)

    # Set model in position and assume qacc == 0
    resetkey!(model, data)
    forward!(model, data)
    data.qacc .= 0

    # Offset the height and check required vertical forces
    data.qpos[3] += heights[k]
    LibMuJoCo.mj_inverse(model, data)
    u_vert[k] = data.qfrc_inverse[3] # 3 -> z-force
end

# Find height corresponding to minimum ficticous force (best offset)
height = heights[argmin(abs.(u_vert))]

# Plot the relationship just to see - compare force to weight of humanoid
weight = sum(model.body_mass) * norm(model.opt.gravity)

fig = Figure(resolution=(600,400))
ax = Axis(fig[1,1], xlabel="Vertical offsets (mm)", ylabel="Vertical force (N)")
lines!(ax, heights*1000, u_vert)
lines!(ax, heights*1000, weight*ones(length(heights)), linestyle=:dash)
lines!(ax, [height, height]*1000, [minimum(u_vert), maximum(u_vert)], linestyle=:dash)
isplot && display(fig)

# We'll use the best-choice offset to get our required ID forces and save q0
resetkey!(model, data)
forward!(model, data)
data.qacc .= 0
data.qpos[3] += height
qpos0 = vec(copy(data.qpos))
LibMuJoCo.mj_inverse(model, data)
qfrc0 = vec(copy(data.qfrc_inverse))
println("Desired forces qfrc0 acquired")

# Need the corresponding control torque (through the actuators)
M_act = data.actuator_moment
ctrl0 = pinv(M_act)' * qfrc0
println("Control set-point ctrl0 acquired")

# Double-check (note: this works because the humanoid is fully-actuated!)
data.ctrl .= ctrl0
forward!(model, data)
qfrc_test = vec(copy(data.qfrc_actuator))
println("Desired force meets actual? ", all((qfrc_test .≈ qfrc0)[7:end]))

# Run the simulation
reset!(model, data)
data.qpos .= qpos0
data.ctrl .= ctrl0

isplot && visualise!(model, data)


################## LQR Design ##################

# Useful dimensions
nu = model.nu
nv = model.nv

# Body IDs
# TODO: Named access like the Python bindings would be great
id_torso = LibMuJoCo.mj_name2id(model, LibMuJoCo.mjOBJ_XBODY, "torso")
id_lfoot = LibMuJoCo.mj_name2id(model, LibMuJoCo.mjOBJ_XBODY, "foot_left")

# R-matrix just identity
R = Matrix{Float64}(I, nu, nu)

# Get Jacobian for torso CoM
reset!(model, data)
data.qpos .= qpos0
forward!(model, data)
jac_com = zeros(nv,3) # TODO: Document row/column-major
LibMuJoCo.mj_jacSubtreeCom(model, data, jac_com, id_torso)
jac_com = transpose(jac_com)

# Get (left) foot Jacobian for balancing
# TODO: Pass in `nothing` instead of C_NULL?
jac_foot = zeros(nv,3)
LibMuJoCo.mj_jacBodyCom(model, data, jac_foot, C_NULL, id_lfoot) 
jac_foot = transpose(jac_foot)

# Design Q-matrix to balance CoM over foot
jac_diff = jac_com .- jac_foot
Qbalance = jac_diff' * jac_diff

# Now we include a cost on joints deviating from the steady-state.
# Torso already sorted. Left leg should remain rigid. Other joints can move for balance.
# Let's start by getting all the joint indices

"""
Python bindings do this:

# Get all joint names.
joint_names = [model.joint(i).name for i in range(model.njnt)]

# Get indices into relevant sets of joints.
root_dofs = range(6)
body_dofs = range(6, nv)
abdomen_dofs = [
    model.joint(name).dofadr[0]
    for name in joint_names
    if 'abdomen' in name
    and not 'z' in name
]
left_leg_dofs = [
    model.joint(name).dofadr[0]
    for name in joint_names
    if 'left' in name
    and ('hip' in name or 'knee' in name or 'ankle' in name)
    and not 'z' in name
]
balance_dofs = abdomen_dofs + left_leg_dofs
other_dofs = np.setdiff1d(body_dofs, balance_dofs)
"""

# TODO: As a place-holder, here are the indices we need. Should get them ourselves later
root_dofs = 1:6
body_dofs = 7:nv
balance_dofs = [8, 9, 16, 18, 19, 20, 21, 25, 26, 27]
other_dofs = [1, 7, 10, 11, 12, 13, 14, 15, 17, 22, 23, 24]

# Cost coefficients
balance_cost       = 1000       # CoM units large, keep it still
balance_joint_cost = 3          # Joints can move a bit and still balance
other_joint_cost   = 0.3        # Other joints can do whatever

# Construct joint Q matrix
Qjoint = Matrix{Float64}(I, nv, nv)
Qjoint[root_dofs, root_dofs] *= 0
Qjoint[balance_dofs, balance_dofs] *= balance_joint_cost
Qjoint[other_dofs, other_dofs] *= other_joint_cost

# Total Q-matrix
Qpos = balance_cost*Qbalance + Qjoint
Q = [Qpos zeros(nv,nv); zeros(nv, 2nv)]  + (1e-10)I # Add ϵI for positive definite Q

# Get A and B matrices from Jacobian
reset!(model, data)
data.ctrl .= ctrl0
data.qpos .= qpos0

A = zeros(2nv, 2nv)
B = zeros(nu, 2nv)
ϵ = 1e-6
centred = true
LibMuJoCo.mjd_transitionFD(model, data, ϵ, centred, A, B, C_NULL, C_NULL)
A = transpose(A)
B = transpose(B)

# Solve LQR with MatrixEquations.jl (faster than loading ControlSystems.jl)
S = zeros(size(Q,1), size(R,1))
_, _, K, _ = ared(A,B,R,Q,S)

# Write the LQR function
function humanoid_ctrl!(m::Model, d::Data)

    # Get difference in states qpos - qpos0 (this function does quaternion diff)
    Δq = zeros(nv)
    LibMuJoCo.mj_differentiatePos(m, Δq, 1, qpos0, d.qpos)
    Δx = vcat(Δq, data.qvel)

    # Compute controls with LQR
    data.ctrl .= (ctrl0 .- K*Δx)

    return nothing
end

# Initialise the humanoid and visualise
reset!(model, data)
data.qpos .= qpos0
visualise!(model, data, controller=humanoid_ctrl!)

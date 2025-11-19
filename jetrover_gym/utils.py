import jax.numpy as jnp

def T_to_xyz_rpy(T, degrees=False):
    """Transform a homogeneous matrix to (xyz, rpy)."""
    T = jnp.asarray(T)
    R = T[:3, :3]
    xyz = T[:3, 3]

    # XYZ (roll→pitch→yaw) extraction; R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    sy = jnp.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    eps = 1e-9

    # nominal
    roll_nom  = jnp.arctan2(R[2, 1], R[2, 2])
    pitch_nom = jnp.arctan2(-R[2, 0], sy)
    yaw_nom   = jnp.arctan2(R[1, 0], R[0, 0])

    # near gimbal lock (sy ~ 0): set yaw = 0 and derive roll from a different pair
    roll_sing  = jnp.arctan2(-R[1, 2], R[1, 1])
    pitch_sing = jnp.arctan2(-R[2, 0], sy)
    yaw_sing   = jnp.zeros_like(roll_sing)

    cond = sy < eps
    roll  = jnp.where(cond, roll_sing,  roll_nom)
    pitch = jnp.where(cond, pitch_sing, pitch_nom)
    yaw   = jnp.where(cond, yaw_sing,   yaw_nom)

    rpy = jnp.stack([roll, pitch, yaw])
    if degrees:
        rpy = rpy * (180.0 / jnp.pi)
    return xyz, rpy

def xyz_rpy_to_T(xyz, rpy, degrees=False):
    import jax.numpy as jnp
    import numpy as np  # allowed; not strictly required

    xyz = jnp.asarray(xyz).reshape(3,)
    rpy = jnp.asarray(rpy).reshape(3,)
    if degrees:
        rpy = rpy * (jnp.pi / 180.0)

    roll, pitch, yaw = rpy
    cr, sr = jnp.cos(roll),  jnp.sin(roll)
    cp, sp = jnp.cos(pitch), jnp.sin(pitch)
    cy, sy = jnp.cos(yaw),   jnp.sin(yaw)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = jnp.array([
        [cy*cp,             cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,             sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,               cp*sr,             cp*cr]
    ])

    T = jnp.eye(4, dtype=R.dtype)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(xyz)
    return T


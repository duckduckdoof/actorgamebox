
"""
fpe_test.py

Author: Caleb Scott

---

Playing around with env/subenv for sacchade movement.

"""

# IMPORTS

# CONSTANTS

# FUNCTIONS
def init_env_subenv():
    """ Playing around with these envs """
    from configs import globals as g
    from modules import environment as e

    game = g.ROMS[15]
    env = e.simple_atari_env(game)
    obs, _ = env.reset()
    print(f"ENV obs shape: {obs.shape}")

    fpe = e.FramePatchEnv(obs)
    fobs, info = fpe.reset()
    print(f"SUBENV obs shape: {fobs.shape}")

    return env, obs, fpe, fobs, info

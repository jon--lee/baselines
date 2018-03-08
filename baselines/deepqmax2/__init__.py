from baselines.deepqmax2 import models  # noqa
from baselines.deepqmax2.build_graph import build_act, build_train  # noqa
from baselines.deepqmax2.simple import learn, load  # noqa
from baselines.deepqmax2.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)

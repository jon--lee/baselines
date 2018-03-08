from baselines.deepqmax import models  # noqa
from baselines.deepqmax.build_graph import build_act, build_train  # noqa
from baselines.deepqmax.simple import learn, load  # noqa
from baselines.deepqmax.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)
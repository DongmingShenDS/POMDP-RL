# LS_POMDP_LTLf_RL
AIDyS large state pomdp

NOTE for SB3-Contrib (has Recurrent PPO)

    https://sb3-contrib.readthedocs.io/en/master/

NOTE for PPO MultiInputPolicy:

    ppo.py 

    => stable_baselines3.common.policies.MultiInputActorCriticPolicy

    => features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor, optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
    
    => stable_baselines3.common.torch_layers.CombinedExtractor

    => NatureCNN OR nn.Flatten()

    => class NatureCNN(BaseFeaturesExtractor): CNN from DQN nature paper
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# import numba
import tianshou
from tianshou.utils import TensorboardLogger
import gymnasium as gym
import argparse
import os
import pprint

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# @numba.njit
def inv_env_step(
    inv_pipeline: np.ndarray, # pipeline[0] as current holding
    action: int,  # query in time t (current)
    lead: int,    # len(inv_pipeline)-1. query at t will add to on-hand holding at t+lead. usually 1, 2, 3, 4
    demand: int,  # demand in time t
    query_cost: float, holding_cost: float, shortage_penalty: float, # reward
    inv_max: int,    # clip current holding if surpass inv_max
    ):
    inv_pipeline[0] = inv_pipeline[0]+inv_pipeline[1]
    cost = -shortage_penalty*min(0, inv_pipeline[0]-demand) + \
            holding_cost*max(inv_pipeline[0]-demand, 0) + \
            query_cost*action
    inv_pipeline[0] = min(max(inv_pipeline[0]-demand, 0), inv_max)
    # inv_pipeline[1:] = np.roll(inv_pipeline[1:], -1)
    if lead > 1:
        for i in range(1, len(inv_pipeline)-1):
            inv_pipeline[i] = inv_pipeline[i+1]
    inv_pipeline[-1] = action
    return -cost, inv_pipeline

class InvEnv(gym.Env):
    def __init__(self,
        poisson_lambda: float,
        lead: int,    # len(inv_pipeline)-1. query at t will add to on-hand holding at t+lead. usually 1, 2, 3, 4
        query_max: int, # max of query for more stock once
        query_cost: float, holding_cost: float, shortage_penalty: float, # reward
        inv_max: int,    # clip current holding if surpass inv_max
        max_episode_length: int,
        ):
        super(InvEnv, self).__init__()
        self.lead = lead
        self.poisson_lambda = poisson_lambda
        self.query_cost = query_cost
        self.holding_cost = holding_cost
        self.shortage_penalty = shortage_penalty
        self.inv_max = inv_max
        self.action_space = query_max
        self.episode_count = 0
        self.max_episode_length = max_episode_length
        self.inv_pipeline = 4*np.ones(lead+1, dtype=np.int32)
        self.inv_pipeline[0] = 0

    def step(self, action):
        demand = np.random.poisson(lam=self.poisson_lambda)
        self.inv_pipeline[0] = self.inv_pipeline[0]+self.inv_pipeline[1]
        cost = -self.shortage_penalty*min(0, self.inv_pipeline[0]-demand) + \
                self.holding_cost*max(self.inv_pipeline[0]-demand, 0) + \
                self.query_cost*action
        self.inv_pipeline[0] = np.clip(self.inv_pipeline[0]-demand, 0, self.inv_max)
        self.inv_pipeline[1:] = np.roll(self.inv_pipeline[1:], -1)
        self.inv_pipeline[-1] = action
        self.episode_count += 1
        # obs, rew, terminate, truncated, info
        return self.inv_pipeline, -cost, False, self.episode_count==self.max_episode_length, {}
    
    def reset(self):
        self.episode_count = 0
        self.inv_pipeline = 4*np.ones(self.lead+1, dtype=np.int32)
        self.inv_pipeline[0] = 0
        return self.inv_pipeline/self.inv_max, {}

    def render(self):
        print(self.inv_pipeline)

class InvPPO(nn.Module):
    def __init__(self, is_actor: bool, input_dim: int, hidden_width_sizes: list, action_size: int, device: str='cuda'):
        super(InvPPO, self).__init__()
        self._is_actor = is_actor
        self.device = device
        layers = []
        for i,j in hidden_width_sizes:
            layers.append(nn.Linear(i, j))
            layers.append(nn.Tanh())
        self.net = torch.nn.Sequential(*layers)
        self.FC = nn.Linear(hidden_width_sizes[-1][-1], 
            action_size if self._is_actor else 1)

    def forward(self, x, state=None, info={}):
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        #x.requires_grad_(requires_grad=True)
        logits = self.FC(self.net(x))
        if self._is_actor:
            return F.softmax(logits, dim=-1), state
        else:
            return logits

if __name__ == "__main__":
    lead_range = [1, 2, 3, 4, 6, 8, 10]
    #torch.set_float32_matmul_precision('high')
    for lead in lead_range:
        print(f'lead: {lead}')
        parser = argparse.ArgumentParser()
        # env
        seed = 0
        poisson_lambda = 5.
        holding_cost = 1.
        # shortage_penalty
        parser.add_argument('--s-p', type=float, default=4.)
        query_max = 20
        inv_max = 100
        train_max_episode_length = 10000
        test_max_episode_length = 50 if lead < 6 else 500
        gamma = 1-1/train_max_episode_length      # discount factor 0.9999
        # torch para
        lr = 1e-6 if lead < 6 else 1e-3
        one_layer_width = 128 if lead <= 6 else 256
        hidden_width_sizes = [(lead+1, one_layer_width), (one_layer_width, one_layer_width), (one_layer_width, one_layer_width)]
        # ppo_adam_eps
        parser.add_argument('--eps', type=float, default=1e-7 if lead < 8 else 1e-5)
        # tianshou trainer
        parser.add_argument('--buffer-size', type=int, default=1000*1000)
        parser.add_argument('--epoch', type=int, default=1000)
        parser.add_argument('--step-per-epoch', type=int, default=100*10000) # test every "step-per-epoch" steps
        parser.add_argument('--step-per-collect', type=int, default=1000)   # update NN after "step_per_collect" steps
        parser.add_argument('--repeat-per-collect', type=int, default=2)
        parser.add_argument('--batch-size', type=int, default=256)
        parser.add_argument('--training-num', type=int, default=100)
        parser.add_argument('--test-num', type=int, default=100)
        # PPO parameter
        parser.add_argument('--vf-coef', type=float, default=0.5)
        parser.add_argument('--ent-coef', type=float, default=0.01)
        parser.add_argument('--eps-clip', type=float, default=0.2) 
        parser.add_argument('--max-grad-norm', type=float, default=None)
        parser.add_argument('--gae-lambda', type=float, default=0.95)
        parser.add_argument('--rew-norm', type=bool, default=False)
        parser.add_argument('--dual-clip', type=float, default=5.0)
        parser.add_argument('--value-clip', type=bool, default=True)
        """
        :param bool advantage_normalization: whether to do per mini-batch advantage
            normalization. Default to True.
        :param bool recompute_advantage: whether to recompute advantage every update
            repeat according to https://arxiv.org/pdf/2006.05990.pdf Sec. 3.5.
            Default to False.
        :param float vf_coef: weight for value loss. Default to 0.5.
        :param float ent_coef: weight for entropy loss. Default to 0.01.
        :param float max_grad_norm: clipping gradients in back propagation. Default to
            None.
        :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
            Default to 0.95.
        :param bool reward_normalization: normalize estimated values to have std close
            to 1, also normalize the advantage to Normal(0, 1). Default to False.
        :param int max_batchsize: the maximum size of the batch when computing GAE,
            depends on the size of available memory and the memory cost of the model;
            should be as large as possible within the memory constraint. Default to 256.
        :param bool action_scaling: whether to map actions from range [-1, 1] to range
            [action_spaces.low, action_spaces.high]. Default to True.
        :param str action_bound_method: method to bound action to range [-1, 1], can be
            either "clip" (for simply clipping the action), "tanh" (for applying tanh
            squashing) for now, or empty string for no bounding. Default to "clip".
        :param Optional[gym.Space] action_space: env's action space, mandatory if you want
            to use option "action_scaling" or "action_bound_method". Default to None.
        :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
            optimizer in each policy.update(). Default to None (no lr_scheduler).
        :param bool deterministic_eval: whether to use deterministic action instead of
            stochastic action sampled by the policy. Default to False.
        """
        args = parser.parse_known_args()[0]
        ppo_adam_eps = args.eps
        shortage_penalty = args.s_p
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_seed(seed)

        actor = InvPPO(is_actor=True, input_dim=lead+1, hidden_width_sizes=hidden_width_sizes, action_size=query_max, device=device).to(device)
        critic = InvPPO(is_actor=False, input_dim=lead+1, hidden_width_sizes=hidden_width_sizes, action_size=query_max, device=device).to(device)
        #actor = torch.compile(actor)
        #critic = torch.compile(critic)
        # orthogonal initialization
        for m in list(actor.modules()) + list(critic.modules()):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        optim = torch.optim.AdamW(
            list(actor.parameters())+list(critic.parameters()), lr=lr, eps=ppo_adam_eps)
        dist = torch.distributions.Categorical
        policy = tianshou.policy.PPOPolicy(
            actor, critic, optim, dist, discount_factor=gamma,
            max_grad_norm=args.max_grad_norm,
            eps_clip=args.eps_clip,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            gae_lambda=args.gae_lambda,
            reward_normalization=args.rew_norm,
            dual_clip=args.dual_clip,
            value_clip=args.value_clip).to(device)
        
        train_envs = tianshou.env.DummyVectorEnv([
            lambda: InvEnv(
                poisson_lambda, lead, query_max=query_max, query_cost=0.,
                holding_cost=holding_cost, shortage_penalty=shortage_penalty, 
                inv_max=inv_max, max_episode_length=train_max_episode_length
        ) for _ in range(args.training_num)])
        test_envs = tianshou.env.DummyVectorEnv([
            lambda: InvEnv(
                poisson_lambda, lead, query_max=query_max,
                query_cost=0., holding_cost=holding_cost, shortage_penalty=shortage_penalty, 
                inv_max=inv_max, max_episode_length=test_max_episode_length
        ) for _ in range(args.test_num)])
        train_collector = tianshou.data.Collector(policy, train_envs, tianshou.data.VectorReplayBuffer(total_size=args.buffer_size, buffer_num=args.training_num))
        test_collector = tianshou.data.Collector(policy, test_envs)
        """
        :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
        :param Collector train_collector: the collector used for training.
        :param Collector test_collector: the collector used for testing. If it's None,
            then no testing will be performed.
        :param int max_epoch: the maximum number of epochs for training. The training
            process might be finished before reaching ``max_epoch`` if ``stop_fn`` is
            set.
        :param int step_per_epoch: the number of transitions collected per epoch.
        :param int repeat_per_collect: the number of repeat time for policy learning,
            for example, set it to 2 means the policy needs to learn each given batch
            data twice.
        :param int episode_per_test: the number of episodes for one policy evaluation.
        :param int batch_size: the batch size of sample data, which is going to feed in
            the policy network.
        :param int step_per_collect: the number of transitions the collector would
            collect before the network update, i.e., trainer will collect
            "step_per_collect" transitions and do some policy network update repeatedly
            in each epoch.
        :param int episode_per_collect: the number of episodes the collector would
            collect before the network update, i.e., trainer will collect
            "episode_per_collect" episodes and do some policy network update repeatedly
            in each epoch.
        ****Only either one of step_per_collect and episode_per_collect can be specified.
        :param function train_fn: a hook called at the beginning of training in each
            epoch. It can be used to perform custom additional operations, with the
            signature ``f(num_epoch: int, step_idx: int) -> None``.
        :param function test_fn: a hook called at the beginning of testing in each
            epoch. It can be used to perform custom additional operations, with the
            signature ``f(num_epoch: int, step_idx: int) -> None``.
        :param function save_best_fn: a hook called when the undiscounted average mean
            reward in evaluation phase gets better, with the signature
            ``f(policy: BasePolicy) -> None``. It was ``save_fn`` previously.
        :param function save_checkpoint_fn: a function to save training process and
            return the saved checkpoint path, with the signature ``f(epoch: int,
            env_step: int, gradient_step: int) -> str``; you can save whatever you want.
        :param bool resume_from_log: resume env_step/gradient_step and other metadata
            from existing tensorboard log. Default to False.
        :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
            bool``, receives the average undiscounted returns of the testing result,
            returns a boolean which indicates whether reaching the goal.
        :param function reward_metric: a function with signature
            ``f(rewards: np.ndarray with shape (num_episode, agent_num)) ->
            np.ndarray with shape (num_episode,)``, used in multi-agent RL.
            We need to return a single scalar for each episode's result to monitor
            training in the multi-agent RL setting. This function specifies what is the
            desired metric, e.g., the reward of agent 1 or the average reward over
            all agents.
        :param BaseLogger logger: A logger that logs statistics during 
            training/testing/updating. Default to a logger that doesn't log anything.
        :param bool verbose: whether to print the information. Default to True.
        :param bool show_progress: whether to display a progress bar when training.
            Default to True.
        :param bool test_in_train: whether to test in the training phase. Default to
            True.
        """
        exp_path = 'exps/' + f'lead{lead}_p{int(shortage_penalty)}_gamma{gamma}_lr{lr}_eps{ppo_adam_eps}/'
        os.makedirs(exp_path)
        writer = SummaryWriter(exp_path[:-1])
        logger = TensorboardLogger(writer)
        def save_fn(policy): torch.save(policy.state_dict(), exp_path+'policy.pth')
        def save_checkpoint_fn(epoch, env_step, gradient_step):
            ckpt_path = os.path.join(exp_path, f"checkpoint_{epoch}.pth")
            torch.save({"model": policy.state_dict()}, ckpt_path)
            return ckpt_path
        result = tianshou.trainer.onpolicy_trainer(
                policy=policy, train_collector=train_collector, test_collector=test_collector, max_epoch=args.epoch,
                step_per_epoch=args.step_per_epoch, step_per_collect=args.step_per_collect, repeat_per_collect=args.repeat_per_collect,
                episode_per_test=test_max_episode_length, batch_size=args.batch_size, save_best_fn=save_fn, save_checkpoint_fn=save_checkpoint_fn,
                logger=logger
                )
        pprint.pprint(result)

        # watch
        policy.eval()
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num, render=0.)
        rew = result["rews"].mean()
        print(f"Mean reward (over {result['n/ep']} episodes): {rew}")
        # direct test
        e = InvEnv(poisson_lambda, lead, query_max=query_max,
                query_cost=0., holding_cost=holding_cost, shortage_penalty=shortage_penalty, 
                inv_max=inv_max, max_episode_length=test_max_episode_length)
        s = e.reset()[0]
        for i in range(test_max_episode_length):
            a = policy.actor([s])
            a = np.argmax((a[0].cpu().detach().numpy()))
            s, r = e.step(a)[0:2]
            print(s, r)
import gym
from delimg import DeliMGMSE, DeliMGMLE
import argparse
from eval import evaluate_deli
import d4rl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", type=str)
    parser.add_argument("--env_name", type=str, default="hopper-medium-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--vae_feature_dim", type=int, default=5)
    parser.add_argument("--latent_dim", type=int, default=25)
    parser.add_argument("--additional_dim", type=int, default=1)

    parser.add_argument("--buffer_size", type=int, default=10)
    parser.add_argument("--perturb", type=float, default=0.0)
    parser.add_argument("--context_length", type=int, default=20)

    parser.add_argument("--var", type=int, default=0)
    parser.add_argument("--grad_flow", action="store_true")
    parser.add_argument("--st", action="store_true", help="If true, use short-term future to predict")
    parser.add_argument("--flow", action="store_true")

    parser.add_argument("--load", action="store_true")

    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.seed(args.seed)

    algo = None
    if args.algo == "mse":
        algo = DeliMGMSE
    elif args.algo == "mle":
        algo = DeliMGMLE

    model = algo(env, seed=args.seed)
    expert_data_path = f"/workspace/expertdata/dttrajectory/{args.env_name}"
    model.load_data(expert_data_path)
    for i in range(200):
        model.learn(total_timesteps=5000, batch_size=256)
        returns, _ = evaluate_deli(seed=args.seed, env=env, model=model, n_eval_episodes=1, deterministic=True)
        model.diagnostics["evaluations/rewards"].append(returns)
        model._dump_logs()

import argparse

import gym
import d4rl

from bc import DeliBC
from delimse import DeliMGMSE
from eval import evaluate_deli, evaluate_bc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", type=str, default="mse")
    parser.add_argument("--env_name", type=str, default="hopper-medium-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--vae_feature_dim", type=int, default=5)
    parser.add_argument("--latent_dim", type=int, default=25)
    parser.add_argument("--additional_dim", type=int, default=1)

    parser.add_argument("--context_length", type=int, default=20)

    parser.add_argument("--flow", action="store_true")
    parser.add_argument("--expert_goal", action="store_true")

    parser.add_argument("--load", action="store_true")

    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.seed(args.seed)

    algo = None
    evaluator = None
    if args.algo == "mse":
        algo = DeliMGMSE
        evaluator = evaluate_deli
    elif args.algo == "bc":
        algo = DeliBC
        evaluator = evaluate_bc

    filename_head = f"/workspace/jaxlog/"
    filename_tail = f"{args.env_name}/" \
                    f"{args.algo}" \
                    f"-grad{int(args.flow)}" \
                    f"-expgoal{int(args.expert_goal)}" \
                    f"-seed{args.seed}"

    tensorboard_log = filename_head + "tensorboard/" + filename_tail
    expert_data_path = f"/workspace/expertdata/dttrajectory/{args.env_name}"

    if args.load:
        model = algo.load(filename_head + "model/" + filename_tail + "ep-10")
        model.load_data(expert_data_path)
        z = evaluator(args.seed, env=env, model=model, n_eval_episodes=10, deterministic=True)
        print("Score", z)
        exit()

    model = algo(
        env,
        seed=args.seed,
        grad_flow=args.flow,
        # tensorboard_log=tensorboard_log,
        data_path=expert_data_path,
        dropout=args.dropout,
        _init_setup_model=True,
        expert_goal=args.expert_goal,
    )

    # model.load_data(expert_data_path)
    for i in range(200):
        model.offline_learn(total_timesteps=5000, batch_size=4096)
        returns, _ = evaluator(seed=args.seed, env=env, model=model, n_eval_episodes=10, deterministic=True)
        normalized_returns = env.get_normalized_score(returns) * 100
        model.diagnostics["evaluations/rewards"].append(returns)
        model.diagnostics["evaluations/normalized_returns"].append(normalized_returns)
        model._dump_logs()
        if i % 10 == 0:
            model.save(filename_head + "model/" + filename_tail + f"ep-{i}")
            print("Model save")

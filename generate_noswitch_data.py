# generate_noswitch_data.py
"""
Generate single-regime (no switching) evaluation datasets.

Mirrors data_generation.py but uses T_no_switch for every dataset so each
series stays in one regime for its entire length.  Saves to
generated_data_noswitch/ to keep them separate from the switching suite.

Each instance alternates init_state (0, 1, 0, 1, ...) so regime coverage
is balanced across the 30 instances.

Usage:
    python generate_noswitch_data.py
    python generate_noswitch_data.py --n_instances 30 --n 1000 --burn 600
"""
import argparse
from data_generation import MSSwitchGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_instances", type=int, default=30,
                    help="Number of instances per dataset (default 30).")
    ap.add_argument("--n",    type=int, default=1000, help="Series length (default 1000)")
    ap.add_argument("--burn", type=int, default=600,  help="Burn-in (default 600)")
    args = ap.parse_args()

    seeds = [42 + i * 100 for i in range(args.n_instances)]
    print(f"Generating {args.n_instances} no-switch instance(s) per dataset")
    print(f"Seeds: {seeds}")
    print(f"Output: generated_data_noswitch/")
    print(f"init_state alternates 0,1,0,1,... across instances\n")

    for i, seed in enumerate(seeds):
        init_state = i % 2
        print(f"\n=== Instance r{i} (seed={seed}, init_state={init_state}) ===")
        gen = MSSwitchGenerator(save_dir="generated_data_noswitch", seed=seed)
        gen.make_noswitch_datasets_menu(
            n=args.n, burn=args.burn, suffix=f"_r{i}",
            init_state=init_state,
        )


if __name__ == "__main__":
    main()

#!/bin/bash

# Main defaults (breakout, pong, boxing)
# echo "Testing baselines..."
# python main_sgf.py --device cuda:0 --game BreakoutNoFrameskip-v4 --project sgf_modded --config configs/default.yaml --amp --compile --seed 1 --wm_eval decoder --notes "Breakout, default config"
# python main_sgf.py --device cuda:0 --game PongNoFrameskip-v4 --project sgf_modded --config configs/default.yaml --amp --compile --seed 1 --wm_eval decoder --notes "Pong, default config"
# python main_sgf.py --device cuda:0 --game BoxingNoFrameskip-v4 --project sgf_modded --config configs/default.yaml --amp --compile --seed 1 --wm_eval decoder --notes "Boxing, default config"

# Modded: add sacchades
echo "Testing sacchade module..."
python main_mod.py --device cuda:0 --game BreakoutNoFrameskip-v4 --project sgf_modded --config configs/mods_default.yaml --amp --compile --seed 1 --wm_eval decoder --notes "Breakout, modded default config"
python main_mod.py --device cuda:0 --game PongNoFrameskip-v4 --project sgf_modded --config configs/mods_default.yaml --amp --compile --seed 1 --wm_eval decoder --notes "Pong, modded default config"
python main_mod.py --device cuda:0 --game BoxingNoFrameskip-v4 --project sgf_modded --config configs/mods_default.yaml --amp --compile --seed 1 --wm_eval decoder --notes "Boxing, modded default config"

# Modded: sacchade + coordconv positional channels
echo "Testing sacchade module + coordconv..."
python main_mod.py --device cuda:0 --game BreakoutNoFrameskip-v4 --project sgf_modded --config configs/mods_coord.yaml --amp --compile --seed 1 --wm_eval decoder --notes "Breakout, modded (+coordconv) config"
python main_mod.py --device cuda:0 --game PongNoFrameskip-v4 --project sgf_modded --config configs/mods_coord.yaml --amp --compile --seed 1 --wm_eval decoder --notes "Pong, modded (+coordconv) config"
python main_mod.py --device cuda:0 --game BoxingNoFrameskip-v4 --project sgf_modded --config configs/mods_coord.yaml --amp --compile --seed 1 --wm_eval decoder --notes "Boxing, modded (+coordconv) config"

# Modded: sacchade + coordconv positional channels + multires patching
echo "Testing sacchade module + coordconv + multires patching..."
python main_mod.py --device cuda:0 --game BreakoutNoFrameskip-v4 --project sgf_modded --config configs/mods_multi_coord.yaml --amp --compile --seed 1 --wm_eval decoder --notes "Breakout, modded (+coordconv +multres-patch) config"
python main_mod.py --device cuda:0 --game PongNoFrameskip-v4 --project sgf_modded --config configs/mods_multi_coord.yaml --amp --compile --seed 1 --wm_eval decoder --notes "Pong, modded (+coordconv +multres-patch) config"
python main_mod.py --device cuda:0 --game BoxingNoFrameskip-v4 --project sgf_modded --config configs/mods_multi_coord.yaml --amp --compile --seed 1 --wm_eval decoder --notes "Boxing, modded (+coordconv +multres-patch) config"
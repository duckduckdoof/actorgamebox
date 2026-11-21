#!/bin/bash

# Create a new job using command-line args.

SCRIPT="${1:-sgf}"

CUDA="${2:-0}"

GAME="${3:-Pong}"

CONFIG="${4:-base}"

SEED="${5:-27}"

# Run in the foreground
python ${SCRIPT}.py \
	--device cuda:${CUDA} \
	--game ${GAME}NoFrameskip-v4 \
	--project sgf-mod-v3 \
	--config configs/${CONFIG}.yaml \
	--amp \
	--compile \
	--save \
	--seed $SEED \
	--name $GAME.$SEED-$CONFIG \
	--group $GAME

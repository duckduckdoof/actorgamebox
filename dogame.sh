#!/bin/bash

# Kick off trials for an entire game.

GAME="${1:-Pong}"
SEED="${2:-27}"
ITR="${3:-1}"

/home/cwscott3/actorgamebox/kickoff.sh sgf 0 $GAME base $SEED $ITR &
/home/cwscott3/actorgamebox/kickoff.sh mods 1 $GAME sacc $SEED $ITR &
/home/cwscott3/actorgamebox/kickoff.sh mods 2 $GAME sacc-coord $SEED $ITR &

wait
echo "All primary variations finished"


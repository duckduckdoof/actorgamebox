#!/bin/bash

# Kick off trials for an entire game.

GAME="${1:-Pong}"
SEED="${2:-27}"

/home/cwscott3/actorgamebox/kickoff.sh sgf 0 $GAME base $SEED &
/home/cwscott3/actorgamebox/kickoff.sh mods 1 $GAME sacc $SEED &
/home/cwscott3/actorgamebox/kickoff.sh mods 2 $GAME sacc-coord $SEED &

wait
echo "All primary variations finished"


#!/bin/bash

source "$HOME/git/dotfiles/shell-utils"
source "$HOME/git/discovery-utils/source-this.sh"

activate-if-venv causal-vr

export WANDB_ENTITY=indylab
export WANDB_PROJECT=causal-vr

# BERT x KungFu

## Prerequisite
Install KungFu
https://github.com/lsds/KungFu

## Run elastic training with config server
1. start config server `zsh peerlist-server.sh`
2. start config client to set peerlist `zsh peerlist-client`
3. set correct parameters in `run.sh`
4. start training `zsh run_elastic_server.sh`

## Run elastic training with schedule
1. install KungFu on branch `mw-schedule`
2. start training `zsh run_elastic_schedule.sh`

## Run static training
1. start training `zsh run_static.sh`

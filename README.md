# Examining Cooperation in Visual Dialog Models

This repository contains experimental code written for ('Examining cooperation in Visual Dialog Models', Mircea Mironenco, Dana Kianfar, Ke Tran, Evangelos Kanoulas, Efstratios Gavves)[https://arxiv.org/abs/1712.01329], which was presented at the Visually-Grounded Interaction and Language (ViGIL) workshop at NIPS 2017 (https://nips2017vigil.github.io).

The recommended and much cleaner PyTorch version of our code can be found [here](https://github.com/danakianfar/Examining-Cooperation-in-VDM). The model replicates the results of the supervised version of [Learning Cooperative Visual Dialog Agents with Deep Reinforcement Learning](https://arxiv.org/abs/1703.06585). The purpose of this repository is to provide a TensorFlow-based version of the model, which also uses round-based rather than full-dialog updates (this improves the performance of the supervised variant to ~94% MPR @ round 10, beating the reinforcement learning version). Note that running this code requires installing [dill](https://pypi.python.org/pypi/dill).

Please see https://visualdialog.org/ for instructions on how to obtain the dataset.

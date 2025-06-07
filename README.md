There are two projects:

One is for Jax version of RSRP cartpole experiment, the original git repo is: https://github.com/imoneoi/EvolvingConnectivity

One is for RL-STDP method cartpole experiment, the original git repo is: https://github.com/NathanKlineInstitute/netpyne-STDP

To reproduce the figure in RSRP paper:
1. pip install -r requirements.txt   This is the experiment environment for both projects.
2. For RSRP cartpole experiment, run:
   
   python ec_cartpole.py sweep
   
   python ec_cartpole_resevior.py sweep

   python es_sweep_program.py
   
3. For RL-STDP, run:
   
   nrnivmodl mod
   
   python3 neurosim/main.py run

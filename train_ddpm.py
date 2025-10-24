
from diffusion import Diffusion_model
import os
from pathlib import Path
import argparse

def main(args):
   
    experiment_name = args.experiment_name
    #Create experiment directory
    Path('./Experiments/'+experiment_name).mkdir(parents=True, exist_ok=True)
    #Use debug to stop after running 10 samples, for debugging purposes
    diff_model = Diffusion_model(experiment_name,debug=args.debug)
    #Check experiment progress and load
    print('##############################')
    if os.path.exists(diff_model.checkpoint_path):
        print("Previous checkpoint from experiment "+experiment_name+" found in: "+diff_model.checkpoint_path)
        print("Loading model!")
        diff_model.load_params()
        print("Loading complete!")
    else:
        print("No checkpoint found for experiment, beginning fresh training...")
    print('##############################')

    # Start and run the training loop
    if(args.pretrain_ddpm):
        diff_model.run()
    if(args.sample):
        diff_model.origin_sampling(999,starting_t=10,eval=True,subset_factor=1000, samples=1)


#
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training and Testing for DDPM diffusion model')
    parser.add_argument('--pretrain_ddpm', action='store_true', default=False,
                        help='Performs DDPM pretraining')
    parser.add_argument('--experiment_name', type=str, default='diffusion_defense_1',
                        help='Name of experiment to load/save to.')
    parser.add_argument('--sample',action='store_true', default=False,
                        help='Performs origin sampling on model')
    parser.add_argument('--debug',action='store_true', default=False,
                        help='runs model in debug mode (each epoch will run only 1 batch to verify process)')                        
    args = parser.parse_args()

    main(args)
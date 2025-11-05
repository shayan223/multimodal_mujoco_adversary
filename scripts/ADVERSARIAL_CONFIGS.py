

class adversarial_cfg():
    def __init__(self):

        #Whether or not to use attack in testbed
        self.ENABLE_ATTACK = True

        #Turns on dataset collection mode
        self.GENERATE_DATASET = False
        #Prefix for file name to save the dataset you generate
        self.DATA_PREFIX = 'fgsm007'

        #Select Defense method, current options are 'VAE', 'VAE_3d', 'Gaussian', 'DDPM', and None
        self.DEF_METHOD = 'DDPM'

        #Set to True if you want to apply defense modifications to the agent's observation during training
        #Set to False to only make changes to the observation during evaluation
        self.TRAIN_ON_DEF = False

        #Select the target modality, options are: None (will use both), 'velocity', or 'angular'
        self.TARGET_MODALITY = None

        #Override the max steps taken to train the agent (Defualt: 3 Million)
        self.MAX_STEPS_OVERRIDE = 3000000

        #Learning Rate for SAC agent
        self.LEARNING_RATE = 1e-3

        #Epsilon value for scaling FGSM attacks
        self.FGSM_MAGNITUDE = 0.007

        #directory to save vae model weights and dataset
        #Please end in '/' directory indicator
        self.SAVE_PATH = '/home/shayan/github/multimodal_mujoco_adversary/'

        if(self.SAVE_PATH == '/path/to/github/multimodal_mujoco_adversary/'):
            print('#####################')
            print('WARNING: SAVE_PATH not set. Set it to absolute path to the repository directory. Please update it in scripts/ADVERSARIAL_CONFIGS.py')
            print('#####################')
        if(self.SAVE_PATH[-1] != '/'):
            print('#####################')
            print('WARNING: File Path does not end with "/". adding one for you. Please double check its correctness.')
            self.SAVE_PATH += '/'
            print('#####################')
        



##################### Setting of training data #####################################

DATA_DIR = '/home/rsofnc000/Multi-Task-LFD-Framework/repo/TinyVLA/dataset'

TASK_CONFIGS = {
    'example_task_config': { # for local debug
        'dataset_dir': [
            "/media/rl/HDD/data/data/act/new_view/8_29_tennis", # task 1
        ],
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'] # corresponding to image keys saved in h5py files
    },
    'libero_object_no_noops':{
        'dataset_dir': DATA_DIR + '/libero_object_no_noops_succ_t0001_s-0-0', # define the path of the dataset
        'episode_len': 1000, #max length of the episode,
        'camera_names': ['image', 'wrist_image'] # define the camera names which are used as the key when reading data
    },
    'libero_goal_no_noops':{
        'dataset_dir': '/home/A.CARDAMONE7/datasets/TinyVLA_datasets/h5py_libero_goal_no_noops/libero_goal_no_noops_succ_t0001_s-0-0', # define the path of the dataset
        'episode_len': 1000, #max length of the episode,
        'camera_names': ['image', 'wrist_image'] # define the camera names which are used as the key when reading data
    },
    'ur5e_pick_place_delta_all':{
        'dataset_dir': DATA_DIR + '/ur5e_pick_place_delta_all_succ_t0001_s-0-0', 
        'episode_len': 1000, #max length of the episode,
        'camera_names': ['camera_front_image', 'camera_gripper_image']
    },
    'ur5e_pick_place_only_0_4_8_12':{
        'dataset_dir': DATA_DIR + '/ur5e_pick_place_only_0_4_8_12_succ_t0001_s-0-0', 
        'episode_len': 1000, #max length of the episode,
        'camera_names': ['camera_front_image', 'camera_gripper_image']
    },
    'ur5e_pick_place_removed_spawn_regions':{
        'dataset_dir': DATA_DIR + '/ur5e_pick_place_removed_spawn_regions_succ_t0001_s-0-0', 
        'episode_len': 1000, #max length of the episode,
        'camera_names': ['camera_front_image', 'camera_gripper_image']
    },
    'ur5e_pick_place_delta_removed_0_5_10_15':{
        'dataset_dir': DATA_DIR + '/ur5e_pick_place_delta_removed_0_5_10_15_succ_t0001_s-0-0', 
        'episode_len': 1000, #max length of the episode,
        'camera_names': ['camera_front_image', 'camera_gripper_image']
    },
    'ur5e_pick_place_rm_central_spawn':{
        'dataset_dir': DATA_DIR + '/ur5e_pick_place_rm_central_spawn_succ_t0001_s-0-0', 
        'episode_len': 1000, #max length of the episode,
        'camera_names': ['camera_front_image', 'camera_gripper_image']
    },
    'ur5e_pick_place_rm_one_spawn':{
        'dataset_dir': DATA_DIR + '/ur5e_pick_place_rm_one_spawn_succ_t0001_s-0-0', 
        'episode_len': 1000, #max length of the episode,
        'camera_names': ['camera_front_image', 'camera_gripper_image']
    },
    'ur5e_pick_place_rm_12_13_14_15':{
        'dataset_dir': DATA_DIR + '/ur5e_pick_place_rm_12_13_14_15_succ_t0001_s-0-0', 
        'episode_len': 1000, #max length of the episode,
        'camera_names': ['camera_front_image', 'camera_gripper_image']
    }
    
}
####################################################################################

#!!!!!!!!!!!!!!!!!!!!!!Followings are copied from aloha which are not used!!!!!!!!!!!!!!!!!!!!!!
### ALOHA fixed constants
DT = 0.02

FPS = 50

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.controllers import controller_factory

class CIP(object):
    """
    enables functionality for resetting with grasping, safety, etc. 
    construct robosuite env as class EnvName(SingleArmEnv, CIP)
    """
    def __init__(self):
        super(CIP, self).__init__()

    def _setup_ik(self):

        # IK solver 
        ik_config = self.robots[0].controller_config
        ik_config.pop("input_max", None)
        ik_config.pop("input_min", None)
        ik_config.pop("output_max", None)
        ik_config.pop("output_min", None)
        print(ik_config)
        self.IK = controller_factory("IK_POSE", ik_config)

    def set_grasp_heuristic(self, target_matrix, root_body, type="top", wide=False):
        self._setup_ik()

        target_pos = target_matrix[:3,3]
        target_ori_mat = target_matrix[:3,:3]

        # ik 
        qpos = self.IK.ik(target_pos, target_ori_mat)

        # update sim 
        self.sim.data.qpos[:7] = qpos
        self.robots[0].init_qpos = qpos
        self.robots[0].initialization_noise['magnitude'] = 0.0

        # override initial gripper qpos for wide grasp 
        if wide:
            self.sim.data.qpos[self.robots[0]._ref_gripper_joint_pos_indexes] = [0.05, -0.05]


    def set_grasp(self, site_id, root_body, type="top", wide=False):

        self._setup_ik()

        # get ee pose
        ee_pos = self.robots[0].controller.ee_pos
        ee_ori_mat = self.robots[0].controller.ee_ori_mat
        ee_quat = T.mat2quat(ee_ori_mat)
        ee_in_world = T.pose2mat((ee_pos, ee_quat))
        
        # get pose of handle site, drawer 
        site_pos = np.array(self.sim.data.site_xpos[site_id])
        site_ori_mat = self.sim.data.site_xmat[site_id]
        site_ori_mat = np.array(site_ori_mat).reshape(3,3)

        body_id = self.sim.model.body_name2id(root_body)
        body_quat = self.sim.model.body_quat[body_id]
        body_ori_mat = T.quat2mat(body_quat)
        
        # compute target
        target_pos = site_pos
        R_x = T.rotation_matrix(-np.pi/2, np.array([1,0,0]))[:3,:3] 
        R_z = T.rotation_matrix(-np.pi/2, np.array([0,0,1]))[:3,:3]

        if type=='top':
            R_y = np.eye(3) 
        else: 
            R_y = T.rotation_matrix(-np.pi/2, np.array([0,1,0]))[:3,:3]

        target_ori_mat = body_ori_mat @ R_x @ R_z @ R_y

        # ik 
        qpos = self.IK.ik(target_pos, target_ori_mat)

        # update sim 
        self.sim.data.qpos[:7] = qpos
        self.robots[0].init_qpos = qpos
        self.robots[0].initialization_noise['magnitude'] = 0.0

        # override initial gripper qpos for wide grasp 
        if wide:
            self.sim.data.qpos[self.robots[0]._ref_gripper_joint_pos_indexes] = [0.05, -0.05]





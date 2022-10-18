import os 
from os.path import join as pjoin

import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.controllers import controller_factory

from tracikpy import TracIKSolver

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
        ik_config["actuator_range"] = (np.array([-2.7973,-1.6628,-2.7973,-2.9718,-2.7973,-0.0175,-2.7973]),np.array([2.7973,1.6628,2.7973,-0.169,2.7973,3.65,2.7973]))
        self.IK = controller_factory("IK_POSE", ik_config)

        self.robot_name = self.robots[0].name
        self.robot_urdf = pjoin(
                os.path.join(robosuite.models.assets_root, "bullet_data"),
                "{}_description/urdf/{}_arm.urdf".format(self.robot_name.lower(), self.robot_name.lower()),
            )
        self.base_link_name = "panda_link0"
        self.ee_link_name = "panda_link7"
        self.solver = TracIKSolver(
                                    self.robot_urdf,
                                    self.base_link_name,
                                    self.ee_link_name
                                )
        
        # store base offset, eef offset
        self.base_orn_offset_inv = self.IK.base_orn_offset_inv # np.eye(3)
        self.ik_robot_target_pos_offset = self.IK.ik_robot_target_pos_offset
        self.rotation_offset = self.IK.rotation_offset

    def set_grasp_tracik(self, target_matrix, wide=False):
        """
        target_matrix: desired ee pose in world frame
        """

        # self.sim.data.site_xpos[self.sim.model.site_name2id(self.eef_name)]

        self.sim.forward()
        self.render()

        cur_ee_pos = self._eef_xpos
        cur_link7_pos = self.sim.data.body_xpos[self.sim.model.body_name2id("robot0_link7")]
        cur_link7_xmat = self.sim.data.body_xmat[self.sim.model.body_name2id("robot0_link7")].reshape(3,3)
        link7_in_world = np.eye(4)
        link7_in_world[:3,:3] = cur_link7_xmat
        link7_in_world[:3,-1] = cur_link7_pos
        base_pos = self.robots[0].base_pos
        base_ori = self.robots[0].base_ori

        base_pose = (base_pos, base_ori)
        base_in_world = T.pose2mat(base_pose)
        link7_in_base = np.linalg.inv(base_in_world) @ link7_in_world
        
        # adjust for eef offset between mujoco and tracIK 
        # target_pos, target_orn = T.mat2pose(target_matrix)
        # target_pos += self.ik_robot_target_pos_offset
        # rotation = T.quat2mat(target_orn)
        # rotation = self.base_orn_offset_inv @ rotation @ self.rotation_offset[:3, :3]
        # target_orn = T.mat2quat(rotation)
        # world_frame_target = T.pose2mat((target_pos, target_orn))

        # transform into frame of link0 for tracIK
        # base_pos = self.robots[0].base_pos
        # base_ori = self.robots[0].base_ori
        # link0_pos = self.robots[0].base_pos
        # link0_ori = base_ori
        # link0_pose = (link0_pos, link0_ori)
        # link0_in_world = T.pose2mat(link0_pose)
        # ee_in_link0 = np.linalg.inv(link0_in_world) @ world_frame_target
        
        # solve 
        qpos = self.solver.ik(link7_in_base)

        breakpoint()

        # set joints 
        self.sim.data.qpos[:7] = qpos
        self.robots[0].init_qpos = qpos
        self.robots[0].initialization_noise['magnitude'] = 0.0

        # override initial gripper qpos for wide grasp 
        if wide:
            self.sim.data.qpos[self.robots[0]._ref_gripper_joint_pos_indexes] = [0.05, -0.05]
        


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





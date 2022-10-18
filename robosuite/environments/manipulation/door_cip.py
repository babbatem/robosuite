import numpy as np 

from robosuite.environments.manipulation.door import Door 
from robosuite.controllers import controller_factory
import robosuite.utils.transform_utils as T

from robosuite.environments.manipulation.cip_env import CIP

import os

import pickle
import random


class DoorCIP(Door, CIP):
    """docstring for DoorCIP"""
    def __init__(self,
                 robots,
                 env_configuration="default",
                 controller_configs=None,
                 gripper_types="default",
                 initialization_noise="default",
                 use_latch=True,
                 use_camera_obs=True,
                 use_object_obs=True,
                 reward_scale=1.0,
                 reward_shaping=True,
                 early_termination=True,
                 penalize_collisions=True,
                 placement_initializer=None,  # TODO: figure out if this is EVER overidden
                 has_renderer=False,
                 has_offscreen_renderer=True,
                 render_camera="frontview",
                 render_collision_mesh=False,
                 render_visual_mesh=True,
                 render_gpu_device_id=-1,
                 control_freq=20,
                 horizon=1000,
                 ignore_done=False,
                 hard_reset=True,
                 camera_names="agentview",
                 camera_heights=256,
                 camera_widths=256,
                 camera_depths=False,
                 camera_segmentations=None,  # {None, instance, class, element}
                 task_config=None,
                 ee_fixed_to_handle=False):

        self.ee_fixed_to_handle = ee_fixed_to_handle
        use_latch = False
        self.use_latch = False
        self.grasp_pose = None

        # super init 
        super().__init__(robots,
                         env_configuration,
                         controller_configs,
                         gripper_types,
                         initialization_noise,
                         use_latch,
                         use_camera_obs,
                         use_object_obs,
                         reward_scale,
                         reward_shaping,
                         placement_initializer,
                         has_renderer,
                         has_offscreen_renderer,
                         render_camera,
                         render_collision_mesh,
                         render_visual_mesh,
                         render_gpu_device_id,
                         control_freq,
                         horizon,
                         ignore_done,
                         hard_reset,
                         camera_names,
                         camera_heights,
                         camera_widths,
                         camera_depths,
                         camera_segmentations=camera_segmentations)  # {None, instance, class, element}

    def _reset_internal(self):

        super()._reset_internal()

        self._setup_ik()
        if self.ee_fixed_to_handle and type(self.grasp_pose) != type(None):
            #TODO replace random sampling with making sure grasp is good
            sampled_pose = self.grasp_pose
            # self.set_grasp_heuristic(sampled_pose, self.door.root_body, type='top', wide=True)
            # self.set_grasp_heuristic(sampled_pose, self.door.root_body, type='top', wide=True)
            self.set_grasp_tracik(sampled_pose, wide=True)
            self.sim.forward()
            self.robots[0].controller.update(force=True)
            self.robots[0].controller.reset_goal()

            breakpoint()
        else:
            self.sim.forward()

        self.handle_current_progress = self.sim.data.qpos[self.hinge_qpos_addr]

    def reward(self, action=None):
        """
        Reward function for the task. 
        Modified from superclass so as to add dense reward on the hinge qpos as well. 

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the door is opened

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.25], proportional to the distance between door handle and robot arm
            - Rotating: in [0, 0.25], proportional to angle rotated by door handled
              - Note that this component is only relevant if the environment is using the locked door version

        Note that a successfully completed task (door opened) will return 1.0 irregardless of whether the environment
        is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # else, we consider only the case if we're using shaped rewards
        elif self.reward_shaping:
            # Add reaching component
            dist = np.linalg.norm(self._gripper_to_handle)
            reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
            #reward += reaching_reward
            # Add rotating component if we're using a locked door
            if self.use_latch:
                handle_qpos = self.sim.data.qpos[self.handle_qpos_addr]
                reward += np.clip(0.25 * np.abs(handle_qpos / (0.5 * np.pi)), -0.25, 0.25)

            # add hinge qpos component 
            hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
            reward_progress = 0
            if hinge_qpos > self.handle_current_progress: #progress has been made
                reward_progress = hinge_qpos - self.handle_current_progress
                self.handle_current_progress = hinge_qpos
            reward += np.clip(reward_progress, 0, 0.5)

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        # make sure the gripper stays closed
        # self.robots[0].grip_action(self.robots[0].gripper, [-1.0])

        # if terminating prematurely, signal episode end
        # if self._check_terminated():
        # if self.terminated:
        #     done = self.early_termination

        # # record collision and joint_limit info for logging
        # info["collisions"] = self.collisions
        # info["joint_limits"] = self.joint_limits
        # info['task_complete'] = self.sim.data.qpos[self.hinge_qpos_addr]
        # info["collision_forces"] = self.col_mags

        info["success"] = self._check_success()
        return reward, done, info
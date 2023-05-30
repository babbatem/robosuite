from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MonkeyBoxThreeObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.mjcf_utils import array_to_string, string_to_array, xml_path_completion
from robosuite.environments.manipulation.cip_env import CIP

import pickle
import random

class LeverBoxCIP(SingleArmEnv, CIP):
    """
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
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
        renderer="mujoco",
        renderer_config=None,
        ee_fixed_to_handle=False
    ):
        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.table_full_size = (0.8, 0.3, 0.05)
        self.table_offset = (-0.6, -0.5, 0.5)

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer
        self.ee_fixed_to_handle = ee_fixed_to_handle
        self.grasp_pose = None

        SingleArmEnv.__init__(
            self,
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )
        CIP.__init__(self)
    
    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the drawer is opened

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.25], proportional to the distance between drawer handle and robot arm
            - Rotating: in [0, 0.25], proportional to angle rotated by drawer handled
              - Note that this component is only relevant if the environment is using the locked drawer version

        Note that a successfully completed task (drawer opened) will return 1.0 irregardless of whether the environment
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

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        )

        # initialize objects of interest
        self.box = MonkeyBoxThreeObject(
            name="Lever Box",
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.box)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.box,
                x_range=[0.07, 0.09],
                y_range=[-0.01, 0.01],
                rotation=(-np.pi / 2.0 - 0.25, -np.pi / 2.0),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.box,
        )
    
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.object_body_ids = dict()

        self.object_body_ids["base"] = self.sim.model.body_name2id(self.box.box_object)
        self.object_body_ids["top"] = self.sim.model.body_name2id(self.box.top_link)
        self.object_body_ids["slider"] = self.sim.model.body_name2id(self.box.slider_link)
        self.object_body_ids["lever"] = self.sim.model.body_name2id(self.box.lever_link)

        self.box_top_handle_site_id = self.sim.model.site_name2id(self.box.important_sites["top_handle"])
        self.box_slide_handle_site_id = self.sim.model.site_name2id(self.box.important_sites["slide_handle"])
        self.box_lever_handle_site_id = self.sim.model.site_name2id(self.box.important_sites["lever_handle"])
        
        self.top_hinge_qpos_addr = self.sim.model.get_joint_qpos_addr(self.box.joints[0])
        self.slide_hinge_qpos_addr = self.sim.model.get_joint_qpos_addr(self.box.joints[1])
        self.lever_hinge_qpos_addr = self.sim.model.get_joint_qpos_addr(self.box.joints[2])

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # Define sensor callbacks
            @sensor(modality=modality)
            def box_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.object_body_ids["top"]])

            @sensor(modality=modality)
            def top_handle_pos(obs_cache):
                return self._top_handle_xpos
            
            @sensor(modality=modality)
            def slide_handle_pos(obs_cache):
                return self._slide_handle_xpos
            
            @sensor(modality=modality)
            def lever_handle_pos(obs_cache):
                return self._lever_handle_xpos

            @sensor(modality=modality)
            def box_to_eef_pos(obs_cache): # What is obs_cache?
                return (
                    obs_cache["box_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "box_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def top_handle_to_eef_pos(obs_cache):
                return (
                    obs_cache["top_handle_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "top_handle_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )
            
            @sensor(modality=modality)
            def slide_handle_to_eef_pos(obs_cache):
                return (
                    obs_cache["slide_handle_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "slide_handle_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )
            
            @sensor(modality=modality)
            def lever_handle_to_eef_pos(obs_cache):
                return (
                    obs_cache["lever_handle_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "lever_handle_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def top_hinge_qpos(obs_cache):
                return np.array([self.sim.data.qpos[self.top_hinge_qpos_addr]])
            
            @sensor(modality=modality)
            def slide_hinge_qpos(obs_cache):
                return np.array([self.sim.data.qpos[self.slide_hinge_qpos_addr]])
            
            @sensor(modality=modality)
            def lever_hinge_qpos(obs_cache):
                return np.array([self.sim.data.qpos[self.lever_hinge_qpos_addr]])

            sensors = [box_pos, top_handle_pos, slide_handle_pos, lever_handle_pos, 
                       box_to_eef_pos, top_handle_to_eef_pos, slide_handle_to_eef_pos, lever_handle_to_eef_pos, 
                       top_hinge_qpos, slide_hinge_qpos, lever_hinge_qpos]
            names = [s.__name__ for s in sensors]

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # We know we're only setting a single object (the door), so specifically set its pose
            box_pos, box_quat, _ = object_placements[self.box.name]
            box_body_id = self.sim.model.body_name2id(self.box.root_body)
            self.sim.model.body_pos[box_body_id] = box_pos
            self.sim.model.body_quat[box_body_id] = box_quat

    def _check_success(self):
        """
        Check if door has been opened.

        Returns:
            bool: True if door has been opened
        """
        hinge_qpos = self.sim.data.qpos[self.top_hinge_qpos_addr]
        return hinge_qpos > 1.5

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the door handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the door handle
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.box.important_sites["handle"], target_type="site"
            )

    @property
    def _top_handle_xpos(self):
        """
        Grabs the position of the door handle handle.

        Returns:
            np.array: Door handle (x,y,z)
        """
        return self.sim.data.site_xpos[self.box_top_handle_site_id]
    
    @property
    def _slide_handle_xpos(self):
        """
        Grabs the position of the door handle handle.

        Returns:
            np.array: Door handle (x,y,z)
        """
        return self.sim.data.site_xpos[self.box_slide_handle_site_id]
    
    @property
    def _lever_handle_xpos(self):
        """
        Grabs the position of the door handle handle.

        Returns:
            np.array: Door handle (x,y,z)
        """
        return self.sim.data.site_xpos[self.box_lever_handle_site_id]
            
    @property
    def _gripper_to_top_handle(self):
        """
        Calculates distance from the gripper to the door handle.

        Returns:
            np.array: (x,y,z) distance between handle and eef
        """
        return self._top_handle_xpos - self._eef_xpos
    
    @property
    def _gripper_to_slide_handle(self):
        """
        Calculates distance from the gripper to the door handle.

        Returns:
            np.array: (x,y,z) distance between handle and eef
        """
        return self._slide_handle_xpos - self._eef_xpos
    
    @property
    def _gripper_to_lever_handle(self):
        """
        Calculates distance from the gripper to the door handle.

        Returns:
            np.array: (x,y,z) distance between handle and eef
        """
        return self._lever_handle_xpos - self._eef_xpos
    
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
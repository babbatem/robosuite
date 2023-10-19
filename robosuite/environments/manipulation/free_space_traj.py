from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import Task
from robosuite.utils import mjcf_utils
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


# /* Default values for free_space_traj task config parameters */
# https://github.com/ARISE-Initiative/robosuite/blob/vices_iros19/robosuite/scripts/config/FreeSpaceTraj_task_config.hjson


DEFAULT_FREESPACE_CONFIG = {
    "acc_vp_reward_mult": 0.0,
    "action_delta_penalty": 0.0,
    "allow_early_end": False,
    "data_logging": False,
    "dist_threshold": 0.05,
    "distance_penalty_weight": 1,
    "distance_reward_weight": 30.0,
    "ee_accel_penalty": 0,
    "end_bonus_multiplier": 25,
    "energy_penalty": 0,
    "logging_filename": "None",
    "num_already_checked": 0,
    "num_via_points": 3,
    "only_cartesian_obs": True,
    "point_randomization": 0,
    "random_point_order": False,
    "randomize_initialization": True,
    "reward_scale": 1.0,
    "timestep_penalty": 0.0,
    "use_debug_cube": False,
    "use_debug_point": True,
    "use_debug_square": False,
    "use_delta_distance_reward": False,
    "via_point_reward": 100.0
}


class FreeSpaceTraj(SingleArmEnv):
    """
    This class corresponds to the free space trajectory following task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If True, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
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
        camera_segmentations=None,  # {None, instance, class, element},
        task_config=None,
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # task config
        task = task_config if task_config is not None else DEFAULT_FREESPACE_CONFIG
        self.dist_threshold = task['dist_threshold']
        self.timestep_penalty = task['timestep_penalty']
        self.distance_reward_weight = task['distance_reward_weight']
        self.distance_penalty_weight = task['distance_penalty_weight']
        self.use_delta_distance_reward = task['use_delta_distance_reward']
        self.via_point_reward = task['via_point_reward']
        self.energy_penalty = task['energy_penalty']
        self.ee_accel_penalty = task['ee_accel_penalty']
        self.action_delta_penalty = task['action_delta_penalty']
        self.acc_vp_reward_mult = task['acc_vp_reward_mult']
        self.end_bonus_multiplier = task['end_bonus_multiplier']
        self.allow_early_end = task['allow_early_end']
        self.random_point_order = task['random_point_order']
        self.point_randomization = task['point_randomization']
        self.randomize_initialization = task['randomize_initialization']

        # Note: should be mutually exclusive
        self.use_debug_cube = task['use_debug_cube']
        self.use_debug_square = task['use_debug_square']
        self.use_debug_point = task['use_debug_point']

        # create ordered list of random points in 3D space the end-effector must touch
        self.num_already_checked = task['num_already_checked']
        self.num_via_points = task['num_via_points']
        self._place_points()
        self.next_idx = task['num_already_checked']

        self.finished_time = None

        super().__init__(
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

    def reward(self, action):
        """
        Return the reward obtained for a given action. Overall, reward increases as the robot
        checks via points in order.
        """
        reward = 0

        dist = np.linalg.norm(self.robots[0].controller.ee_pos[:3] - self.via_points[self.next_idx][1:])

        # check if robot hit the next via point
        if self.finished_time is None and dist < self.dist_threshold:
            self.sim.model.site_rgba[self.next_idx] = mjcf_utils.GREEN
            self.via_points[self.next_idx][0] = 1  # mark as visited
            self.next_idx += 1
            reward += self.via_point_reward

            # if there are still via points to go
            if self.next_idx != self.num_via_points:
                # color next target red
                self.sim.model.site_rgba[self.next_idx] = mjcf_utils.RED

        # reward for remaining distance
        else:
            # if robot starts 0.3 away and dist_threshold is 0.05: [0.005, 0.55] without scaling
            if not self.use_delta_distance_reward:
                reward += self.distance_reward_weight * (1 - np.tanh(5 * dist))  # was 10
            else:
                prev_dist = np.linalg.norm(self.prev_ee_pos[:3] - self.via_points[self.next_idx][1:])
                reward += self.distance_reward_weight * (prev_dist - dist)
                reward -= self.distance_penalty_weight * np.tanh(10 * dist)

        # What we want is to reach the points fast
        # We add a reward that is proportional to the number of points crossed already
        reward += self.next_idx * self.acc_vp_reward_mult

        # penalize for taking another timestep
        # (e.g. 0.001 per timestep, for a total of 4096 timesteps means a penalty of 40.96)
        reward -= self.timestep_penalty

        # penalize for jerkiness
        torques = self.robots[0].controller.torques
        reward -= self.energy_penalty * np.sum(np.abs(torques))
        # reward -= self.ee_accel_penalty * np.mean(abs(self.ee_acc))
        # reward -= self.action_delta_penalty * np.mean(abs(self._compute_a_delta()[:3]))

        return reward

    def _place_points(self):
        """
        Randomly generate via points to set self.via_points. Note that each item in self.via_points
        is 4 elements long: the first element is a 1 if the point has been checked, 0 otherwise.
        The remaining 3 elements are the x, y and z position of the point.
        """
        min_vals = {'x': 0.4, 'y': -0.1, 'z': 1.4}
        max_vals = {'x': 0.6, 'y': 0.1, 'z': 1.6}

        def place_point(min_vals, max_vals):
            pos = []
            for axis in ['x', 'y', 'z']:
                pos.append(np.random.uniform(low=min_vals[axis], high=max_vals[axis]))
            return pos

        if self.use_debug_point:
            self.via_points = [np.array((0.5, -0.15, 1.4))]
            self.num_via_points = 1
        elif self.use_debug_square:
            box_1 = np.array((0.5, -0.15, 1.4))
            box_2 = box_1 + np.array((0.0, 0.3, 0.0))
            box_3 = box_2 + np.array((0.0, 0.0, -0.2))
            box_4 = box_1 + np.array((0.0, 0.0, -0.2))
            if self.random_point_order:
                if np.random.choice([True, False]):
                    # clockwise
                    self.via_points = [box_1, box_2, box_3, box_4]
                else:
                    # counter-clockwise
                    self.via_points = [box_1, box_4, box_3, box_2]
            else:
                # clockwise
                self.via_points = [box_1, box_2, box_3, box_4]
            if self.point_randomization != 0:
                # preserve constant x
                randomized_viapoints = []
                for p in self.via_points:
                    p[1:] += np.random.randn(2) * self.point_randomization
                    randomized_viapoints.append(p)
                self.via_points = randomized_viapoints
            self.num_via_points = len(self.via_points)
        elif self.use_debug_cube:
            self.via_points = [[min_vals['x'], max_vals['y'], max_vals['z']],
                               [min_vals['x'], min_vals['y'], max_vals['z']],
                               [max_vals['x'], min_vals['y'], max_vals['z']],
                               [max_vals['x'], max_vals['y'], max_vals['z']],
                               [min_vals['x'], min_vals['y'], min_vals['z']],
                               [min_vals['x'], max_vals['y'], min_vals['z']],
                               [max_vals['x'], min_vals['y'], min_vals['z']],
                               [max_vals['x'], max_vals['y'], min_vals['z']]]
            self.num_via_points = len(self.via_points)
        else:
            self.via_points = [place_point(min_vals, max_vals) for _ in range(self.num_via_points)]

        final_via_points = []
        for i, point in enumerate(self.via_points):
            final_via_points.append([1 if i < self.num_already_checked else 0, *point])
        self.via_points = np.array(final_via_points)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = [0,0,0]
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = EmptyArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])
        self.model = Task(mujoco_arena=mujoco_arena,  
                          mujoco_robots=[robot.robot_model for robot in self.robots],)

        self.robot_contact_geoms = self.robots[0].robot_model.contact_geoms

        # ensure both robots start at similar positions:
        self.robots[0]._init_qpos = np.array(
            [0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi / 4])

        # add sites for each point
        for i, data in enumerate(self.via_points):
            point = data[1:]  # ignore element 0 (just indicates whether point has been pressed)

            # pick a color
            color = None
            if i < self.num_already_checked:
                color = mjcf_utils.GREEN
            elif i == self.num_already_checked:
                color = mjcf_utils.RED
            else:
                color = mjcf_utils.BLUE

            site = mjcf_utils.new_site(name='via_point_%d' % i,
                                       pos=tuple(point),
                                       size=(0.01,),
                                       rgba=color)
            self.model.worldbody.append(site)

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

            # cube-related observables
            @sensor(modality=modality)
            def via_pos(obs_cache):
                return self.via_points.flatten()

            sensors = [via_pos]
            names = [s.__name__ for s in sensors]

            # Create observables
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
        self._place_points()

        # reset joint positions
        if not self.deterministic_reset:
            self.sim.data.qpos[self.robots[0]._ref_joint_pos_indexes] = np.array(
                self.robots[0].init_qpos + np.random.randn(7) * 0.02)
        else:
            self.sim.data.qpos[self.robots[0]._ref_joint_pos_indexes] = np.array(self.robots[0].init_qpos)

        self.next_idx = self.num_already_checked
        self.timestep = 0
        self.finished_time = None

    def _check_success(self):
        """
        Returns True if task is successfully completed
        """
        return self.next_idx == self.num_via_points
    
    def _post_action(self, action):
        """
        If something to do
        """

        reward, done, info = super()._post_action(action)

        # allow episode to finish early
        if self._check_success():
            reward += self.end_bonus_multiplier * (self.horizon - self.timestep)
            self.finished_time = self.timestep
            if self.allow_early_end:
                done = True
            else:
                # reset goal
                self.next_idx -= 1
                self.via_points[self.next_idx][0] = 0  # mark as not visited

        info['add_vals'] = ['percent_viapoints_', 'finished_time', 'finished']
        info['percent_viapoints_'] = self.next_idx / self.num_via_points if self.finished_time is None else 1
        info['finished'] = 1 if self.finished_time is not None else 0
        info['finished_time'] = self.finished_time if self.finished_time is not None else self.horizon

        # logger.debug('Process {process_id}, timestep {timestep} reward: {reward:.2f}, checked vps: {viapoints}'.format(
        #     process_id=str(id(multiprocessing.current_process()))[-5:],
        #     timestep=self.timestep,
        #     reward=reward,
        #     viapoints=self.next_idx))

        # if self.data_logging:
        #     # NOTE: counter is -1 because it is already incremented in robot_arm.py
        #     eef_position = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')])
        #     dist = np.linalg.norm(eef_position - self.via_points[self.next_idx][1:]) if not done else 0
        #     self.file_logging['reward'][self.counter - 1] = reward
        #     self.file_logging['distance_to_via_point'][self.counter - 1] = dist
        #     self.file_logging['next_via_point_idx'][
        #         self.counter - 1] = self.next_idx if self.finished_time is None else -1
        #     if not done:
        #         self.file_logging['current_via_point'][self.counter - 1] = self.via_points[self.next_idx][1:]
        #     self.file_logging['distance_reward'][self.counter - 1] = \
        #     [self.distance_reward_weight * (1 - np.tanh(5 * dist)), self.via_point_reward][dist < self.dist_threshold]
        #     self.file_logging['acc_vp_reward'][self.counter - 1] = self.next_idx * self.acc_vp_reward_mult
        #     self.file_logging['timestep_penalty'][self.counter - 1] = -self.timestep_penalty
        #     self.file_logging['energy_penalty'][self.counter - 1] = -self.energy_penalty * self.total_joint_torque
        #     self.file_logging['ee_accel_penalty'][self.counter - 1] = -self.ee_accel_penalty * np.mean(abs(self.ee_acc))

        return reward, done, info    

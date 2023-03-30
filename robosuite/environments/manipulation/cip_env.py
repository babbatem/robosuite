import os 
from os.path import join as pjoin
import math 

from os import listdir
from os.path import isfile, join
import pickle 

import numpy as np
import yaml 
from numpy import linalg as LA

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.controllers import controller_factory

from ikflow.utils import get_ik_solver, get_solution_errors

with open("ikflow/model_descriptions.yaml", "r") as f:
    MODEL_DESCRIPTIONS = yaml.safe_load(f)

GRIP_NAMES = {'DoorCIP': 'Door_handle', 'DrawerCIP': 'Drawer_handle','SlideCIP': 'Slide_grip','LeverCIP': 'Lever_lever'}
OBJECT_NAMES = {'DoorCIP': 'door', 'DrawerCIP': 'drawer','SlideCIP': 'slide','LeverCIP': 'lever'}

class CIP(object):
    """
    enables functionality for resetting with grasping, safety, etc. 
    construct robosuite env as class EnvName(SingleArmEnv, CIP)
    """
    def __init__(self, ik_pos_tol=1e-3, samples_per_pose=50, p_constant=1, m_constant=1, ttt_constant = 1, manip_strategy = 'old', manipulability_flip = 'superaverage', follow_demo = False):
        super(CIP, self).__init__()
        self.solver = None
        self.setGeomIDs()
        self.ik_pos_tol = ik_pos_tol
        self.samples_per_pose = samples_per_pose
        self.solver_kwargs = {
                                "latent_noise_distribution" : "gaussian",
                                "latent_noise_scale" : 0.75,
                                "refine_solutions" : True
                            }

        print(self.__class__.__name__)
        self.task_mean = self.calculate_task_vector()
        self.p_constant = p_constant
        self.m_constant = m_constant
        self.ttt_constant = ttt_constant
        self.manip_strategy = manip_strategy
        self.manipulability_flip = manipulability_flip
        self.follow_demo = follow_demo
        print('P_constant')
        print(self.p_constant)
        print('M_constant')
        print(self.m_constant)
        print('ttt_constant')
        print(self.ttt_constant)
        print('Manipulation strategy')
        print(self.manip_strategy)

        self.demo_data_list, self.grasp_data_list = self.get_task_demos_and_grasps()
        print('len_grasp_data_list')
        print(len(self.grasp_data_list))
        stacked_arrays = np.stack(self.grasp_data_list)
        reshaped_array = stacked_arrays.reshape(len(self.grasp_data_list), -1)
        unique_rows, indices = np.unique(reshaped_array, axis = 0, return_index = True)
        print("unique_rows")
        print(unique_rows.shape[0])


    def get_task_demos_and_grasps(self):

        demos_path = "./auto_demos/" + str(self.__class__.__name__)
        demo_files = [f for f in listdir(demos_path) if isfile(join(demos_path, f))]
        folder_len = len(demo_files)
        print("folder_len")
        print(folder_len)
        grasps_path = "./auto_demos/" + str(self.__class__.__name__) + "_grasps"
        grasp_files = [f for f in listdir(grasps_path) if isfile(join(grasps_path, f))]
        grasp_folder_len = len(grasp_files)
        print("grasp_folder_len")
        print(grasp_folder_len)

        demo_data_list = []
        grasp_data_list = []

        for zz in range(len(demo_files)):
            demo_file = demo_files[zz]
            grasp_file = grasp_files[zz]
            full_demo_path  = demos_path + "/" + demo_file
            full_grasp_path = grasps_path + "/" + grasp_file
            demo_data = pickle.load(open(full_demo_path,"rb"), encoding='latin1')
            grasp_data = pickle.load(open(full_grasp_path,"rb"), encoding='latin1')
            demo_data_list.append(demo_data)
            grasp_data_list.append(grasp_data)

        return demo_data_list, grasp_data_list

    def calculate_task_vector(self):
        demos_path = "./auto_demos/" + str(self.__class__.__name__)
        demo_files = [f for f in listdir(demos_path) if isfile(join(demos_path, f))]
        folder_len = len(demo_files)
        #demo_file = demo_files[np.random.randint(folder_len-1)]
        outer_loop_actions = []
        for demo_file in demo_files:
            full_demo_path  = demos_path + "/" + demo_file
            demo_data = pickle.load(open(full_demo_path,"rb"), encoding='latin1')

            s0, a, r, done_p, sp = demo_data[0]
            
            actions = []

            for i, transition_tuple in enumerate(demo_data):
                s, a, r, done_p, sp = transition_tuple
                a = a[0:6]#np.pad(a, (0, 3), 'constant', constant_values=(0, 0))
                mag = np.sqrt(a.dot(a))
                actions.append(a/mag)
            actions = np.array(actions)
            action_mean = np.mean(actions, axis = 0)
            action_mean /= np.sqrt(action_mean.dot(action_mean))
            # print('action_mean')
            # print(action_mean)
            outer_loop_actions.append(action_mean)
        outer_loop_actions = np.array(outer_loop_actions)
        outer_action_mean = np.mean(outer_loop_actions, axis = 0)
        outer_action_mean /= np.sqrt(outer_action_mean.dot(outer_action_mean))
        # print('Outer action mean')
        # print(outer_action_mean)
        # print(np.sqrt(outer_action_mean.dot(outer_action_mean)))

        return outer_action_mean

    def closest (self, num, arr):
        curr = arr[0]
        for val in arr:
            new = num - val
            old = num - curr
            if np.sqrt(new.dot(new)) < np.sqrt(old.dot(old)):
                curr = val
        return curr


    def find_closest_grasp_traj(self,grasp_pose):

        closest = self.grasp_data_list[0]
        indice = 0
        for zz in range(len(self.grasp_data_list)):
            diff_new = self.grasp_data_list[zz] - grasp_pose
            diff_old = closest - grasp_pose
            new_dist = LA.norm(diff_new, 'fro')
            old_dist = LA.norm(diff_old, 'fro')
            if new_dist < old_dist:
                closest = self.grasp_data_list[zz]
                indice = zz
        closest_trajectory = self.demo_data_list[indice]

        return closest, closest_trajectory


    def calculate_demo_specific_task_vector(self,grasp_pose):

        # print("Grasp_pose")
        # print(grasp_pose)

        # closest = self.grasp_data_list[0]
        # indice = 0
        # for zz in range(len(self.grasp_data_list)):
        #     diff_new = self.grasp_data_list[zz] - grasp_pose
        #     # print('A')
        #     # print(self.grasp_data_list[zz])
        #     # print('B')
        #     # print(grasp_pose)
        #     # print('C')
        #     # print(diff_new)
        #     diff_old = closest - grasp_pose
        #     new_dist = LA.norm(diff_new, 'fro')
        #     old_dist = LA.norm(diff_old, 'fro')
        #     if new_dist < old_dist:
        #         closest = self.grasp_data_list[zz]
        #         indice = zz

        # # print("Grasp_pose")
        # # print(grasp_pose)
        # # print('Closest')
        # # print(self.grasp_data_list[indice])
        # closest_trajectory = self.demo_data_list[indice]

        closest_grasp, closest_trajectory = self.find_closest_grasp_traj(grasp_pose)

        actions = []
        actions2 = []

        for i, transition_tuple in enumerate(closest_trajectory):
            s, a, r, done_p, sp = transition_tuple
            a = a[0:6]#np.pad(a, (0, 3), 'constant', constant_values=(0, 0))
            actions2.append(a)
            mag = np.sqrt(a.dot(a))
            actions.append(a/mag)
        actions = np.array(actions)
        action_mean = np.mean(actions, axis = 0)
        action_mean /= np.sqrt(action_mean.dot(action_mean))

        #print(self.grasp_data_list)

        #breakpoint()
        #     demo_file = demo_files[zz]
        #     full_demo_path  = demos_path + "/" + demo_file
        #     demo_data = pickle.load(open(full_demo_path,"rb"), encoding='latin1')
        #     s0, a, r, done_p, sp = demo_data[0]

        #     actions = []
        #     actions2 = []
        #     start_positions.append(s0[:7])

        #     for i, transition_tuple in enumerate(demo_data):
        #         s, a, r, done_p, sp = transition_tuple
        #         a = a[0:6]#np.pad(a, (0, 3), 'constant', constant_values=(0, 0))
        #         actions2.append(a)
        #         mag = np.sqrt(a.dot(a))
        #         actions.append(a/mag)
        #     actions = np.array(actions)
        #     action_mean = np.mean(actions, axis = 0)
        #     action_mean /= np.sqrt(action_mean.dot(action_mean))
        #     #print('Len')
        #     #print(len(actions2))
        #     # print('action_mean')
        #     # print(action_mean)
        #     outer_loop_actions.append(action_mean)
        # outer_loop_actions = np.array(outer_loop_actions)
        # outer_action_mean = np.mean(outer_loop_actions, axis = 0)
        # outer_action_mean /= np.sqrt(outer_action_mean.dot(outer_action_mean))

        # closest_start = self.closest(grasp_pose, start_positions)

        # # print('Outer action mean')
        # # print(outer_action_mean)
        # # print(np.sqrt(outer_action_mean.dot(outer_action_mean)))

        # return np.array(actions2)
        return action_mean


    def check_gripper_contact(self, task_name):
        for contact_index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[contact_index]
            if self.contactBetweenGripperAndSpecificObj(contact, GRIP_NAMES[task_name]):
                return True
        return False

    def _setup_ik(self):

        # Build IkflowSolver and set weights
        model_name = "panda_lite"
        model_weights_filepath = MODEL_DESCRIPTIONS[model_name]["model_weights_filepath"]
        robot_name = MODEL_DESCRIPTIONS[model_name]["robot_name"]
        hparams = MODEL_DESCRIPTIONS[model_name]
        ik_solver, hyper_parameters, robot_model = get_ik_solver(model_weights_filepath, hparams, robot_name)
        self.solver = ik_solver

    def set_qpos_and_update(self, qpos):
        self.sim.data.qpos[:7] = qpos
        self.robots[0].init_qpos = qpos
        self.robots[0].initialization_noise['magnitude'] = 0.0

        self.sim.forward()
        self.robots[0].controller.update(force=True)
        self.robots[0].controller.reset_goal()
        self.robots[0].controller.update_initial_joints(qpos)

    def reset_to_qpos(self, qpos, wide=False):

        if wide:
            self.sim.data.qpos[self.robots[0]._ref_gripper_joint_pos_indexes] = [0.05, -0.05]

        # set joints 
        self.set_qpos_and_update(qpos)

        # ensure valid
        collision_score = self.isInvalidMJ()
        if collision_score != 0:
            return False

        if self.checkJointPosition(qpos):
            return False 

        return True

    def reset_to_grasp(self, grasp_pose, wide=False, optimal_ik=False, verbose=False, frame=None):
        grasp_pose_old = grasp_pose
        if self.solver is None: 
            self._setup_ik()

        # maybe change frame
        if frame is not None: 
            assert frame.shape == (4,4)
            grasp_pose = frame @ grasp_pose 

      
        # override initial gripper qpos for wide grasp 
        if wide:
            self.sim.data.qpos[self.robots[0]._ref_gripper_joint_pos_indexes] = [0.05, -0.05]

        qpos = None
        best_manip = -np.inf
        best_qpos = None
        candidate_qpos, ee_target = self.solve_ik(grasp_pose)
        print("manipulability_flip")
        print(self.manipulability_flip)
        if self.manipulability_flip == "demoaverage":
            self.task_mean =  self.calculate_demo_specific_task_vector(grasp_pose_old)
            print("Self_task_mean_demoaverage")
            print(self.task_mean)
        else:
            assert self.manipulability_flip == "superaverage"
            self.task_mean =  self.calculate_task_vector()
            print("Self_task_mean_superaverage")
            print(self.task_mean)

        if len(candidate_qpos) == 0: 
            if verbose: 
                print('solver returned none')
                self.render()
            return False 

        if self.follow_demo == True:

                closest_grasp, closest_trajectory = self.find_closest_grasp_traj(grasp_pose_old)


        for qpos in candidate_qpos:

            if self.follow_demo == True:

                if self.checkJointPosition(qpos):
                    if verbose: 
                        print('joint limit')
                        self.render()
                    qpos = None 
                    continue

                # set joints 
                self.sim.data.qpos[:7] = qpos
                self.sim.forward()

                # ensure valid
                collision_score = self.isInvalidMJ()
                if collision_score != 0:
                    if verbose: 
                        print('collision')
                        self.render()
                    qpos = None 
                    continue 

                # maybe keep qpos w/ highest manipulability score 
                if not optimal_ik: 
                    print('NOT ENTERING')
                    best_qpos = qpos 
                    break

                else:
                    wp_sum = 0
                    for i, transition_tuple in enumerate(closest_trajectory):
                        
                        if self.manip_strategy == 'ellipsoid':
                            w,p,wp = self.check_manipulability_ellipsoid()
                            wp_sum += wp
                        elif self.manip_strategy == 'paper':
                            w,p,wp = self.check_manipulability_paper()
                            wp_sum += wp
                        elif self.manip_strategy == 'old':
                            w,p,wp = self.check_manipulability_old()
                            wp_sum += wp

                        s, a, r, done_p, sp = transition_tuple
                        
                        self.step(a)
                        #self.render()
                        #self.sim.forward()
                        

                    if wp_sum > best_manip:
                        best_manip = wp_sum 
                        best_qpos = qpos
                    self._reset_internal_hacky()
                    #self.render()
                    #self.sim.forward()

            else:

                if self.checkJointPosition(qpos):
                    if verbose: 
                        print('joint limit')
                        self.render()
                    qpos = None 
                    continue

                # set joints 
                self.sim.data.qpos[:7] = qpos
                self.sim.forward()

                # ensure valid
                collision_score = self.isInvalidMJ()
                if collision_score != 0:
                    if verbose: 
                        #print('collision')
                        self.render()
                    qpos = None 
                    continue 

                # maybe keep qpos w/ highest manipulability score 
                if not optimal_ik: 
                    best_qpos = qpos 
                    break

                else:
                    if self.manip_strategy == 'ellipsoid':
                        w,p,wp = self.check_manipulability_ellipsoid()
                    elif self.manip_strategy == 'paper':
                        w,p,wp = self.check_manipulability_paper()
                    elif self.manip_strategy == 'old':
                        w,p,wp = self.check_manipulability_old()
                    if wp > best_manip:
                        best_manip = wp 
                        best_qpos = qpos
        self._reset_internal()
        if best_qpos is None:
            return False 

        self.set_qpos_and_update(best_qpos)   
        return True

    def solve_ik(self, target_matrix, wide=False):
        """
        Given ee pose in world frame, returns a valid qpos or None. 

        @target_matrix: desired (mujoco) ee pose in world frame. 

        Two messy bits: 
        a) we need to solve IK in "panda_link0" frame, not world.  
        b) bullet URDF lacks the gripper, so we're solving for link7/link8 pose. 

        mujoco lacks link8, so we compute link7 pose, link8_in_link7, etc. 
        """
        if self.solver is None: 
            self._setup_ik()

        # update sim.data but don't step 
        self.sim.forward()

        # find link7 pose in the world
        cur_link7_pos = self.sim.data.body_xpos[self.sim.model.body_name2id("robot0_link7")]
        cur_link7_xmat = self.sim.data.body_xmat[self.sim.model.body_name2id("robot0_link7")].reshape(3,3)
        link7_in_world = np.eye(4)
        link7_in_world[:3,:3] = cur_link7_xmat
        link7_in_world[:3,-1] = cur_link7_pos
        
        # compute link7 in base frame 
        base_pos = self.robots[0].base_pos
        base_ori = self.robots[0].base_ori
        base_pose = (base_pos, base_ori)
        base_in_world = T.pose2mat(base_pose)
        link7_in_base = np.linalg.inv(base_in_world) @ link7_in_world

        # IKFlow solves for link8. compute link8 pose in base
        # note: hardcoded transform from link7 to link8 from URDF
        link8_in_link7 = np.eye(4)
        link8_in_link7[:3, -1] = [0., 0., 0.107]
        link8_in_base = link7_in_base @ link8_in_link7

        # compute relative transform from gripper to link8
        grp_pos = self.sim.data.site_xpos[self.sim.model.site_name2id("gripper0_grip_site")]
        grp_xmat = self.sim.data.site_xmat[self.sim.model.site_name2id("gripper0_grip_site")].reshape(3,3)
        grp_in_world = np.eye(4)
        grp_in_world[:3, :3] = grp_xmat
        grp_in_world[:3, -1] = grp_pos
        grp_in_base = np.linalg.inv(base_in_world) @ grp_in_world

        # compute target for link8 in base frame
        target_in_base = np.linalg.inv(base_in_world) @ target_matrix
        link8_in_gripper = np.linalg.inv(grp_in_base) @ link8_in_base
        target_link8 = target_in_base @ link8_in_gripper
        
        # solve
        # qpos = self.solver.ik(target_link8) 
        # return qpos 

        # ikflow samples 
        ee_pose_target = np.zeros(7)
        ee_pose_target[:3] = target_link8[:3,-1]
        quat = T.mat2quat(target_link8[:3, :3])
        quat = T.convert_quat(quat, to="wxyz")
        ee_pose_target[3:7] = quat

        samples, _ = self.solver.make_samples(
                ee_pose_target,
                self.samples_per_pose,
                **self.solver_kwargs
            )

        # check joint limits 
        jlim_mask = np.logical_not(list(map(self.checkJointPosition, samples)))
        
        # check tolerance
        pos_error, ang_error = get_solution_errors(self.solver.robot_model, samples, ee_pose_target)
        error_mask = pos_error < self.ik_pos_tol

        # mask 
        mask = np.logical_and(jlim_mask, error_mask)
        return samples[mask].detach().cpu().numpy(), ee_pose_target


    def check_manipulability_ellipsoid(self):
        ### Manipulability elipsoid
        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.robots[0].sim.data.get_body_jacp(self.robots[0].robot_model.eef_name).reshape((3, -1))
        Jp_joint = Jp[:, self.robots[0]._ref_joint_vel_indexes]

        Jr = self.robots[0].sim.data.get_body_jacr(self.robots[0].robot_model.eef_name).reshape((3, -1))
        Jr_joint = Jr[:, self.robots[0]._ref_joint_vel_indexes]

        J = np.concatenate((Jp,Jr),axis=0)

        JJt = np.matmul(J,J.transpose())
        Jdet = np.linalg.det(JJt)
        w = math.sqrt(Jdet)


        ########### NEW ###########
        invJJt = np.linalg.inv(JJt)
        ttt = 1 / np.dot(np.dot(self.task_mean[None], invJJt), self.task_mean[None].transpose())[0][0]
        ttt = np.sqrt(ttt)
        ### Penalization for distance to joint limits
        p = 1

        k = 1 #hyperparameter for adjust behavior near joint limits
        joint_total = 1
        for (qidx, (q, q_limits)) in enumerate(
            zip(self.robots[0].sim.data.qpos[self.robots[0]._ref_joint_pos_indexes], self.robots[0].sim.model.jnt_range[self.robots[0]._ref_joint_indexes])
        ):
            joint_total *= (q - q_limits[0])*(q_limits[1]-q)/np.square(q_limits[1]-q_limits[0])
        p -= math.exp(-k*joint_total)

        return(w,p,np.power(w,self.m_constant) * np.power(p,self.p_constant) * np.power(ttt,self.ttt_constant))


    def check_manipulability_paper(self):
        ### Manipulability paper
        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.robots[0].sim.data.get_body_jacp(self.robots[0].robot_model.eef_name).reshape((3, -1))
        Jp_joint = Jp[:, self.robots[0]._ref_joint_vel_indexes]

        Jr = self.robots[0].sim.data.get_body_jacr(self.robots[0].robot_model.eef_name).reshape((3, -1))
        Jr_joint = Jr[:, self.robots[0]._ref_joint_vel_indexes]

        J = np.concatenate((Jp,Jr),axis=0)

        JJt = np.matmul(J,J.transpose())
        Jdet = np.linalg.det(JJt)
        w = math.sqrt(Jdet)


        #### new
 
        ttt = np.dot(J.transpose(), self.task_mean[None].transpose())[:,0]

        ttt = np.sqrt(ttt.dot(ttt))
        
        #### new


        ### Penalization for distance to joint limits
        p = 1

        k = 1 #hyperparameter for adjust behavior near joint limits
        joint_total = 1
        for (qidx, (q, q_limits)) in enumerate(
            zip(self.robots[0].sim.data.qpos[self.robots[0]._ref_joint_pos_indexes], self.robots[0].sim.model.jnt_range[self.robots[0]._ref_joint_indexes])
        ):
            joint_total *= (q - q_limits[0])*(q_limits[1]-q)/np.square(q_limits[1]-q_limits[0])
        p -= math.exp(-k*joint_total)

        return(w,p,np.power(w,self.m_constant) * np.power(p,self.p_constant) * np.power(ttt,self.ttt_constant))

    def check_manipulability_old(self):
        
        ### Manipulability old
        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.robots[0].sim.data.get_body_jacp(self.robots[0].robot_model.eef_name).reshape((3, -1))
        Jp_joint = Jp[:, self.robots[0]._ref_joint_vel_indexes]

        Jr = self.robots[0].sim.data.get_body_jacr(self.robots[0].robot_model.eef_name).reshape((3, -1))
        Jr_joint = Jr[:, self.robots[0]._ref_joint_vel_indexes]

        J = np.concatenate((Jp,Jr),axis=0)

        JJt = np.matmul(J,J.transpose())
        Jdet = np.linalg.det(JJt)
        w = math.sqrt(Jdet)

        ### Penalization for distance to joint limits
        p = 1

        k = 1 #hyperparameter for adjust behavior near joint limits
        joint_total = 1
        for (qidx, (q, q_limits)) in enumerate(
            zip(self.robots[0].sim.data.qpos[self.robots[0]._ref_joint_pos_indexes], self.robots[0].sim.model.jnt_range[self.robots[0]._ref_joint_indexes])
        ):
            joint_total *= (q - q_limits[0])*(q_limits[1]-q)/np.square(q_limits[1]-q_limits[0])
        p -= math.exp(-k*joint_total)

        return(w,p,np.power(w,self.m_constant) * np.power(p,self.p_constant))

    def setGeomIDs(self):
        self.robot_geom_ids = []
        self.obj_geom_ids = []

        for n in range(self.sim.model.ngeom):
            body = self.sim.model.geom_bodyid[n]
            body_name = self.sim.model.body_id2name(body)
            geom_name = self.sim.model.geom_id2name(n)

            if geom_name == "ground" and body_name == "world":
                self.ground_geom_id = n
            elif "robot0_" in body_name or "gripper0_" in body_name:
                self.robot_geom_ids.append(n)
            elif body_name != "world":
                #print(geom_name)
                self.obj_geom_ids.append(n)

    def contactBetweenRobotAndObj(self,contact):
        if contact.geom1 in self.robot_geom_ids and contact.geom2 in self.obj_geom_ids:
            #print("Contact between {one} and {two}".format(one=contact.geom1, two=contact.geom2))
            return True
        if contact.geom2 in self.robot_geom_ids and contact.geom1 in self.obj_geom_ids:
            #print("Contact between {one} and {two}".format(one=contact.geom1, two=contact.geom2))
            return True
        return False

    def contactBetweenGripperAndSpecificObj(self,contact, name):

        if self.sim.model.geom_id2name(contact.geom1)[:8] == 'gripper0' and self.sim.model.geom_id2name(contact.geom2) == name:
            #print("Contact between {one} and {two}".format(one=self.sim.model.geom_id2name(contact.geom1), two=self.sim.model.geom_id2name(contact.geom2)))
            return True
        if self.sim.model.geom_id2name(contact.geom2)[:8] == 'gripper0' and self.sim.model.geom_id2name(contact.geom1) == name:
            #print("Contact between {one} and {two}".format(one=self.sim.model.geom_id2name(contact.geom1), two=self.sim.model.geom_id2name(contact.geom2)))
            return True
        return False

    def contactBetweenRobotAndFloor(self,contact):

        if contact.geom1 == self.ground_geom_id and contact.geom2 in self.robot_geom_ids:
            return True
        if contact.geom2 == self.ground_geom_id and contact.geom1 in self.robot_geom_ids:
            return True
        return False

    def isInvalidMJ(self):
        # Note that the contact array has more than `ncon` entries,
        # so be careful to only read the valid entries.
        for contact_index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[contact_index]
            if self.contactBetweenRobotAndObj(contact):
                return 1
            # elif self.contactBetweenRobotAndFloor(contact):
            #     return 2
        return 0

    def checkJointPosition(self, qpos):
        """
        Check if this robot is either very close or at the joint limits

        Returns:
            bool: True if this arm is near its joint limits
        """
        tolerance = 0.1
        for (qidx, (q, q_limits)) in enumerate(
            zip(qpos, self.sim.model.jnt_range[self.robots[0]._ref_joint_indexes])
        ):
            if q_limits[0] != q_limits[1] and not (q_limits[0] + tolerance < q < q_limits[1] - tolerance):
                #print("Joint limit reached in joint " + str(qidx))
                #print("Joint min is {min} and max is {max}, joint {qidx} violated with {j}".format(qidx=qidx, min=q_limits[0], max=q_limits[1], j=q))
                return True
        return False

    def get_obj_pose(self):
        obj_name = OBJECT_NAMES[self.__class__.__name__]
        obj_id = self.object_body_ids[obj_name]
        xpos = np.array(self.sim.data.body_xpos[obj_id])
        xmat = np.array(self.sim.data.body_xmat[obj_id]).reshape(3,3)
        quat = T.mat2quat(xmat)
        pose = (xpos, quat)
        return T.pose2mat(pose)

    def grasp_to_obj_frame(self, grasp_in_world):
        obj_in_world = self.get_obj_pose()
        return np.matmul(np.linalg.inv(obj_in_world), grasp_in_world)

    def grasp_to_world_frame(self, grasp_in_obj):
        obj_in_world = self.get_obj_pose()
        return np.matmul(obj_in_world, grasp_in_obj)

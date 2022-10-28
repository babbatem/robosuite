import os 
from os.path import join as pjoin
import math 

import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.utils.collision_utils import isInvalidMJ, checkJointPosition, setGeomIDs, contactBetweenGripperAndSpecificObj
from robosuite.controllers import controller_factory

GRIP_NAMES = {'DoorCIP': 'Door_handle', 'DrawerCIP': 'Drawer_handle','SlideCIP': 'Slide_grip','LeverCIP': 'Lever_lever'}

class CIP(object):
    """
    enables functionality for resetting with grasping, safety, etc. 
    construct robosuite env as class EnvName(SingleArmEnv, CIP)
    """
    def __init__(self):
        super(CIP, self).__init__()
        self.solver = None
        setGeomIDs(self)


    def check_contact(self,task_name):

        for contact_index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[contact_index]
            if contactBetweenGripperAndSpecificObj(contact, GRIP_NAMES[task_name]):
                return True

        return False

    def _setup_ik(self):

        from tracikpy import TracIKSolver
        self.robot_name = self.robots[0].name
        self.robot_urdf = pjoin(
                os.path.join(robosuite.models.assets_root, "bullet_data"),
                "{}_description/urdf/{}_arm.urdf".format(self.robot_name.lower(), self.robot_name.lower()),
            )
        self.base_link_name = "panda_link0"
        self.ee_link_name = "panda_link8"
        self.solver = TracIKSolver(
                                    self.robot_urdf,
                                    self.base_link_name,
                                    self.ee_link_name
                                )

        self.num_attempts = 1000 
        

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
        collision_score = isInvalidMJ(self)
        if collision_score != 0:
            return False

        if checkJointPosition(self, qpos):
            return False 

        return True

    def reset_to_grasp(self, grasp_pose, wide=False, optimal_ik=False):

        if self.solver is None: 
            self._setup_ik()
      
        # override initial gripper qpos for wide grasp 
        if wide:
            self.sim.data.qpos[self.robots[0]._ref_gripper_joint_pos_indexes] = [0.05, -0.05]

        qpos = None
        best_manip = -np.inf
        best_qpos = None
        for _ in range(self.num_attempts):
            qpos = self.solve_ik(grasp_pose)
            if qpos is None: 
                continue 

            # set joints 
            self.sim.data.qpos[:7] = qpos
            self.sim.forward()

            # ensure valid
            collision_score = isInvalidMJ(self)
            if collision_score != 0:
                qpos = None 
                continue 

            if checkJointPosition(self, qpos):
                qpos = None 
                continue

            # maybe keep qpos w/ highest manipulability score 
            if not optimal_ik: 
                best_qpos = qpos 
                break

            else:
                w,p,wp = self.check_manipulability()
                if wp > best_manip:
                    best_manip = wp 
                    best_qpos = qpos

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
        qpos = self.solver.ik(target_link8) 
        return qpos 

    def check_manipulability(self):
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

        ### Penalization for distance to joint limits
        p = 1

        k = 1 #hyperparameter for adjust behavior near joint limits
        joint_total = 1
        for (qidx, (q, q_limits)) in enumerate(
            zip(self.robots[0].sim.data.qpos[self.robots[0]._ref_joint_pos_indexes], self.robots[0].sim.model.jnt_range[self.robots[0]._ref_joint_indexes])
        ):
            joint_total *= (q - q_limits[0])*(q_limits[1]-q)/(q_limits[1]-q_limits[0])
        p -= math.exp(-k*joint_total)

        return(w,p,w*p)
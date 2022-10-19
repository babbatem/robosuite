def setGeomIDs(env):
    global ground_geom_id
    global robot_geom_ids
    global obj_geom_ids

    for n in range(env.sim.model.ngeom):
        body = env.sim.model.geom_bodyid[n]
        body_name = env.sim.model.body_id2name(body)
        geom_name = env.sim.model.geom_id2name(n)

        if geom_name == "ground" and body_name == "world":
            ground_geom_id = n
        elif "robot0_" in body_name or "gripper0_" in body_name:
            robot_geom_ids.append(n)
        elif body_name != "world":
            print(geom_name)
            obj_geom_ids.append(n)

def contactBetweenRobotAndObj(contact):
    global ground_geom_id
    global robot_geom_ids
    global obj_geom_ids
    if contact.geom1 in robot_geom_ids and contact.geom2 in obj_geom_ids:
        print("Contact between {one} and {two}".format(one=contact.geom1, two=contact.geom2))
        return True
    if contact.geom2 in robot_geom_ids and contact.geom1 in obj_geom_ids:
        print("Contact between {one} and {two}".format(one=contact.geom1, two=contact.geom2))
        return True
    return False

def contactBetweenGripperAndSpecificObj(contact, name):
    global ground_geom_id
    global robot_geom_ids
    global obj_geom_ids

    if env.sim.model.geom_id2name(contact.geom1)[:8] == 'gripper0' and env.sim.model.geom_id2name(contact.geom2) == name:
        print("Contact between {one} and {two}".format(one=env.sim.model.geom_id2name(contact.geom1), two=env.sim.model.geom_id2name(contact.geom2)))
        return True
    if env.sim.model.geom_id2name(contact.geom2)[:8] == 'gripper0' and env.sim.model.geom_id2name(contact.geom1) == name:
        print("Contact between {one} and {two}".format(one=env.sim.model.geom_id2name(contact.geom1), two=env.sim.model.geom_id2name(contact.geom2)))
        return True
    return False

def contactBetweenRobotAndFloor(contact):
    global ground_geom_id
    global robot_geom_ids
    global obj_geom_ids

    if contact.geom1 == ground_geom_id and contact.geom2 in robot_geom_ids:
        return True
    if contact.geom2 == ground_geom_id and contact.geom1 in robot_geom_ids:
        return True
    return False

def isInvalidMJ(env):
    # Note that the contact array has more than `ncon` entries,
    # so be careful to only read the valid entries.
    for contact_index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[contact_index]
        if contactBetweenRobotAndObj(contact):
            return 1
        elif contactBetweenRobotAndFloor(contact):
            return 2
    return 0

def checkJointPosition(env, qpos):
    """
    Check if this robot is either very close or at the joint limits

    Returns:
        bool: True if this arm is near its joint limits
    """
    tolerance = 0.1
    for (qidx, (q, q_limits)) in enumerate(
        zip(qpos, env.sim.model.jnt_range[env.robots[0]._ref_joint_indexes])
    ):
        if q_limits[0] != q_limits[1] and not (q_limits[0] + tolerance < q < q_limits[1] - tolerance):
            #print("Joint limit reached in joint " + str(qidx))
            #print("Joint min is {min} and max is {max}, joint {qidx} violated with {j}".format(qidx=qidx, min=q_limits[0], max=q_limits[1], j=q))
            return True
    return False
import numpy as np
import cvxpy as cp

def normalize(x):
    mag = np.linalg.norm(x)
    if mag == 0:
        mag = mag + 1e-10
    return x / mag


def hat(v):
    if v.shape == (3, 1) or v.shape == (3,):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    else:
        raise ValueError


def generate_contact_frame(pos, normal):
    """Generate contact frame, whose z-axis aligns with the normal direction (inward to the object)
    """
    up = normalize(np.random.rand(3))
    z = normalize(normal)
    # print(f"up:{up}, z:{z}")
    up_z = np.cross(up, z)
    x = normalize(np.cross(up, z))
    y = normalize(np.cross(z, x))

    result = np.eye(4)
    result[0:3, 0] = x
    result[0:3, 1] = y
    result[0:3, 2] = z
    result[0:3, 3] = pos
    return result


def adj_T(frame):
    """Compute the adjoint matrix for the contact frame
    """
    assert frame.shape[0] == frame.shape[1] == 4, 'Frame needs to be 4x4'

    R = frame[0:3, 0:3]
    p = frame[0:3, 3]
    result = np.zeros((6, 6))
    result[0:3, 0:3] = R
    result[3:6, 0:3] = hat(p) @ R
    result[3:6, 3:6] = R
    return result


def compute_grasp_map(contact_pos, contact_normal, soft_contact=False):
    """ Computes the grasp map for all contact points.
    Check chapter 5 of http://www.cse.lehigh.edu/~trink/Courses/RoboticsII/reading/murray-li-sastry-94-complete.pdf for details.
    Args:
        contact_pos: location of contact in the object frame
        contact_normal: surface normals at the contact location, point inward !!!, N x 3, in the object frame
        soft_contact: whether use soft contact model. Defaults to False.
    Returns:
        G: grasp map for the contacts
    """
    n_point = len(contact_pos)

    # Compute the contact basis B
    if soft_contact:
        B = np.zeros((6, 4))
        B[0:3, 0:3] = np.eye(3)
        B[5, 3] = 1
    else:  # use point contact w/ friction
        B = np.zeros((6, 3))
        B[0:3, 0:3] = np.eye(3)

    # Compute the contact frames, adjoint matrix, and grasp map
    contact_frames = []
    grasp_maps = []
    for pos, normal in zip(contact_pos, contact_normal):
        contact_frame = generate_contact_frame(pos, normal)
        contact_frames.append(contact_frame)

        adj_matrix = adj_T(contact_frame)
        grasp_map = adj_matrix @ B
        grasp_maps.append(grasp_map)

    G = np.hstack(grasp_maps)
    assert G.shape == (6, n_point * B.shape[1]), 'Grasp map shape does not match'

    return G, contact_frames

def fc_optimization(contact_positions, contact_normals, weight, friction_coeff=0.3, soft_contact=False, orientation=0):
    
    if orientation == 0:
        w_ext = np.array([0.0, 0.0, -weight, 0.0, 0.0, 0.0])  # External wrench (gravity)
    elif orientation == 1:
        w_ext = np.array([0.0, -weight, 0.0, 0.0, 0.0, 0.0])  # External wrench (gravity)
    else:
        w_ext = np.array([-weight, 0.0, 0.0, 0.0, 0.0, 0.0])  # External wrench (gravity)
    
    f1 = cp.Variable(4) if soft_contact else cp.Variable(3)
    f2 = cp.Variable(4) if soft_contact else cp.Variable(3)
    f3 = cp.Variable(4) if soft_contact else cp.Variable(3)
    f4 = cp.Variable(4) if soft_contact else cp.Variable(3)
    
    G1, CF1 = compute_grasp_map(contact_pos=[contact_positions[0]], contact_normal=[contact_normals[0]], soft_contact=soft_contact)
    G2, CF2 = compute_grasp_map(contact_pos=[contact_positions[1]], contact_normal=[contact_normals[1]], soft_contact=soft_contact)
    G3, CF3 = compute_grasp_map(contact_pos=[contact_positions[2]], contact_normal=[contact_normals[2]], soft_contact=soft_contact)
    G4, CF4 = compute_grasp_map(contact_pos=[contact_positions[3]], contact_normal=[contact_normals[3]], soft_contact=soft_contact)
    G = np.concatenate([G1, G2, G3, G4], axis=1)
    
    # grasp matrix not having rank 6 are directly discarded
    if np.linalg.matrix_rank(G) != 6:
        return None, None, None, None, 1000, None
    
    f_low = 0.1
    f_high = 60
        
    constraints = [
        cp.norm(f1) <= f_high,
        cp.norm(f2) <= f_high,
        cp.norm(f3) <= f_high,
        cp.norm(f4) <= f_high,
        # cp.norm(f1) >= f_low,
        # cp.norm(f2) >= f_low,
        # cp.norm(f3) >= f_low,
        # cp.norm(f4) >= f_low,
        
        cp.SOC(friction_coeff * f1[2], f1[0:2]),
        cp.SOC(friction_coeff * f2[2], f2[0:2]),
        cp.SOC(friction_coeff * f3[2], f3[0:2]),
        cp.SOC(friction_coeff * f4[2], f4[0:2]),
    ]
    
    problem = cp.Problem(cp.Minimize(cp.norm(G1 @ f1 + G2 @ f2 + G3 @ f3 + G4 @ f4 + w_ext)),
                         constraints)
    
    try:
        problem.solve()
    except Exception as e:
        # Optimization failed to solve. Return zeros and high loss value (1000) to remove these outliers. 
        # Observed about < 10 grasps fail to be solved. Safe to remove them. 
        print(e)
        return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 1000, [CF1, CF2, CF3, CF4]
    
    return CF1[0][:3,:3] @ f1.value, CF2[0][:3,:3] @ f2.value, CF3[0][:3,:3] @ f3.value, CF4[0][:3,:3] @ f4.value, problem.value, [CF1, CF2, CF3, CF4]
     
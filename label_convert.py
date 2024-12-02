import numpy as np
from glob import glob
import os
import mano
import matplotlib.pyplot as plt
from grabnet.tools.utils import showHandJoints, project_3D_points
import torch
from grabnet.tools.utils import aa2rotmat, rotmat2aa
from arctic_data_utils import get_obj_info, construct_obj_tensors
from psbody.mesh import Mesh
from tqdm import tqdm

hand_bone = [[0, 1], [1, 2], [2, 3], [3, 4],
             [0, 5], [5, 6], [6, 7], [7, 8],
             [0, 9], [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15], [15, 16],
             [0, 17], [17, 18], [18, 19], [19, 20]]

jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]

hand_mean = np.array(
            [0.1116787156904433,-0.04289217484104638,0.41644183681244606,0.10881132513711185,0.0659856788782267,
             0.7562200100001023,-0.09639296514009964,0.09091565922041477,0.18845929069701614,-0.11809503934386674,
             -0.05094385260705537,0.529584499976084,-0.14369840869244885,-0.05524170001527845,0.7048571406917286,
             -0.01918291683986582,0.09233684821597642,0.3379135244947721,-0.4570329833266365,0.19628394516488204,
             0.6254575328442732,-0.21465237881086027,0.06599828649166892,0.5068942070038754,-0.36972435736649994,
             0.06034462636097784,0.07949022787634759,-0.14186969453172144,0.08585263331718808,0.6355282566897771,
             -0.3033415874850632,0.05788097522832922,0.6313892099233043,-0.17612088501838064,0.13209307627166406,
             0.37335457646458126,0.8509642789706306,-0.2769227420650918,0.09154806978240378,-0.49983943762160615,
             -0.026556472160458842,-0.05288087673273012,0.5355591477841195,-0.0459610409551833,0.2773580212595623])

work_dir = 'output/'
label_dir = os.path.join(work_dir, 'label')
rh_model = mano.load(model_path='/home/user/q_16T_2024/cj/mano_v1_2/models/MANO_RIGHT.pkl',
                    model_type='mano',
                    num_pca_comps=45,
                    batch_size=1,
                    flat_hand_mean=False)

def plot_hand_joint2d(rgb_image, joint2d):
    plt.imshow(rgb_image)
    for i in range(joint2d.shape[0]):
        plt.plot(joint2d[i, 0], joint2d[i, 1], 'ro')
    for bone in hand_bone:
        plt.plot(joint2d[bone, 0], joint2d[bone, 1], 'r')
    plt.show()

def plot_object_joint2d(rgb_image, joint2d):
    plt.imshow(rgb_image)
    for i in range(joint2d.shape[0]):
        plt.plot(joint2d[i, 0], joint2d[i, 1], 'ro')
    plt.show()

def aa2rotmat_numpy(aa):
    rotmat = aa2rotmat(torch.tensor(aa)).detach().cpu().numpy()
    return rotmat

def rotmat2aa_numpy(rotmat):
    aa = rotmat2aa(torch.tensor(rotmat)).detach().cpu().numpy()
    return aa

if __name__ == '__main__':
    os.makedirs(os.path.join(work_dir, 'label_convert'), exist_ok=True)
    for label_path in tqdm(sorted(glob(os.path.join(label_dir, '*.npz')))):
        # print(label_path)
        # rgb_path = label_path.replace('label', 'rgb').replace('npz', 'jpg')
        # rgb_image = plt.imread(rgb_path)
        
        obj_name = os.path.basename(label_path).split('_')[0]
        label = np.load(label_path)
        joint3d = label["joint3d"]
        handparam = label["handparam"]
        trans = label["trans"]
        obj_pose = label["obj_pose"]
        obj_trans = label["obj_trans"]
        intrinsics = label["intrinsics"]
        extrinsics = label["extrinsics"]
        # print(obj_name)
        # print("joint3d:", joint3d.shape)
        # print("handparam:", handparam.shape)
        # print("trans:", trans.shape)
        # print("obj_pose:", obj_pose.shape)
        # print("obj_trans:", obj_trans.shape)
        # print("intrinsics:", intrinsics.shape)
        # print("extrinsics:", extrinsics.shape)
        
        joint3d = joint3d[jointsMapManoToSimple]
        joint3d_cam = (joint3d + obj_trans) @ extrinsics[:3, :3].T + extrinsics[:3, 3]
        joint2d = (joint3d_cam * 1000) @ intrinsics.T
        joint2d = joint2d[:, :2] / joint2d[:, 2:]
        # plot_hand_joint2d(rgb_image, joint2d)
        
        hand_global_orient = handparam[:, :3]
        hand_pose = handparam[:, 3:48]
        hand_pose -= hand_mean
        hand_rotmat = aa2rotmat_numpy(hand_global_orient).reshape(-1, 3, 3)
        hand_rotmat_cam = extrinsics[:3, :3] @ hand_rotmat
        parm3d_r = np.array((0.0957, 0.0064, 0.0062))
        cur_palm_r = trans + parm3d_r
        trans_cam = (trans + parm3d_r + obj_trans) @ extrinsics[:3, :3].T + extrinsics[:3, 3] - parm3d_r
        hand_cam_orient = rotmat2aa_numpy(hand_rotmat_cam).reshape(-1, 3)
        
        # with torch.no_grad():
        #     mano_output = rh_model(global_orient=torch.from_numpy(hand_cam_orient.astype(np.float32)), 
        #                         hand_pose=torch.from_numpy(hand_pose.astype(np.float32)), 
        #                         transl=torch.from_numpy(trans_cam.astype(np.float32)), return_verts=True, return_tips=True)
        # mano_joint3d = mano_output.joints.numpy()[0, jointsMapManoToSimple]
        # mano_joint2d = (mano_joint3d * 1000) @ intrinsics.T
        # mano_joint2d = mano_joint2d[:, :2] / mano_joint2d[:, 2:]
        # plot_hand_joint2d(rgb_image, mano_joint2d)
        
        targets = {}
        targets["mano.pose.r"] = np.concatenate([hand_cam_orient, hand_pose], axis=-1)
        targets["mano.pose.l"] = np.concatenate([hand_cam_orient, hand_pose], axis=-1)
        targets["mano.beta.r"] = rh_model.betas.cpu().detach().numpy()
        targets["mano.beta.l"] = rh_model.betas.cpu().detach().numpy()
        targets["mano.j2d.norm.r"] = joint2d
        targets["mano.j2d.norm.l"] = joint2d
        
        # object
        obj_tensor = construct_obj_tensors(obj_name)
        object_fullpts = obj_tensor['v'].numpy()[0]
        maximum = object_fullpts.max(0)
        minimum = object_fullpts.min(0)
        offset = ( maximum + minimum) / 2
        kp3d = get_obj_info(obj_name)
        # obj_mesh = Mesh(v=obj_tensor['v'].numpy()[0], f=obj_tensor['f'].numpy()[0])
        # obj_mesh.write_obj('obj_mesh.obj')
        # with open("kp3d.obj", "w") as f:
        #     for i in range(kp3d.shape[0]):
        #         f.write(f"v {kp3d[i, 0]} {kp3d[i, 1]} {kp3d[i, 2]}\n")
        
        # kp3d -= offset
        # kp3d = kp3d @ obj_pose.T + obj_trans
        # kp3d_cam = (kp3d) @ extrinsics[:3, :3].T + extrinsics[:3, 3]
        
        obj_pose_cam = extrinsics[:3, :3] @ obj_pose
        obj_trans_cam = -offset @ (extrinsics[:3, :3] @ obj_pose).T + obj_trans @ extrinsics[:3, :3].T + extrinsics[:3, 3]
        kp3d_cam = kp3d @ obj_pose_cam.T + obj_trans_cam
        
        kp2d = (kp3d_cam*1000) @ intrinsics.T
        kp2d = kp2d[:, :2] / kp2d[:, 2:]
        # plot_object_joint2d(rgb_image, kp2d)
        
        targets["object.kp3d.full.b"] = kp3d_cam[16:, :3]
        targets["object.kp2d.norm.b"] = kp2d[16:, :2]
        targets["object.kp3d.full.t"] = kp3d_cam[:16, :3]
        targets["object.kp2d.norm.t"] = kp2d[:16, :2]

        targets["object.bbox3d.full.b"] = np.zeros((8,3))
        targets["object.bbox2d.norm.b"] = np.zeros((8,2))
        targets["object.bbox3d.full.t"] = np.zeros((8,3))
        targets["object.bbox2d.norm.t"] = np.zeros((8,2))
        targets["object.radian"] = np.array(0)

        targets["object.kp2d.norm"] = kp2d[:, :2]
        targets["object.bbox2d.norm"] = np.zeros((16,2))
        targets['trans_obj'] = obj_trans_cam
        # print(obj_pose_cam.shape)
        targets['rot_obj'] = rotmat2aa_numpy(obj_pose_cam[None, ...]).reshape(3)
        # print(targets['rot_obj'].shape)
        
        targets["mano.j3d.full.r"] = joint3d_cam[:, :3]
        targets["mano.j3d.full.l"] = joint3d_cam[:, :3]

        is_valid = True
        left_valid = False
        right_valid = True
        targets["is_valid"] = float(is_valid)
        targets["left_valid"] = float(left_valid) * float(is_valid)
        targets["right_valid"] = float(right_valid) * float(is_valid)
        targets["joints_valid_r"] = np.ones(21) * targets["right_valid"]
        targets["joints_valid_l"] = np.ones(21) * targets["left_valid"]

        targets["intrinsics"] = intrinsics
        
        label_convert_path = label_path.replace('label', 'label_convert').replace('npz', 'npy')
        np.save(label_convert_path, targets)

        # break
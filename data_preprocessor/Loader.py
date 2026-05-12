import torch
import numpy as np
import pandas as pd
import os
import glob
from PIL import Image
import natsort  # pip install natsort

from data_utils.figs import image_features_extractor
from data_utils.PCDtoBEV import PCDtoBEV, model_loader
from data_utils.PHY_toolbox import DFT_codebook_3D, channel_eff, cir_to_ofdm_channel, subcarrier_frequencies

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Configuration Parameters ---
BASE_DATA_DIR = r'./'
output_dir = r'/data/test_lidar/foggyy'  # Change this to your desired output directory
os.makedirs(output_dir, exist_ok=True)
lidar_dir = os.path.join(output_dir, 'lidar')
os.makedirs(lidar_dir, exist_ok=True)
vision_dir = os.path.join(output_dir, 'vision')
os.makedirs(vision_dir, exist_ok=True)

# --- Antenna and Beam Parameters ---
H_bs_ant_shape = (1, 64)
bs_ant_shape = (1, 64)
bs_beam_shape = (1, 64)
arr_len = bs_beam_shape[0] * bs_beam_shape[1]

H_ue_ant_shape = (1, 16)
ue_ant_shape = (1, 16)
ue_beam_shape = (1, 64)
ue_arr_len = ue_beam_shape[0] * ue_beam_shape[1]

bs_beam_sizes = [64, 128, 256]
bs_codebooks = {s: DFT_codebook_3D(bs_ant_shape, (1, s)) for s in bs_beam_sizes}
ue_codebook = DFT_codebook_3D(ue_ant_shape, ue_beam_shape)

# Sequencing parameters
num_neighbor = 2
num_carrier = 1024
subcarrier_spacing = 120e3
fc = 28e9
frequencies = subcarrier_frequencies(num_carrier, subcarrier_spacing)

cfg_file = './data_utils/config.yaml'
ckpt_file = './data_utils/checkpoint.pth'
model, config = model_loader(cfg_file, ckpt_file, device)

file_cnt = 0
scene_cnt = 0
frame_cnt = 0

# get Town folders
all_items_in_base = os.listdir(BASE_DATA_DIR)
town_folder_names = natsort.natsorted([
    item for item in all_items_in_base
    if os.path.isdir(os.path.join(BASE_DATA_DIR, item)) and \
       item != os.path.basename(output_dir)
])

# Iterate over each Town folder
for town_name in town_folder_names:
    current_town_path = os.path.join(BASE_DATA_DIR, town_name)
    print(f"\n--- Processing Town: {town_name} ---------")

    # get Scene folders
    scene_folders = natsort.natsorted([
        d for d in os.listdir(current_town_path)
        if os.path.isdir(os.path.join(current_town_path, d))
    ])

    for scene_name in scene_folders:
        scene_cnt += 1
        current_scene_path = os.path.join(current_town_path, scene_name)
        current_output_scene_path = os.path.join(output_dir)
        os.makedirs(current_output_scene_path, exist_ok=True)
        print(f"\n------ processing scene: {scene_name} ------")

        # get cav_X folders
        cav_folders = natsort.natsorted([
            os.path.join(current_scene_path, d)
            for d in os.listdir(current_scene_path)
            if os.path.isdir(os.path.join(current_scene_path, d)) and d.startswith('cav_')
        ])

        for cav_folder_path in cav_folders:
            file_cnt += 1
            cav_name = os.path.basename(cav_folder_path)

            # get .npz files
            npz_files = natsort.natsorted(glob.glob(os.path.join(cav_folder_path, '*.pcd')))
            seq_len = len(npz_files)

            seq_beam_powers_dict = {s: [] for s in bs_beam_sizes}
            seq_h_list = [] # Store projected channel
            seq_h_eff_list = [] # Store raw H_eff for later projection
            phi_temp = []
            theta_temp = []

            for file_idx, npz_file_path in enumerate(npz_files):
                paths_data = np.load(npz_file_path)
                a = paths_data['a']
                tau = paths_data['tau']
                H = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
                H = np.squeeze(H)
                H_eff = channel_eff(H, H_ue_ant_shape, H_bs_ant_shape, ue_ant_shape, bs_ant_shape)
                H_eff = np.sum(H_eff, axis=2) / num_carrier
                
                seq_h_eff_list.append(H_eff) # Store H_eff (1, 16)

                for size in bs_beam_sizes:
                    bs_cb = bs_codebooks[size]
                    received_signal_on_beams = np.matmul(ue_codebook.conj().T, np.matmul(H_eff, bs_cb))
                    MM_chs = np.abs(received_signal_on_beams).squeeze()
                    if ue_arr_len == 1:
                        MM_chs = np.expand_dims(MM_chs, axis=0)
                    seq_beam_powers_dict[size].append(MM_chs)

                if len(paths_data['phi_t']) > 0:
                    phi_temp.append(paths_data['phi_t'][0, 0, 0, 0])
                    theta_temp.append(paths_data['theta_t'][0, 0, 0, 0])

                # -- Handle vision data for this frame ---
                vision_path = npz_file_path.replace('.pcd', '_camera0.png')
                image_features = image_features_extractor(
                    image_path=vision_path,
                    patch_size=16,
                    embedding_dim=768)
                data_vision_numpy = image_features.detach().numpy()
                vision_filename = os.path.join(vision_dir, f'{frame_cnt}.npy')
                np.save(vision_filename, data_vision_numpy)
                # -- Handle LiDAR data for this frame ---
                lidar_path = vision_path.replace('_camera0.png', '.pcd')
                fig_filename = os.path.join(lidar_dir, f'{frame_cnt}.png')
                data_lidar = PCDtoBEV(model, config, lidar_path, device, visualize=False, save_path=fig_filename)
                data_lidar_numpy = data_lidar.cpu().numpy()
                lidar_filename = os.path.join(lidar_dir, f'{frame_cnt}.npy')
                np.save(lidar_filename, data_lidar_numpy)

                frame_cnt += 1

            # --- handle beam, gain, phi, theta for each size ---
            for size in bs_beam_sizes:
                current_seq_beam_powers = seq_beam_powers_dict[size]
                current_arr_len = size
                
                seq_beam_label = np.zeros((seq_len, 1))
                seq_true_label = np.zeros((seq_len, 1))  # Ground Truth (Global Optimal)
                seq_normal_gain = np.zeros((seq_len, current_arr_len))
                seq_gain = np.zeros((seq_len, current_arr_len))
                seq_beam_snr = np.zeros((seq_len, 1 + 2 * num_neighbor))
                seq_h = np.zeros((seq_len, bs_ant_shape[0]*bs_ant_shape[1]), dtype=complex)

                # optimal tx beam index of last time slot
                last_optimal_bs_beam = 0

                for t_step in range(seq_len):
                    Mm_chs = current_seq_beam_powers[t_step]

                    # --- 1. Calculate Ground Truth (Global Optimal) ---
                    global_best_idx = np.unravel_index(np.argmax(Mm_chs), Mm_chs.shape)
                    seq_true_label[t_step, :] = global_best_idx[1] # BS beam index

                    # --- 2. Calculate Tracked Label (Local Search / Simulation) ---
                    if t_step == 0:
                        # Global search for initialization
                        candidate_index = global_best_idx # Use the global best for init
                        candidate_ue_beam = candidate_index[0]
                        candidate_bs_beam = candidate_index[1]
                        current_optimal_bs = candidate_bs_beam
                        
                        # For t=0, we use the found beam as center for SNR
                        center_beam = current_optimal_bs
                    else:
                        # Local Search (Tracking)
                        neighbor_indices = []
                        for offset in range(-num_neighbor, num_neighbor + 1):
                            neighbor_indices.append((last_optimal_bs_beam + offset) % current_arr_len)
                        
                        mm_chs_neighbors = Mm_chs[:, neighbor_indices]
                        
                        local_best_idx = np.unravel_index(np.argmax(mm_chs_neighbors), mm_chs_neighbors.shape)
                        candidate_ue_beam = local_best_idx[0]
                        best_neighbor_idx = local_best_idx[1]
                        
                        current_optimal_bs = neighbor_indices[best_neighbor_idx]
                        center_beam = last_optimal_bs_beam

                    seq_beam_label[t_step, :] = current_optimal_bs

                    final_ue_beam = candidate_ue_beam
                    mm_chs = Mm_chs[final_ue_beam, :]
                    
                    # Calculate h_projected using the selected UE beam
                    if size == 64:
                        # Get H_eff for this time step
                        H_eff_t = seq_h_eff_list[t_step] # (1, 16)
                        
                        # Get UE beam vector from codebook
                        # ue_codebook shape: (16, 64)
                        ue_vec = ue_codebook[:, final_ue_beam] # (16,)
                        ue_vec = ue_vec.reshape(16, 1) # (16, 1)
                
                        
                        h_proj = np.matmul(ue_vec.conj().T, H_eff_t) # (1, 16)
                        seq_h_list.append(h_proj.squeeze())

                    values = []
                    for left in range(num_neighbor, 0, -1):
                        values.append(mm_chs[(center_beam - left) % current_arr_len])
                    values.append(mm_chs[center_beam])
                    for right in range(1, num_neighbor + 1):
                        values.append(mm_chs[(center_beam + right) % current_arr_len])
                    seq_beam_snr[t_step, :] = np.array(values)

                    mm_chs_2 = np.power(mm_chs, 2)
                    mm_chs_max = np.max(mm_chs_2)
                    seq_normal_gain[t_step, :] = mm_chs_2 / mm_chs_max
                    seq_gain[t_step, :] = 10 * np.log10(mm_chs_2)
                    
                    last_optimal_bs_beam = current_optimal_bs

                # Save files
                if size == 64:
                    save_dir = current_output_scene_path
                    suffix = ""
                else:
                    save_dir = os.path.join(current_output_scene_path, str(size))
                    suffix = f"_{size}"
                
                os.makedirs(save_dir, exist_ok=True)
                
                # save beam true label (Ground Truth)
                beam_true_label_filename = os.path.join(save_dir, f'beam_label{suffix}_{file_cnt}.csv')
                pd.DataFrame(seq_true_label.T).to_csv(beam_true_label_filename, index=False, header=False)
                
                if file_cnt > 45:
                    normal_gain_flatten = seq_normal_gain.flatten()
                    output_normal_gain = normal_gain_flatten.reshape(1, current_arr_len * seq_len)
                    normal_gain_filename = os.path.join(save_dir, f'normal_gain{suffix}_{file_cnt}.csv')
                    pd.DataFrame(output_normal_gain).to_csv(normal_gain_filename, index=False, header=False)

                    gain_flatten = seq_gain.flatten()
                    output_gain = gain_flatten.reshape(1, current_arr_len * seq_len)
                    gain_filename = os.path.join(save_dir, f'gain{suffix}_{file_cnt}.csv')
                    pd.DataFrame(output_gain).to_csv(gain_filename, index=False, header=False)

                # Save common data only once (for default size 64)
                if size == 64:
                    beam_snr_filename = os.path.join(current_output_scene_path, f'beam_snr_{file_cnt}.csv')
                    pd.DataFrame(seq_beam_snr.T).to_csv(beam_snr_filename, index=False, header=False)
                    
                    # Save projected channel H as .npy for complex number support
                    # seq_h_list is list of (16,) arrays. Stack to (seq_len, 16).
                    seq_h_arr = np.stack(seq_h_list)
                    h_filename = os.path.join(current_output_scene_path, f'beam_h_{file_cnt}.npy')
                    np.save(h_filename, seq_h_arr)
                    
                    phi_filename = os.path.join(current_output_scene_path, f'beam_phi_{file_cnt}.csv')
                    theta_filename = os.path.join(current_output_scene_path, f'beam_theta_{file_cnt}.csv')
                    pd.DataFrame(np.array(phi_temp).reshape(1, -1)).to_csv(phi_filename, index=False, header=False)
                    pd.DataFrame(np.array(theta_temp).reshape(1, -1)).to_csv(theta_filename, index=False, header=False)

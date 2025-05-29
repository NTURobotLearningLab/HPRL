import torch
import h5py
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import traceback
import re
from pretrain.models_option_new_vae import ActionBehaviorEncoder, ProgramEncoder
from karel_env.dsl import get_DSL_option_v2
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn


def load_encoders():
    """
    Load the pre-trained behavior and program encoders from saved model weights.
    
    Returns:
        tuple: (behavior_encoder, program_encoder) - The initialized and loaded encoder models
    """
    # Load the trained weights
    params_list = torch.load('best_valid_params.ptp', map_location=torch.device('cpu'))
    param_dict = params_list[0]

    # Create config for behavior encoder
    config = {
        'recurrent_policy': True,
        'dsl': {'num_agent_actions': 6},
        'num_lstm_cell_units': 64,
        'net': {
            'rnn_type': 'GRU',
            'dropout': 0.0,
            'use_linear': True,
            'num_rnn_encoder_units': 256
        },
        'input_channel': 8,
        'input_height': 8,
        'input_width': 8,
        'fuse_s_0': True  # Add fuse_s_0 parameter
    }

    # Initialize behavior encoder
    behavior_encoder = ActionBehaviorEncoder(
        recurrent=config['recurrent_policy'],
        num_actions=config['dsl']['num_agent_actions'],
        hidden_size=config['num_lstm_cell_units'],
        rnn_type=config['net']['rnn_type'],
        dropout=config['net']['dropout'],
        use_linear=config['net']['use_linear'],
        unit_size=config['net']['num_rnn_encoder_units'],
        **config  # Pass all config parameters as kwargs
    )

    # Load behavior encoder weights
    behavior_encoder_state_dict = {}
    for key, value in param_dict.items():
        if key.startswith('vae.behavior_encoder'):
            new_key = key.replace('vae.behavior_encoder.', '')
            behavior_encoder_state_dict[new_key] = value

    behavior_encoder.load_state_dict(behavior_encoder_state_dict)
    behavior_encoder.eval()

    # Initialize program encoder
    program_encoder = ProgramEncoder(
        num_inputs=35,  # token space size
        num_outputs=64,  # latent space size
        recurrent=True,
        hidden_size=config['num_lstm_cell_units'],
        rnn_type=config['net']['rnn_type'],
        dropout=config['net']['dropout'],
        use_linear=config['net']['use_linear'],
        unit_size=config['net']['num_rnn_encoder_units']
    )

    # Load program encoder weights
    program_encoder_state_dict = {}
    for key, value in param_dict.items():
        if key.startswith('vae.program_encoder'):
            new_key = key.replace('vae.program_encoder.', '')
            program_encoder_state_dict[new_key] = value

    program_encoder.load_state_dict(program_encoder_state_dict)
    program_encoder.eval()

    return behavior_encoder, program_encoder

def get_exec_data(hdf5_file, program_id, num_agent_actions):
    """
    Extract execution data from the HDF5 file for a given program ID.
    
    Args:
        hdf5_file: Open HDF5 file containing program execution data
        program_id: ID of the program to extract data for
        num_agent_actions: Number of possible actions in the DSL
    
    Returns:
        tuple: (s_h, s_h_len, a_h, a_h_len) - Numpy arrays of execution data
    """
    try:
        s_h = np.moveaxis(np.copy(hdf5_file[program_id]['s_h']), [-1, -2, -3], [-3, -1, -2])
        a_h = np.copy(hdf5_file[program_id]['a_h'])
        s_h_len = np.copy(hdf5_file[program_id]['s_h_len'])
        a_h_len = np.copy(hdf5_file[program_id]['a_h_len'])

        # expand demo length if needed (only for 1-timestep demos)
        if s_h.shape[1] == 1:
            s_h = np.concatenate((s_h, s_h), axis=1)
            a_h = np.ones((s_h.shape[0], 1))

        # Add dummy actions for demos with a_h_len == 0
        for i in range(s_h_len.shape[0]):
            if a_h_len[i] == 0:
                assert s_h_len[i] == 1
                a_h_len[i] += 1
                s_h_len[i] += 1
                s_h[i][1] = s_h[i][0]
                a_h[i][0] = num_agent_actions - 1

        return s_h, s_h_len, a_h, a_h_len
        
    except Exception as e:
        print(f"Error in get_exec_data for {program_id}: {e}")
        traceback.print_exc()
        return np.array([]), np.array([]), np.array([]), np.array([])

def process_hdf5_file(file_path, behavior_encoder):
    """
    Process an HDF5 file and extract program IDs and their execution data.
    
    Args:
        file_path: Path to the HDF5 file
        behavior_encoder: The behavior encoder model
        
    Returns:
        list: List of tuples (program_id, states, actions, action_lengths)
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # Get all program IDs
            programs = []

            # Filter out metadata keys like 'data_info_XXXX'
            sample_keys = [k for k in f.keys() if not k.startswith('data_info_')]

            if not sample_keys:
                print(f"No valid program data found in {file_path}")
                return []

            for program_id in sample_keys:
                try:
                    # Extract execution data
                    s_h, s_h_len, a_h, a_h_len = get_exec_data(f, program_id, behavior_encoder.num_actions)
                    if len(s_h) > 0 and len(a_h) > 0:
                        programs.append((program_id, s_h, s_h_len, a_h, a_h_len))
                except Exception as e:
                    print(f"Error processing program {program_id}: {e}")
                    traceback.print_exc()
            return programs
    except Exception as e:
        print(f"Error opening file {file_path}: {e}")
        traceback.print_exc()
        return []

def encode_demos(programs, behavior_encoder):
    """
    Encode program demonstrations using the behavior encoder.
    
    Args:
        programs: List of tuples (program_id, states, actions, action_lengths)
        behavior_encoder: The behavior encoder model
        
    Returns:
        tuple: (latent_vectors, program_ids) - Encoded vectors and corresponding program IDs
    """
    latent_vectors = []
    program_ids = []
    max_demo_length = 100  # Should match training config

    for program_id, s_h, s_h_len, a_h, a_h_len in programs:
        try:
            # Convert to torch tensors with correct shapes
            s_h_tensor = torch.FloatTensor(s_h).unsqueeze(0)  # (B=1, R=num_demos, 1, C=8, H=8, W=8) where B=batch, R=rollouts
            a_h_tensor = torch.LongTensor(a_h).unsqueeze(0)   # (B=1, R=num_demos, T) where T=sequence length
            s_h_len_tensor = torch.LongTensor(s_h_len).unsqueeze(0)  # (B=1, R=num_demos) containing state sequence lengths
            a_h_len_tensor = torch.LongTensor(a_h_len).unsqueeze(0)  # (B=1, R=num_demos) containing action sequence lengths

            # Encode to latent space
            with torch.no_grad():
                latent = behavior_encoder(s_h_tensor, a_h_tensor, s_h_len_tensor, a_h_len_tensor)
                latent_vectors.append(latent.cpu().numpy())
                program_ids.append(program_id)

        except Exception as e:
            print(f"Error encoding program {program_id}: {e}")
            traceback.print_exc()

    return np.array(latent_vectors), program_ids

def load_program_from_txt(id_file_path, dsl):
    """
    Load program IDs and their code from a text file.
    
    Args:
        id_file_path: Path to the text file containing program IDs and code
        dsl: The DSL object for parsing programs
        
    Returns:
        list: List of tuples (program_id, program_code)
    """
    program_data = []

    try:
        with open(id_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    program_id = parts[0]
                    # If the line contains the program code
                    if len(parts) > 1:
                        program_code = ' '.join(parts[1:])
                    else:
                        program_code = None
                    program_data.append((program_id, program_code))
    except Exception as e:
        print(f"Error loading programs from {id_file_path}: {e}")
        traceback.print_exc()

    return program_data

def encode_programs(program_data, program_encoder, dsl):
    """
    Encode programs using the program encoder.
    
    Args:
        program_data: List of tuples (program_id, program_code)
        program_encoder: The program encoder model
        dsl: The DSL object for parsing programs
        
    Returns:
        tuple: (latent_vectors, program_ids) - Encoded vectors and corresponding program IDs
    """
    latent_vectors = []
    program_ids = []

    # Add tracking variables
    total_input_programs = len(program_data)
    programs_with_encoding = 0

    print(f"Starting to encode {total_input_programs} programs from text file")

    for program_id, program_code in program_data:
        if program_code is None:
            continue

        # Convert program to token indices
        try:
            program_tokens = dsl.str2intseq(program_code)
        except:
            # Try to extract program from the line
            match = re.search(r'DEF run m\(\) \{(.*?)\}', program_code)
            if match:
                program_code = f"DEF run m() {{{match.group(1)}}}"
                try:
                    program_tokens = dsl.str2intseq(program_code)
                except:
                    print(f"Could not parse program: {program_code}")
                    continue
            else:
                print(f"Could not extract program from: {program_code}")
                continue

        # Convert to tensor
        tokens_tensor = torch.LongTensor(program_tokens).unsqueeze(0)  # Add batch dimension
        src_len = torch.LongTensor([len(program_tokens)])
        
        def _sample_latent(h_enc, hidden_size):
            """
            Return the latent normal sample z ~ N(mu, sigma^2)
            """
            _enc_mu = torch.nn.Linear(hidden_size, hidden_size)
            _enc_log_sigma = torch.nn.Linear(hidden_size, hidden_size)
            
            mu = _enc_mu(h_enc)
            log_sigma = _enc_log_sigma(h_enc)
            sigma = torch.exp(log_sigma)
            std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).to(torch.float).to(h_enc.device)
            
            return mu + sigma * Variable(std_z, requires_grad=False) 

        # Encode program
        with torch.no_grad():
            # Get the output from the encoder
            _, encoder_output = program_encoder(tokens_tensor, src_len)
            
            z = _sample_latent(encoder_output.squeeze(), 64)
            
            latent_vectors.append(z.cpu().numpy())
            program_ids.append(program_id)


    # Print encoding statistics
    print(f"\nENCODING STATISTICS:")
    print(f"Input programs: {total_input_programs}")
    print(f"Programs successfully encoded: {programs_with_encoding}")
    print(f"Total encodings generated: {len(latent_vectors)}")


    # Convert list of arrays to single numpy array
    if latent_vectors:
        # Check if all vectors have the same shape
        shapes = [lv.shape for lv in latent_vectors]
        if len(set(shapes)) > 1:
            print(f"Warning: Different shapes in latent vectors: {set(shapes)}")
            # Find common dimension or pad/truncate as needed
            min_dim = min([lv.shape[1] for lv in latent_vectors])
            latent_vectors = [lv[:, :min_dim] for lv in latent_vectors]

        return np.vstack(latent_vectors), program_ids
    else:
        return np.array([]), []

def main():
    """
    Main function that orchestrates the encoding and visualization process.
    
    This function:
    1. Loads the encoders
    2. Processes HDF5 files to encode program behaviors
    3. Processes TXT files to encode program source code
    4. Combines the encodings with a normal distribution
    5. Applies PCA for dimensionality reduction
    6. Visualizes the encodings in 2D space
    7. Saves the results
    """
    try:
        # Initialize DSL for parsing programs
        seed = 42
        dsl = get_DSL_option_v2(seed=seed)

        # Load behavior and program encoders
        behavior_encoder, program_encoder = load_encoders()
        print("Successfully loaded encoders")

        # Process hdf5 files for behavior encoding
        all_behavior_vectors = []
        all_behavior_program_ids = []

        # Specify the HDF5 file path
        hdf5_file_path = "/tmp2/hubertchang/datasets_options_L30_1m_cover_branch/karel_dataset_option_L30_1m_cover_branch/data.hdf5"
        print(f"Processing {hdf5_file_path}")
        programs = process_hdf5_file(hdf5_file_path, behavior_encoder)

        if programs:
            latent_vectors, program_ids = encode_demos(programs, behavior_encoder)

            all_behavior_vectors.extend(latent_vectors)
            all_behavior_program_ids.extend(program_ids)

        # Process txt files for program encoding
        all_program_vectors = []
        all_program_ids = []

        # Specify the TXT file path
        txt_file_path = "/tmp2/hubertchang/datasets_options_L30_1m_cover_branch/karel_dataset_option_L30_1m_cover_branch/id.txt"
        print(f"Processing {txt_file_path}")
        program_data = load_program_from_txt(txt_file_path, dsl)

        if program_data:
            print(f"Loaded {len(program_data)} programs from text file")
            # Process all programs instead of just a subset
            latent_vectors, program_ids = encode_programs(program_data, program_encoder, dsl)

            if len(latent_vectors) > 0:
                all_program_vectors.extend(latent_vectors)
                all_program_ids.extend(program_ids)

        # Convert to numpy arrays
        if all_behavior_vectors and all_program_vectors:
            all_behavior_vectors = np.array(all_behavior_vectors)
            all_program_vectors = np.array(all_program_vectors)

            print(f"Encoded {len(all_behavior_vectors)} behavior samples")
            print(f"Behavior vectors shape: {all_behavior_vectors.shape}")

            print(f"Encoded {len(all_program_vectors)} program samples")
            print(f"Program vectors shape: {all_program_vectors.shape}")

            # Flatten vectors for PCA
            flat_behavior_vectors = all_behavior_vectors.reshape(all_behavior_vectors.shape[0], -1)
            flat_program_vectors = all_program_vectors.reshape(all_program_vectors.shape[0], -1)

            # Trim dimensions if needed to match
            min_dim = min(flat_behavior_vectors.shape[1], flat_program_vectors.shape[1])
            flat_behavior_vectors = flat_behavior_vectors[:, :min_dim]
            flat_program_vectors = flat_program_vectors[:, :min_dim]

            # Stack only behavior and program vectors for PCA
            combined_data = np.vstack([flat_behavior_vectors, flat_program_vectors])

            # Fit PCA on combined data to reduce to 2D
            pca = PCA(n_components=2)
            combined_pca = pca.fit_transform(combined_data)

            # Split PCA results back into their original groups
            behavior_pca = combined_pca[:len(flat_behavior_vectors)]
            program_pca = combined_pca[len(flat_behavior_vectors):]

            # Generate 2D normal distribution for comparison
            num_samples = min(len(behavior_pca) + len(program_pca), 10000)
            normal_2d = np.random.normal(0, 1, (num_samples, 2))

            # Plot results
            plt.figure(figsize=(12, 10))

            # Plot behavior encodings
            plt.scatter(behavior_pca[:, 0], behavior_pca[:, 1], alpha=0.5, label='Behavior Encodings', color='lightblue')

            # Plot program encodings
            plt.scatter(program_pca[:, 0], program_pca[:, 1], alpha=0.5, label='Program Encodings', color='orange')

            # Plot normal distribution
            plt.scatter(normal_2d[:, 0], normal_2d[:, 1], alpha=0.05, label='Normal Distribution (0,1)', color='lightgreen')

            plt.title('Latent Space Encodings in 2D PCA Space')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig('pca_plot.png', dpi=300)
            plt.close()

            # Save results to a file for further analysis
            results = {
                'behavior_program_ids': all_behavior_program_ids,
                'behavior_vectors': flat_behavior_vectors,
                'behavior_pca': behavior_pca,
                'program_ids': all_program_ids,
                'program_vectors': flat_program_vectors,
                'program_pca': program_pca,
                'normal_2d': normal_2d,
                'pca_explained_variance_ratio': pca.explained_variance_ratio_
            }
            np.save('pca_plot.npy', results)
            print("Results saved to pca_plot.npy")
        else:
            print("No latent vectors were encoded. Check for errors above.")

    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 
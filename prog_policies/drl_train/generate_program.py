#!/usr/bin/env python
# generate_from_one_bz.py  – minimal version
# ---------------------------------------------------------------
# 1. play the trained PPO “stair-climber” agent   → 10 roll-outs
# 2. push those demos through the *BehaviorEncoder* (only!)      → b_z
# 3. sample the *Decoder* 10× from that single b_z               → programs
# ---------------------------------------------------------------

import random, importlib.util, pickle, numpy as np, torch, os
from stable_baselines3 import PPO
from prog_policies.drl_train  import KarelGymEnv
from prog_policies.karel_tasks import get_task_cls
from fetch_mapping import fetch_mapping
from pretrain.utils import convert_to_POMDP_np
from prog_policies.drl_train import save_gif

# ─────────────────────────  paths ───────────────────────────────
# PPO_ZIP = "/home/hubertchang/HPRL/ppo_karel_stairclimber_sparse.zip"
TaskName = "StairClimberSparse"
PPO_ZIP = f"/home/hubertchang/HPRL/expert_ckpt/ppo_karel_{TaskName}.zip"
# CKPT    = "/home/hubertchang/HPRL/pretrain/output_dir_new_vae_L40_1m_30epoch_20230104/LEAPSL_tanh_epoch30_L40_1m_h64_u256_option_latent_p1_gru_linear_cuda8-handwritten-123-20250520-021307/final_params.ptp"
CKPT = "/home/hubertchang/HPRL/pretrain/output_dir_new_vae_L40_1m_30epoch_20230104/LEAPSL_tanh_epoch30_L40_1m_h64_u256_option_latent_p1_gru_linear_cuda8-handwritten-123-20250514-033432/best_valid_params.ptp"
# CKPT    = "/home/hubertchang/HPRL/pretrain/output_dir_new_vae_L40_1m_30epoch_20230104/LEAPSL_tanh_epoch30_L40_1m_h64_u256_option_latent_p1_gru_linear_cuda8-handwritten-123-20250514-033609/best_valid_params.ptp"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)

# ──────────────────────  DSL / mapping stuff  ───────────────────

from pretrain.cfg_option_new_vae import config
dsl2prl, prl2dsl, dsl_tokens, prl_tokens = \
        fetch_mapping('mapping_karel2prl_new_vae_v2.txt')
NUM_PROGRAM_TOKENS  = len(dsl_tokens)            # 35
PAD_ID              = NUM_PROGRAM_TOKENS         # pad token index (= 35)

config['dsl2prl'] = dsl2prl
config['prl2dsl'] = prl2dsl
config['dsl_tokens'] = dsl_tokens
config['prl_tokens'] = prl_tokens
config['num_program_tokens'] = NUM_PROGRAM_TOKENS

print(f"num_program_tokens:{config['num_program_tokens']}")


# ─── ENVIRONMENT helper (matches PPO training) ────────────────────
env_args = dict(env_height=8, env_width=8, crashable=True,
                leaps_behaviour=False, max_calls=10000)
# TaskName = "StairClimber"
# TaskName = "Maze"
def make_env(seed): return KarelGymEnv(task_name=TaskName, env_args=env_args, seed=seed)

dummy_env  = make_env(0)
NUM_ACT    = dummy_env.action_space.n           # ← fixed bug: ask env
MAX_DEMO_L = config['max_demo_length']


# ─── 1. COLLECT 10 EXPERT ROLLOUTS ─────────────────────────────────
ppo = PPO.load(PPO_ZIP, device=device)

states, actions, s_len, a_len = [], [], [], []
num_rollouts = 10
for _ in range(num_rollouts):
    env, done = make_env(random.randint(0, 2**31-1)), False
    obs,_ = env.reset()
    S, A = [env.task.environment.get_state().copy()], []
    S[0] =  S[0][:8,:,:]
    while not done and len(A) < MAX_DEMO_L-1:
        act,_ = ppo.predict(obs, deterministic=True)
        obs,_,term,trunc,_ = env.step(act)
        A.append(int(act));  S.append((env.task.environment.get_state().copy())[:8,:,:])
        done = term or trunc
    states.append(np.stack(S,0));          actions.append(np.array(A,np.int16))
    s_len.append(len(A)+1);                a_len.append(len(A))

# pack tensors  (B=1 because R=10 roll-outs live in second dim)
B,R,C,H,W = 1,num_rollouts,*states[0].shape[1:]
Tmax = MAX_DEMO_L
s_h = np.zeros((B,R,Tmax+1,C,H,W),np.float32)
a_h = np.full ((B,R,Tmax)       , NUM_ACT-1,np.int16)
s_h_len = np.zeros((B,R),np.int16);  a_h_len = np.zeros((B,R),np.int16)
for r,(S,A,sl,al) in enumerate(zip(states,actions,s_len,a_len)):
    s_h[0,r,:sl] = S; a_h[0,r,:al] = A; s_h_len[0,r]=sl; a_h_len[0,r]=al

if config['POMDP']:
    s_h = s_h.reshape(B*R, Tmax+1, C,H,W)
    s_h, ok_map = convert_to_POMDP_np(s_h)
    s_h_gif = s_h.reshape(B*R*(Tmax+1), C, 3, 3)
    s_h_list = []
    for i in range(len(s_h_gif)):
        s_h_list.append(s_h_gif[i]) 
    save_gif("test.gif", s_h_list)
    s_h = s_h.reshape(B, R, Tmax+1, C, 3, 3)
    
s_h, a_h = map(lambda x: torch.tensor(x,device=device), (s_h,a_h))
s_h_len, a_h_len = map(lambda x: torch.tensor(x,device=device), (s_h_len,a_h_len))

# ─────────────────────  2. build tiny network  ──────────────────
from pretrain.models_option_new_vae import ActionBehaviorEncoder, Decoder, ActionBehaviorEncoderTransformer, DecoderTransformer
from karel_env.dsl import get_DSL_option_v2
dsl = get_DSL_option_v2(seed=0, environment=config['rl']['envs']['executable']['name'])
config['dsl']['num_agent_actions'] = len(dsl.action_functions) + 1   


HIDDEN_Z     = 64
RNN_UNITS    = 256

if config['net']['use_transformer_encoder_behavior']:
    behavior_enc = ActionBehaviorEncoderTransformer(
                                recurrent=config['recurrent_policy'],
                                num_actions=config['dsl']['num_agent_actions'],
                                hidden_size=config['num_lstm_cell_units'], 
                                rnn_type=config['net']['rnn_type'],
                                dropout=config['net']['dropout'], 
                                use_linear=config['net']['use_linear'],
                                unit_size=config['net']['num_rnn_encoder_units'],
                                transformer_layers=config['net']['transformer_layers'],
                                transformer_heads=config['net']['transformer_heads'],
                                **config).to(device)
else:
    behavior_enc = ActionBehaviorEncoder(
            recurrent=True, num_actions=NUM_ACT+1,      # +1 for <NOP>
            hidden_size=HIDDEN_Z, rnn_type='GRU',
            dropout=0.0, use_linear=True, unit_size=RNN_UNITS,
            **config).to(device)


if config['net']['use_transformer_decoder_behavior']:
    decoder = DecoderTransformer(NUM_PROGRAM_TOKENS+1, NUM_PROGRAM_TOKENS+1, recurrent=config['recurrent_policy'],
                        hidden_size=config['num_lstm_cell_units'], rnn_type=config['net']['rnn_type'],
                         dropout=config['net']['dropout'], 
                        unit_size=config['net']['num_rnn_decoder_units'], **config).to(device)
else:
    decoder = Decoder(
            num_inputs = NUM_PROGRAM_TOKENS+1,   # +<PAD>
            num_outputs= NUM_PROGRAM_TOKENS+1,
            recurrent=True, hidden_size=HIDDEN_Z,
            rnn_type='GRU', dropout=0.0, use_linear=True,
            unit_size=RNN_UNITS,  **config).to(device)

# ─────────────────────  3. load weights  ────────────────────────
ckpt = torch.load(CKPT, map_location=device)
state = ckpt[0] if isinstance(ckpt, list) else ckpt
beh_sd, dec_sd = {}, {}
for k,v in state.items():
    if   k.startswith('vae.behavior_encoder.'):
        beh_sd[k.replace('vae.behavior_encoder.','')] = v
    elif k.startswith('vae.decoder.'):
        dec_sd[k.replace('vae.decoder.','')] = v
behavior_enc.load_state_dict(beh_sd, strict=True)
decoder.load_state_dict(dec_sd, strict=True)
behavior_enc.eval(); decoder.eval()
print('✓ weights loaded')

# ─────────────────────  4. get  b_z  and decode  ────────────────


with torch.no_grad():
    bz_all = behavior_enc(s_h, a_h, s_h_len, a_h_len)           # (1,10,64)


# print(f"b_z_all_shape:{bz_all.shape}")
# print(f"bz_all:{bz_all}")
b_z = torch.tanh(bz_all).mean(dim=0)        # (1,64)

# print(f"b_z type: {type(b_z)}")
# print(f"b_z shape : {b_z.shape}")
# print(f"b_z:{b_z}")
b_z = b_z.unsqueeze(0)
b_z = b_z.repeat(2, 1)

# add normal noise with std 0, sigma 0.1 to b_z
# b_z = b_z + torch.normal(0, 0.05, size=b_z.shape).to(device)


print(f"after unsqueece{b_z.shape}")

# max_prog_len = 40
# pad_stub = torch.full((1, max_prog_len), PAD_ID, dtype=torch.long, device=device)
def run_decoder_safe(decoder, z, *dec_args, **dec_kwargs):
    """
    Wrap `decoder(...)` so the internal batch is never 1.
    z can be shape  (H,)  or  (1,H)  or  (B,H).
    """
    if z.dim() == 1:          # (H,) → (1,H)
        z = z.unsqueeze(0)

    if z.size(0) == 1:        # (1,H) → (2,H)
        z = z.repeat(2, 1)    # second row is a harmless copy

    out = decoder(None, z, *dec_args, **dec_kwargs)

    # keep only the part that corresponds to the real example
    # (index 0 everywhere the batch dim appears)
    strip = lambda t: t[0] if torch.is_tensor(t) and t.size(0) == 2 else t
    return tuple(map(strip, out))


def sample_program(b_z):
    program_output = run_decoder_safe(decoder, b_z,
                                    teacher_enforcing=False,
                                    deterministic=True)
    # print(f"program_output shape{program_output.shape}")
    # print(f"program_output : {program_output[1]}")
    listed_program_output  = program_output[1].tolist()
    # print(f"listed_program  : {listed_program_output}")
    generated_programs = dsl.intseq2str(listed_program_output)
    return generated_programs

programs = []
programs.append(sample_program(b_z))
for i in range(100):
    noisy_bz = b_z + torch.normal(0, 0.01, size=b_z.shape).to(device)
    programs.append(sample_program(noisy_bz))

# ─────────────────────  5. print / save  ─────────────────────────
print("\nPrograms decoded from one behaviour embedding:")
for i,p in enumerate(programs):
    print(f"DEF {p}")

np.save("programs_from_one_bz.npy", np.array(programs, dtype=object))
print("\nSaved to  programs_from_one_bz.npy")

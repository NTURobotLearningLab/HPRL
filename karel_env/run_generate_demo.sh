# HPRL GRU
tasks="fourCorners topOff_sparse harvester randomMaze stairClimber_sparse doorkey oneStroke seeder"
#"fourCorners topOff_sparse cleanHouse harvester randomMaze stairClimber_sparse doorkey oneStroke snake seeder" # fourCorners topOff_sparse cleanHouse harvester randomMaze stairClimber_sparse
# doorkey oneStroke snake seeder

for task in $tasks
do
CUDA_VISIBLE_DEVICES="0" python3 karel_env/karel_generate_demo.py -c pretrain/cfg_option_new_vae.py  --verbose  --mdp_type ProgramEnv_option_new_vae_v2_key2door_fixed --rl.envs.executable.task_definition custom_reward --env_task ${task} --program_file tasks/example/${task}.txt --algorithm CEM --CEM.population_size 10 --input_channel 8 --input_height 8 --input_width 8 --max_program_len 40 --dsl.max_program_len 40 --num_demo 10 --max_episode_steps 1 #--input_width 22 --input_height 14 
done

# cleanHouse: --max_episode_steps 3  --input_width 22 --input_height 14 
CUDA_VISIBLE_DEVICES="0" python3 karel_env/karel_generate_demo.py -c pretrain/cfg_option_new_vae.py  --verbose  --mdp_type ProgramEnv_option_new_vae_v2_key2door_fixed --rl.envs.executable.task_definition custom_reward --env_task cleanHouse --program_file tasks/example/cleanHouse.txt --algorithm CEM --CEM.population_size 10 --input_channel 8 --input_height 8 --input_width 8 --max_program_len 40 --dsl.max_program_len 40 --num_demo 10 --max_episode_steps 3  --input_width 22 --input_height 14 

# snake: --max_episode_steps 3
CUDA_VISIBLE_DEVICES="0" python3 karel_env/karel_generate_demo.py -c pretrain/cfg_option_new_vae.py  --verbose  --mdp_type ProgramEnv_option_new_vae_v2_key2door_fixed --rl.envs.executable.task_definition custom_reward --env_task snake --program_file tasks/example/snake.txt --algorithm CEM --CEM.population_size 10 --input_channel 8 --input_height 8 --input_width 8 --max_program_len 40 --dsl.max_program_len 40 --num_demo 1 --max_episode_steps 3



CUDA_VISIBLE_DEVICES="1" python3 pretrain/trainer_option_new_vae_L30.py -c pretrain/cfg_option_new_vae.py -d datasets_options_L30_1m_cover_branch/karel_dataset_option_L30_1m_cover_branch --verbose --train.batch_size 256 --num_lstm_cell_units 64 --net.num_rnn_encoder_units 256 --net.num_rnn_decoder_units 256 --loss.latent_loss_coef 0.1 --net.use_linear True --net.tanh_after_sample True --device cuda:0 --mdp_type ProgramEnv1_new_vae_v2 --optimizer.params.lr 1e-3 --net.latent_mean_pooling False --prefix LEAPSL_tanh_epoch30_L40_1m_h64_u256_option_latent_p1_gru_linear_cuda8 --max_program_len 40 --dsl.max_program_len 40 --input_channel 8 --train.max_epoch 30 --outdir pretrain/output_dir_new_vae_L40_1m_30epoch_20230104
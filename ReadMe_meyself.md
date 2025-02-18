# train autoencoder
python ./mian_peli 

# train ldm
python  ./main_pelvic_new.py  --cfg_path '.\models\ldm\ldm_32x32x32_8_gloabl_mlp.yaml' --max_steps 80000 --train_mode 'ldm' --log_dir '../logs/stage_2_v3

genneration image and evalutation psnr  ssim 
python ./generation_custom.py   --cfg_path .\models\ldm\ldm_32x32x32_8_gloabl_mlp.yaml --out_dirs results_ldm_global --ddim_steps 100 --vis_option False


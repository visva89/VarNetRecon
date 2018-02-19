# download dat_128_066us_complex_permuted.mat and current_filters.mat
# here https://polybox.ethz.ch/index.php/s/E9FgAzi21iVJiF5
#

python3 learn_model_300_066.py dat_128_066us_complex_permuted.mat ckpts/model.ckpt current_filters.mat
python3 recon_model_300_066.py dat_128_066us_complex_permuted.mat ckpts/model.ckpt reson_result.mat

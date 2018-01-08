function mask_visualization(mask_path)
mask = load(mask_path);
lambda_noise = transpose(fliplr(mask.lambda_noise_r));
lambda_noisy = transpose(fliplr(mask.lambda_noisy_r));
colormap gray
subplot(1, 2, 1), imagesc(lambda_noise);
title('noise mask');
colorbar
subplot(1, 2, 2), imagesc(lambda_noisy);
title('noisy mask');
colorbar


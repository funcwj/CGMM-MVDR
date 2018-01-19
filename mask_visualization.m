function mask_visualization(mask_path)
mask = load(mask_path);
lambda_noise = real(mask.lambda_noise);
lambda_clean = 1 - real(mask.lambda_noise);
lambda_noise = transpose(fliplr(lambda_noise));
lambda_noisy = transpose(fliplr(lambda_clean));
colormap gray
subplot(1, 2, 1), imagesc(lambda_noise);
title('noise mask');
colorbar
subplot(1, 2, 2), imagesc(lambda_noisy);
title('clean mask');
colorbar


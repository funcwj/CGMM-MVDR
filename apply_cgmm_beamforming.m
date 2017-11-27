function apply_cgmm_beamforming(prefix, output, iters)

%  Apply MVDR based on mask estimated by CGMM

if nargin < 1 || nargin > 3
    error('format error: apply_cgmm_beamforming(prefix, output, [iters = 20])');
end

if nargin <= 2
    iters = 20;
end

if nargin == 1
    output = 'CGMM_ENHANCED';
end

assert(ischar(prefix));
assert(ischar(output));

num_channels = 6;
num_iters    = iters;
frame_length = 400;
fft_length   = 512;
frame_shift  = 100;
theta        = 10^-4;
% hamming_wnd  = hamming(frame_length, 'periodic');
hanning_wnd  = hanning(frame_length, 'periodic');

for c = 1: num_channels
    samples = audioread([prefix '.CH' int2str(c) '.wav']);
    frames  = enframe(samples, hanning_wnd, frame_shift);
    frames_size = size(frames);
    frames_padding = zeros(frames_size(1), fft_length);
    frames_padding(:, 1: frame_length) = frames;
    % rfft: T x F
    spectrums(:, :, c) = rfft(frames_padding, fft_length, 2);
end

specs = permute(spectrums(:, :, [1, 3, 4, 5, 6]), [3, 1, 2]);
[num_channels, num_frames, num_bins] = size(specs);

% CGMM parameters
lambda_noise = zeros(num_frames, num_bins);
lambda_noisy = zeros(num_frames, num_bins);
phi_noise    = ones(num_frames, num_bins);
phi_noisy    = ones(num_frames, num_bins);
R_noise      = zeros(num_channels, num_channels, num_bins);
R_noisy      = zeros(num_channels, num_channels, num_bins);
R_xn         = zeros(num_channels, num_channels, num_bins);


yyh = zeros(num_channels, num_channels, num_frames, num_bins);

% init R_noisy R_noise R_xn
for f = 1: num_bins
    for t = 1: num_frames
        y = specs(:, t, f);
        h = y * y';
        yyh(:, :, t, f) = h;
        R_noisy(:, :, f) = R_noisy(:, :, f) + h;
    end
    R_noisy(:, :, f) = R_noisy(:, :, f) / num_frames;
    R_noise(:, :, f) = eye(num_channels, num_channels);
end

R_xn = R_noisy;

% start CGMM training
p_noise = ones(num_frames, num_bins);
p_noisy = ones(num_frames, num_bins);

for iter = 1: num_iters

    for f = 1: num_bins
        R_noisy_onbin = R_noisy(:, :, f);
        R_noise_onbin = R_noise(:, :, f);
        
        if rcond(R_noisy_onbin) < theta
            R_noisy_onbin = R_noisy_onbin + theta * eye(num_channels) * max(diag(R_noisy_onbin));
        end
        
        if rcond(R_noise_onbin) < theta
            R_noise_onbin = R_noise_onbin + theta * eye(num_channels) * max(diag(R_noisy_onbin));
        end
       
        R_noisy_inv = inv(R_noisy_onbin);
        R_noise_inv = inv(R_noise_onbin);
        R_noisy_accu = zeros(num_channels, num_channels);
        R_noise_accu = zeros(num_channels, num_channels);
        
        for t = 1: num_frames
            corre   = yyh(:, :, t, f);
            obs     = specs(:, t, f);
            
            % update phi
            phi_noise(t, f) = trace(corre * R_noise_inv) / num_channels;
            phi_noisy(t, f) = trace(corre * R_noisy_inv) / num_channels;
            
            % update lambda
            k_noise = obs' * (R_noise_inv / phi_noise(t, f)) * obs;
            p_noise(t, f) = exp(-k_noise) / (pi * det(phi_noise(t, f) * R_noise_onbin));
            k_noisy = obs' * (R_noisy_inv / phi_noisy(t, f)) * obs;
            p_noisy(t, f) = exp(-k_noisy) / (pi * det(phi_noisy(t, f) * R_noisy_onbin));
            lambda_noise(t, f) = p_noise(t, f) / (p_noise(t, f) + p_noisy(t, f));
            lambda_noisy(t, f) = p_noisy(t, f) / (p_noise(t, f) + p_noisy(t, f));
            
            % accu R
            R_noise_accu = R_noise_accu + lambda_noise(t, f) / phi_noise(t, f) * corre;
            R_noisy_accu = R_noisy_accu + lambda_noisy(t, f) / phi_noisy(t, f) * corre;
        end
        % update R
        R_noise(:, :, f) = R_noise_accu / sum(lambda_noise(:, f));
        R_noisy(:, :, f) = R_noisy_accu / sum(lambda_noisy(:, f));
        
    end
    % Q = sum(sum(lambda_noise .* log(p_noise) + lambda_noisy .* log(p_noisy))) / (num_frames * num_bins)
end

% bigger entropy assigned to noise part
for f = 1: num_bins
    eig_value1 = eig(R_noise(:, :, f));
    eig_value2 = eig(R_noisy(:, :, f));
    en_noise = -eig_value1' / sum(eig_value1) * log(eig_value1 / sum(eig_value1));
    en_noisy = -eig_value2' / sum(eig_value2) * log(eig_value2 / sum(eig_value2));
    
    if en_noise < en_noisy
        Rn = R_noise(:, :, f);
        R_noise(:, :, f) = R_noisy(:, :, f);
        R_noisy(:, :, f) = Rn;
    end
end

% get Rn, reference to eq.4
R_n = zeros(num_channels, num_channels, num_bins);
for f = 1: num_bins
    for t = 1: num_frames
        R_n(:, :, f) = R_n(:, :, f) + lambda_noise(t, f) * yyh(:, :, t, f);
    end
    R_n(:, :, f) = R_n(:, :, f) / sum(lambda_noise(:, f));
end

R_x = R_xn - R_n;

% apply MVDR beamforming

specs_enhan = zeros(num_frames, num_bins);

for f = 1: num_bins
    % using Rx to estimate steer vector
    [vector, value] = eig(R_x(:, :, f));
    steer_vector = vector(:, 1);
    % feed Rn into MVDR
    Rn_inv = inv(R_n(:, :, f));
    % w: M x 1
    w = Rn_inv * steer_vector / (steer_vector' * Rn_inv * steer_vector);
    % specs M x T x F
    specs_enhan(:, f) = w' * specs(:, :, f);
end

% reconstruction
frames_enhan = irfft(specs_enhan, fft_length, 2);
% size(frames_enhan)
signal_enhan = overlapadd(frames_enhan(:, 1: frame_length), hanning_wnd, frame_shift);
audiowrite([output '.wav'], signal_enhan ./ max(abs(signal_enhan)), 16000);

end


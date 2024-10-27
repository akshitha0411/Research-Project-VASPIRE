% Load the image 
img = im2double(imread('download.jpeg'));

% Display the original blurry image
figure(1);
imshow(img);
title('Original Blurry Image');

% Adjust 'LEN' (length of blur) and 'THETA' (angle of blur) as needed
LEN = 5;   % Length of the blur (pixels)
THETA = 0.1; % Angle of the blur (degrees)
PSF = fspecial('motion', LEN, THETA);

% Number of iterations for Lucy-Richardson deconvolution
num_iterations = 20;

% Apply Lucy-Richardson deconvolution
deblurred_img = deconvlucy(img, PSF, num_iterations);

% deblurred image
figure(2);
imshow(deblurred_img);
title('Deblurred Image using Lucy-Richardson Deconvolution');
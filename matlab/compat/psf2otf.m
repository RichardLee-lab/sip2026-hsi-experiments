function otf = psf2otf(psf, outSize)
% Replacement for the Image Processing Toolbox function psf2otf, in base MATLAB, for the calls in
% LRTDTV.m (difference kernels [+1; -1] and [+1, -1], of which only abs(otf) is used).
% Same algorithm as the toolbox: pad the PSF with zeros to outSize, shift its centre to the first
% element, take the n-D FFT, and drop an imaginary part that is only round-off.
% run_matlab_all.m puts this folder on the path only when the toolbox is not available.
psfSize = size(psf);
n = max(numel(psfSize), numel(outSize));
psfSize(end + 1:n) = 1;
outSize(end + 1:n) = 1;
if all(psf(:) == 0)
    otf = zeros(outSize);
    return;
end
padded = zeros(outSize, 'like', psf);
idx = arrayfun(@(k) 1:psfSize(k), 1:n, 'UniformOutput', false);
padded(idx{:}) = psf;
padded = circshift(padded, -floor(psfSize / 2));
otf = fftn(padded);
nElem = prod(psfSize);
nOps = 0;
for k = 1:n
    nOps = nOps + psfSize(k) * log2(psfSize(k)) * nElem / psfSize(k);
end
if max(abs(imag(otf(:)))) / max(abs(otf(:))) <= nOps * eps
    otf = real(otf);
end
end

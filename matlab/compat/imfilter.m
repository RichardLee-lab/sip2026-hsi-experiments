function B = imfilter(A, H, varargin)
% Replacement for the Image Processing Toolbox function imfilter, in base MATLAB, for the one call
% in LRTV_accelerate.m: imfilter(temp, H, 'circular') with H = 1.
% It implements imfilter(A, H, 'circular') for a 2-D H: correlation with circular boundary
% conditions, output of the same size as A, H applied to every slice of A (as the toolbox does).
% run_matlab_all.m puts this folder on the path only when the toolbox is not available.
if ~(numel(varargin) == 1 && strcmpi(varargin{1}, 'circular')) || ~ismatrix(H)
    error('compat imfilter: only imfilter(A, H, ''circular'') with a 2-D H is implemented');
end
c = floor((size(H) + 1) / 2);             % centre element of H, as in the toolbox
B = zeros(size(A), 'like', A);
for i = 1:size(H, 1)
    for j = 1:size(H, 2)
        if H(i, j) ~= 0                   % B(x, y) += H(i, j) * A(x + i - c1, y + j - c2), circularly
            B = B + H(i, j) * circshift(A, [c(1) - i, c(2) - j]);
        end
    end
end
end

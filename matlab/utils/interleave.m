function res = interleave(v)
%INTERLEAVE Takes a row vector and return a column vector with each element
%repeated two times.
%
%   INTERLEAVE([1, 2]) = [1; 1; 2; 2]

res = [v; v];
res = res(:);

end
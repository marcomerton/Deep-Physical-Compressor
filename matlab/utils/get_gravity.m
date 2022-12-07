function gravity = get_gravity(intensity, theta, size)
%gravity = GET_GRAVITY Returns the gravity vector of the specificed size.
%The gravity has module 'intensity' and orientation angle 'theta'.

gravity = zeros(size);
gravity(1:2:end) = intensity * cos(theta);
gravity(2:2:end) = intensity * sin(theta);

end


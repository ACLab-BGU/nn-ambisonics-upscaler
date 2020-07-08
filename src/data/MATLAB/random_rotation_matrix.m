function R = random_rotation_matrix()

x = rand_on_sphere(1, "cart");
theta = rand()*2*pi;
R = axang2rotm([x theta]);

end


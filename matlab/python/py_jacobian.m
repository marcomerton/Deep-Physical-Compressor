function J = py_jacobian(q)
%J = PY_JACOBIAN
%

J = pyrun([ ...
    "emb = torch.Tensor(np.array(q))" ....
    "j = jacobian(modelf_j, emb).detach().numpy()"], ...
    "j", ...
    q=q);

J = double(J);

end
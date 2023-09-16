function emb = py_encode(pos)
%emb = PY_ENCODE Returns the encoding of the masses configuration. Assumes
%a model and connectivity (edge_index & edge_attr) are in python workspace.

emb = pyrun([ ...
    "d.pos = torch.Tensor(np.array(x))" ...
    "with torch.no_grad(): emb = model.encode(d)" ...
    "emb = emb.numpy().squeeze()" ...
    "emb_size = len(emb)"], ...
    "emb", ...
    x=pos);

emb = double(emb);

end
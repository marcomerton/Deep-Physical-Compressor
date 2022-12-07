import torch


def train(model, optimizer, lr_scheduler, loss, epochs, train_data, valid_data, device='cpu',
        save_model=True, save_best=True, save_scores=True, filename=None, noise=None, verbose=True
    ):
    """ Training loop function.
    If a filename is provided, this function also saves the final model state dictionary and the training and validation scores throughout the training.
    The last training and validation scores are returned in any case.

    ARGS:
    - model: the model to be trained.
    - optimizer: optimizer to update model's parameters.
    - lr_scheduler: scheduler to adjust the optimizer learning rate.
    - loss: loss function used for training.
    - epochs: number of training epochs.
    - train_data: iterable over torch_geometric.data.Data (or similar) used as training examples.
    - valid_data: iterable over torch_geometric.data.Data (or similar) used to evaluate the model.
    - device: device where the model and data are stored (any device is fine, but using the same one can avoid some overheads).
    - save_model: whether to save the model after training
    - save_best: whether to save the model with the best validation score
    - save_scores: whether to save the training/validation scores along the training
    - filename: file where the model state is saved at the end of training. This is also the prefix for the file names where the learning curves are saved.
    - noise: 
    - verbose: whether to pring training and validation score at the end of each epoch.
    """
    if (save_scores or save_model or save_best) and filename is None:
        print("'filename' must not be None is 'save_scores' or 'save_model' are set...")
        print("Not saving results: 'filename' is None")
        save_scores = False
        save_model = False
        save_best = False

    if save_scores:
        tr_losses = torch.empty(len(train_data) * epochs, device=device)
        vl_losses = torch.empty(epochs, device=device)
    
    if save_best:
        best_tr = float("inf")
        best_vl = float("inf")

    for e in range(epochs):
        if verbose: print(f"Epoch: {e+1}/{epochs}")

        # Update phase
        tr_loss = 0
        model.train()
        for i, b in enumerate(train_data):
            optimizer.zero_grad()
            if noise is not None: b.pos += noise * (2 * torch.rand_like(b.pos) - 1)
            out = model(b)
            l = loss(out, b.pos)
            l.backward()
            optimizer.step()
            if save_scores: tr_losses[e * len(train_data) + i] = l.detach()
            tr_loss += l
        tr_loss /= len(train_data)

        # Evaluate phase
        if save_scores or verbose or save_best:
            model.eval()
            val_loss = evaluate(model, valid_data, loss)

        if save_scores: vl_losses[e] = val_loss.item()
        if verbose: print(f"\ttrain_loss: {tr_loss:0.5e}\tval_loss: {val_loss:0.5e}")
        if save_best and (val_loss < best_vl):
            # Move to cpu before saving
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, filename)
            best_vl = val_loss
            best_tr = tr_loss

        # Adjust lr
        lr_scheduler.step()


    if save_scores:
        torch.save(tr_losses.cpu(), f"{filename}_tr")
        torch.save(vl_losses.cpu(), f"{filename}_vl")
    
    if save_best:
        return (best_tr.item(), best_vl.item())
    
    else:
        # Recompute training loss on whole dataset
        model.eval()
        tr_loss = evaluate(model, train_data, loss)
        if not verbose: val_loss = evaluate(model, valid_data, loss)

        if save_model:
            # Move to cpu before saving
            torch.save(model.cpu().state_dict(), filename)
        
        return (tr_loss.item(), val_loss.item())



@torch.no_grad()
def evaluate(model, data, metric):
    """ Evaluate a model on the given data with the given metric.
    The average score is returned.
    """
    score = 0
    for d in data:
        out = model(d)
        score += metric(out, d.pos)

    return score / len(data)

@torch.no_grad()
def evaluate_simulation(model, xs, d, metric):
    """"""
    score = 0
    for x in xs:
        d.pos = x
        out = model(d)
        score += metric(out, d.pos)
    
    return score / len(xs)


def train_with_early_stopping(model, optimizer, loss, epochs, train_data, valid_data,
        patience=30, save_model=True, save_scores=True, filename=None, verbose=True
    ):

    tr_losses = []
    vl_losses = []
    
    best_tr = float("inf")
    best_vl = float("inf")
    best_epoch = 0

    for e in range(epochs):
        if verbose: print(f"Epoch: {e+1}/{epochs}")

        # Update phase
        tr_loss = 0
        model.train()
        for b in train_data:
            optimizer.zero_grad()
            out = model(b)
            l = loss(out, b.pos)
            l.backward()
            optimizer.step()
            tr_losses.append( l.detach() )
            tr_loss += l.detach()
        tr_loss /= len(train_data)

        model.eval()
        vl_loss = evaluate(model, valid_data, loss)
        vl_losses.append( vl_loss )

        if verbose: print(f"\ttrain_loss: {tr_loss:0.5e}\tval_loss: {vl_loss:0.5e}")

        if vl_loss < best_vl:
            # Move to cpu before saving
            if save_model: torch.save({k: v.cpu() for k, v in model.state_dict().items()}, filename)
            best_tr = tr_loss
            best_vl = vl_loss
            best_epoch = e
        
        if e - best_epoch > patience:
            break


    if save_scores:
        torch.save(torch.tensor(tr_losses), f"{filename}_tr")
        torch.save(torch.tensor(vl_losses), f"{filename}_vl")

    return best_epoch+1, best_tr, best_vl

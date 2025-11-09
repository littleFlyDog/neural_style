import torch
from dataset import extract_features, content_layers, style_layers
from loss_function import compute_loss


def train_fn(model, contents_X, styles_X_gram, num_epochs, trainer, scheduler):
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_X_hat, styles_X_hat = extract_features(
            model(), content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            model(), contents_X_hat, styles_X_hat, contents_X, styles_X_gram)
        l.backward()
        print(f'epoch {epoch + 1}, ' +f'loss {l.item():.4f}, ' )
        trainer.step()
        scheduler.step()
    return model()
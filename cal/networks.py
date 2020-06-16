import torch
from torch import nn


class MLP(nn.Module):
    """MLP specified by a list of hidden layer sizes."""
    def __init__(self, n_hidden, input_dim=1, output_dim=1):
        super(MLP, self).__init__()
        self.layers = [nn.Linear(input_dim, n_hidden[0])]
        for cur, nxt in zip(n_hidden, n_hidden[1:]):
            self.layers.append(nn.Linear(cur, nxt))
        self.layers.append(nn.Linear(n_hidden[-1], output_dim))
        self.model = nn.Sequential(*self.layers)

    def forward(self, xs):
        """Compute the forward pass using inputs `xs`."""
        if len(xs.shape) == 1:
            out = xs.view(xs.shape[0], 1)
        else:
            out = xs
        for layer in self.layers[:-1]:
            out = nn.functional.relu(layer(out))
        out = self.layers[-1](out)
        return out.view(xs.shape)

    def ll(self, xs, ys, **kwargs):
        """Compute the log-likelihood of `ys` given the inputs `xs`."""
        locs = self.forward(xs)
        scale = kwargs.get("scale", 1.0)
        log_probs = torch.distributions.Normal(locs, scale).log_prob(ys)
        return log_probs


class Generator(MLP):
    """An MLP generator network. This is a stochastic network, conditioned on
    a Gaussian noise variable.
    """
    def __init__(self, n_hidden, input_dim=1, noise_dim=1):
        super(Generator, self).__init__(
            n_hidden, input_dim=noise_dim + input_dim
        )
        self.noise_dim = noise_dim

    def forward(self, xs, n_samples=1, device="cpu"):
        """Compute `n_samples` forward passes for given inputs `xs`."""
        xs = [x.view(x.shape[0], 1) if len(x.shape) == 1 else x for x in xs]
        out = torch.cat(xs, dim=1)
        batch_size = out.shape[0]
        out_repeated = torch.repeat_interleave(out, repeats=n_samples, dim=0)
        noise = torch.randn(batch_size * n_samples, self.noise_dim).to(device)
        out = torch.cat([out_repeated, noise], dim=1)
        for i, layer in enumerate(self.layers[:-1]):
            out = nn.functional.leaky_relu(layer(out))
        out = self.layers[-1](out)
        return out.view(batch_size * n_samples, 1)

    def ll(self, xs, ys, **kwargs):
        raise AttributeError("The generator has no explicit likelihood.")


class Discriminator(MLP):
    def __init__(self, input_dim, n_hidden):
        """An MLP binary discriminator network."""
        super(Discriminator, self).__init__(n_hidden, input_dim=input_dim)

    def forward(self, xs, as_probs=False):
        """Compute the forward pass of inputs `xs` and return the logits,
        if `as_probs` is False, of `xs` being real data."""
        xs = [x.view(x.shape[0], 1) if len(x.shape) == 1 else x for x in xs]
        out = torch.cat(xs, dim=1)
        for i, layer in enumerate(self.layers[:-1]):
            out = nn.functional.leaky_relu(layer(out))
        out = self.layers[-1](out)
        if as_probs:
            out = torch.sigmoid(out)
        return out

    def ll(self, xs, ys, **kwargs):
        """Compute the likelihood of `ys` for inputs `xs`."""
        ys = ys.view(ys.shape[0], 1) if len(ys.shape) == 1 else ys
        logits = self.forward(xs, as_probs=False)
        log_probs = torch.distributions.Bernoulli(logits=logits).log_prob(ys)
        return log_probs

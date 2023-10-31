import haiku as hk
import jax
from jax import numpy as jnp


def original_encoder(input_size, latent_size):
    def net_fn(inputs):
        network = hk.Sequential(
            [
                hk.Linear(input_size),
                jax.nn.relu,
                hk.Linear(20),
                jax.nn.relu,
                hk.Linear(15),
                jax.nn.relu,
                hk.Linear(10),
                jax.nn.relu,
                hk.Linear(latent_size),
            ]
        )
        return network(inputs)

    return net_fn


def original_decoder(input_size):
    def net_fn(inputs):
        network = hk.Sequential(
            [
                hk.Linear(10),
                jax.nn.relu,
                hk.Linear(15),
                jax.nn.relu,
                hk.Linear(20),
                jax.nn.relu,
                hk.Linear(input_size),
            ]
        )
        return network(inputs)

    return net_fn


def original_autoencoder(input_size, latent_size):
    def net_fn(inputs):
        network = hk.Sequential(
            [
                original_encoder(input_size, latent_size),
                original_decoder(input_size),
            ]
        )
        return network(inputs)

    return net_fn


def encoder(input_size, latent_size):
    def net_fn(inputs):
        network = hk.Sequential(
            [
                hk.Linear(input_size),
                jax.nn.relu,
                hk.Linear(latent_size),
                jax.nn.relu,
            ]
        )
        return network(inputs)

    return net_fn


def decoder(input_size):
    def net_fn(inputs):
        network = hk.Sequential(
            [
                hk.Linear(input_size),
                jax.nn.sigmoid,
            ]
        )
        return network(inputs)

    return net_fn


def autoencoder(input_size, latent_size):
    def net_fn(inputs):
        network = hk.Sequential([encoder(input_size, latent_size), decoder(input_size)])
        return network(inputs)

    return net_fn


def custom_encoder(layer_sizes):
    def net_fn(inputs):
        layers = []
        for l in layer_sizes:
            layers.append(hk.Linear(l))
            layers.append(jax.nn.relu)

        network = hk.Sequential(layers)
        return network(inputs)

    return net_fn


def custom_decoder(layer_sizes):
    def net_fn(inputs):
        layers = []
        for l in layer_sizes:
            layers.append(hk.Linear(l))
            layers.append(jax.nn.relu)

        network = hk.Sequential(layers)
        return network(inputs)

    return net_fn


def custom_autoencoder(encoder_layers, decoder_layers):
    def net_fn(inputs):
        network = hk.Sequential(
            [custom_encoder(encoder_layers), custom_decoder(decoder_layers)]
        )
        return network(inputs)

    return net_fn


def mse(params, data, model):
    predictions = model.apply(params, data)
    losses = jnp.sum(jnp.square(predictions - data), axis=-1)

    loss = jnp.sqrt(jnp.mean(losses))
    return loss


classical_loss_dict = {"mse": mse}

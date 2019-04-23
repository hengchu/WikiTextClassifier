import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

def train(model, X, y, loss_fun, optimizer):
  model.train()
  optimizer.zero_grad()
  output = model(X)
  loss = loss_fun(output, y)
  loss.backward()
  optimizer.step()

  with torch.no_grad():
    _, predicted = torch.max(model(X), 1)
    return (loss, (predicted == y).float().sum() / X.shape[0])

def transformer_train(model, X, masks, y, loss_fun, optimizer):
  model.train()
  optimizer.zero_grad()
  output = model(X, masks)
  loss = loss_fun(output, y)
  loss.backward()
  optimizer.step()

  with torch.no_grad():
    _, predicted = torch.max(model(X, masks), 1)
    return (loss, (predicted == y).float().sum() / X.shape[0])

def ae_train(model, X, loss_fun, optimizer):
  model.train()
  optimizer.zero_grad()
  output = model(X)
  loss = loss_fun(output, X)
  loss.backward()
  optimizer.step()
  return loss

def vae_train(model, X, loss_fun, optimizer):
  model.train()
  optimizer.zero_grad()
  output, mu, logvar = model(X)
  loss = loss_fun(output, X, mu, logvar)
  loss.backward()
  optimizer.step()
  return loss

def unrolled_gan_train(g_model, d_model, X, z, g_opt, d_opt, g_loss, d_loss, unrolled_steps=5):
  # train the discriminator
  def train_discriminator():
    d_model.train()
    d_opt.zero_grad()
    d_loss_tensor, real_loss, fake_loss = d_loss(d_model, g_model, X, z)
    d_loss_tensor.backward(create_graph=True)
    d_opt.step()

    return d_loss_tensor, real_loss, fake_loss

  def train_generator():
    g_model.train()
    g_opt.zero_grad()

    d_backup = None
    if unrolled_steps > 0:
      d_backup = copy.deepcopy(d_model)
      for i in range(unrolled_steps):
        train_discriminator()

    g_loss_tensor = g_loss(d_model, g_model, z)
    g_loss_tensor.backward()
    g_opt.step()

    if d_backup:
      d_model.load_state_dict(d_backup.state_dict())
      del d_backup

    return g_loss_tensor

  g_loss = train_generator()
  d_loss, real_loss, fake_loss = train_discriminator()

  return g_loss, d_loss, real_loss, fake_loss

def gan_train(g_model, d_model, X, z, g_opt, d_opt, g_loss, d_loss):
  # train the generator twice
  g_model.train()
  g_opt.zero_grad()
  g_loss_tensor = g_loss(d_model, g_model, z)
  g_loss_tensor.backward()
  g_opt.step()

  g_model.train()
  g_opt.zero_grad()
  g_loss_tensor = g_loss(d_model, g_model, z)
  g_loss_tensor.backward()
  g_opt.step()

  d_model.train()
  d_opt.zero_grad()
  d_loss_tensor, real_loss, fake_loss = d_loss(d_model, g_model, X, z)
  d_loss_tensor.backward()
  d_opt.step()

  return g_loss_tensor, d_loss_tensor, real_loss, fake_loss

def transformer_validate(model, dataloader, device):
  model.eval()

  with torch.no_grad():
    num_correct = 0
    total = 0

    for _, batch in enumerate(dataloader):
      X = batch[0]
      masks = batch[1]
      y = batch[2]
      _, predicted = torch.max(model(X.to(device), masks.to(device)), 1)
      num_correct += (predicted == y.to(device)).float().sum()
      total += X.shape[0]

    return num_correct / total

def validate(model, dataloader, device):
  model.eval()

  with torch.no_grad():
    num_correct = 0
    total = 0

    for _, batch in enumerate(dataloader):
      X, y = batch
      _, predicted = torch.max(model(X.to(device)), 1)
      num_correct += (predicted == y.to(device)).float().sum()
      total += X.shape[0]

    return num_correct / total

def conv_size(in_channels, height, width, out_channels, kernel_size, stride=1, padding=0):
  fake_data = torch.zeros((1, in_channels, height, width))
  conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
  output = conv(fake_data)
  return output.squeeze().size()

def maxpool_size(in_channels, height, width, kernel_size, stride=1, padding=0):
  return conv_size(in_channels, height, width, in_channels, kernel_size, stride, padding)

def up_conv_size(in_channels, height, width,
                 out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0):
  fake_data = torch.zeros((1, in_channels, height, width))
  conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
  output = conv(fake_data)
  return output.squeeze().size()

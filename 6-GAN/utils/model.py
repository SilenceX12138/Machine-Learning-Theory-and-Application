import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from config import config, project_path
# from qqdm import qqdm
from tqdm.auto import tqdm
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader


def weights_init(model: nn.Module):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


def model_function(dataloader: DataLoader, G: nn.Module, D: nn.Module,
                   z_sample: Variable):
    # Loss
    criterion = nn.BCELoss()
    """ Medium: Use RMSprop for WGAN. """
    # Optimizer
    # opt_D = torch.optim.Adam(D.parameters(), lr=config.lr, betas=(0.5, 0.999))
    # opt_G = torch.optim.Adam(G.parameters(), lr=config.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.RMSprop(D.parameters(), lr=config.lr)
    opt_G = torch.optim.RMSprop(G.parameters(), lr=config.lr)
    steps = 0
    for e, epoch in enumerate(range(config.n_epoch)):
        progress_bar = tqdm(dataloader)
        for i, data in enumerate(progress_bar):
            imgs = data
            imgs = imgs.cuda()

            bs = imgs.size(0)

            # ============================================
            #  Train D
            # ============================================
            z = Variable(torch.randn(bs, config.z_dim)).cuda()
            r_imgs = Variable(imgs).cuda()
            f_imgs = G(z)
            """ Medium: Use WGAN Loss. """
            # Label
            # r_label = torch.ones((bs)).cuda()
            # f_label = torch.zeros((bs)).cuda()

            # Model forwarding
            # r_logit = D(r_imgs.detach())
            # f_logit = D(f_imgs.detach())

            # Compute the loss for the discriminator.
            # r_loss = criterion(r_logit, r_label)
            # f_loss = criterion(f_logit, f_label)
            # loss_D = (r_loss + f_loss) / 2

            # WGAN Loss
            loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs))

            # Model backwarding
            D.zero_grad()
            loss_D.backward()

            # Update the discriminator.
            opt_D.step()
            """ Medium: Clip weights of discriminator. """
            for p in D.parameters():
               p.data.clamp_(-config.clip_value, config.clip_value)

            # ============================================
            #  Train G
            # ============================================
            if steps % config.n_critic == 0:
                # Generate some fake images.
                z = Variable(torch.randn(bs, config.z_dim)).cuda()
                f_imgs = G(z)

                # Model forwarding
                # f_logit = D(f_imgs)
                """ Medium: Use WGAN Loss"""
                # Compute the loss for the generator.
                # loss_G = criterion(f_logit, r_label)
                # WGAN Loss
                loss_G = -torch.mean(D(f_imgs))

                # Model backwarding
                G.zero_grad()
                loss_G.backward()

                # Update the generator.
                opt_G.step()

            steps += 1

            # Set the info of the progress bar
            #   Note that the value of the GAN loss is not directly related to
            #   the quality of the generated images.
            # progress_bar.set_infos({
            #     'Loss_D': round(loss_D.item(), 4),
            #     'Loss_G': round(loss_G.item(), 4),
            #     'Epoch': e + 1,
            #     'Step': steps,
            # })

        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(project_path.log_dir,
                                f'Epoch_{epoch+1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')

        # Show generated images in the jupyter notebook.
        grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        G.train()

        if (e + 1) % 5 == 0 or e == 0:
            # Save the checkpoints.
            torch.save(G.state_dict(),
                       os.path.join(project_path.ckpt_dir, 'G.pth'))
            torch.save(D.state_dict(),
                       os.path.join(project_path.ckpt_dir, 'D.pth'))

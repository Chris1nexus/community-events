import argparse
import os
import numpy as np
import itertools
from pathlib import Path
import datetime
import time
import sys

from PIL import Image

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader

from modeling_cyclegan import GeneratorResNet, Discriminator

from utils import ReplayBuffer, LambdaLR

from datasets import load_dataset

from accelerate import Accelerator

import torch.nn as nn
import torch

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="huggan/facades", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU threads to use during batch generation")
    parser.add_argument("--image_size", type=int, default=256, help="Size of images for training")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
            "--push_to_hub",
            action="store_true",
            help="Whether to push the model to the HuggingFace hub after training.",
            )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        required="--push_to_hub" in sys.argv,
        type=Path,
        help="Path to save the model. Will be created if it doesn't exist already.",
    )
    parser.add_argument(
        "--model_name",
        required="--push_to_hub" in sys.argv,
        type=str,
        help="Name of the model on the hub.",
    )
    parser.add_argument("--wandb", action="store_true", help="If passed, will log to Weights and Biases.")

    parser.add_argument(
        "--organization_name",
        required=False,
        default="Chris1",
        type=str,
        help="Organization name to push to, in case args.push_to_hub is specified.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./output"), help="Name of the directory to dump generated images during training.")
    return parser.parse_args(args=args)


def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)


def training_function(config, args):
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu, mixed_precision=args.mixed_precision)
    
    
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.wandb and accelerator.is_local_main_process:
        import wandb
        wandb.init(project='--'.join(str(args.output_dir).split("/")),#[-1], 
                       entity="chris1nexus")    
        wandb.config = vars(args)
                
    
    # Create sample and checkpoint directories
    os.makedirs(os.path.join(args.output_dir,"images/%s" % args.dataset_name), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,"saved_models/%s" % args.dataset_name), exist_ok=True)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    input_shape = (args.channels, args.image_size, args.image_size)
    # Calculate output shape of image discriminator (PatchGAN)
    output_shape = (1, args.image_size // 2 ** 4, args.image_size // 2 ** 4)

    # Initialize generator and discriminator
    G_AB = GeneratorResNet(input_shape, args.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, args.n_residual_blocks)
    D_A = Discriminator(args.channels)
    D_B = Discriminator(args.channels)

    if args.epoch != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load(
            os.path.join(args.output_dir,"saved_models/%s/G_AB_%d.pth" % (args.dataset_name, args.epoch))))
        G_BA.load_state_dict(torch.load(
            os.path.join(args.output_dir,"saved_models/%s/G_BA_%d.pth" % (args.dataset_name, args.epoch))))
        D_A.load_state_dict(torch.load(
            os.path.join(args.output_dir,"saved_models/%s/D_A_%d.pth" % (args.dataset_name, args.epoch))))
        D_B.load_state_dict(torch.load(
            os.path.join(args.output_dir,"saved_models/%s/D_B_%d.pth" % (args.dataset_name, args.epoch))))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=args.lr, betas=(args.beta1, args.beta2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(args.num_epochs, args.epoch, args.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(args.num_epochs, args.epoch, args.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(args.num_epochs, args.epoch, args.decay_epoch).step
    )

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Image transformations
    transform = Compose([
        Resize(int(args.image_size * 1.12), Image.BICUBIC),
        RandomCrop((args.image_size, args.image_size)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def transforms(examples):
        examples["A"] = [transform(image.convert("RGB")) for image in examples["imageA"]]
        examples["B"] = [transform(image.convert("RGB")) for image in examples["imageB"]]

        del examples["imageA"]
        del examples["imageB"]

        return examples
    print('Loading dataset')
    dataset = load_dataset(args.dataset_name)
    print('Loading completed')
    print('Transforming dataset.')
    transformed_dataset = dataset.with_transform(transforms)
    print('Dataset transformation completed.')
    splits = transformed_dataset['train'].train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']

    dataloader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_ds, batch_size=5, shuffle=True, num_workers=1)

    def sample_images(args, batches_done):
        """Saves a generated sample from the test set"""
        batch = next(iter(val_dataloader))
        G_AB.eval()
        G_BA.eval()
        real_A = batch["A"]
        fake_B = G_AB(real_A)#.detach().cpu()
        real_B = batch["B"]
        fake_A = G_BA(real_B)#.detach().cpu()
        # Arange images along x-axis
        real_A = make_grid(real_A,#.detach().cpu(),
                           nrow=5, normalize=True)
        real_B = make_grid(real_B,#.detach().cpu(), 
                           nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        save_path = os.path.join(args.output_dir,  "images/%s/%s.png" % (args.dataset_name, batches_done) )
        save_image(image_grid, save_path, normalize=False)
        return save_path

    G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, dataloader, val_dataloader = accelerator.prepare(G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, dataloader, val_dataloader)
    print('Starting training')
    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    for epoch in range(args.epoch, args.num_epochs):
        for i, batch in enumerate(dataloader):

            # Set model input
            real_A = batch["A"]
            real_B = batch["B"]
            #print('a')
            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), *output_shape), device=accelerator.device)
            fake = torch.zeros((real_A.size(0), *output_shape), device=accelerator.device)
            #print('b')
            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            #print('b1')
            loss_identity = (loss_id_A + loss_id_B) / 2
            #print('b2')
            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
            #print('b3')
            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            #print('b4')
            # Total loss
            loss_G = loss_GAN + args.lambda_cyc * loss_cycle + args.lambda_id * loss_identity
    
            accelerator.backward(loss_G)
            optimizer_G.step()
            #print('b5')
            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            accelerator.backward(loss_D_A)
            optimizer_D_A.step()
            
            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            accelerator.backward(loss_D_B)
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------

            #print('c')
            #'''
            if accelerator.state.num_processes > 1:
                    #print('cA')
                    errD = accelerator.gather(loss_D).sum() / accelerator.state.num_processes
                    errG = accelerator.gather(loss_G).sum() / accelerator.state.num_processes
                    errGAN = accelerator.gather(loss_GAN).sum() / accelerator.state.num_processes
                    errCycle = accelerator.gather(loss_cycle).sum() / accelerator.state.num_processes   
                    errIdentity = accelerator.gather(loss_identity).sum() / accelerator.state.num_processes


                    err_id_A = accelerator.gather(loss_id_A).sum() / accelerator.state.num_processes
                    err_id_B = accelerator.gather(loss_id_B).sum() / accelerator.state.num_processes
                    err_cycle_A = accelerator.gather(loss_cycle_A).sum() / accelerator.state.num_processes
                    err_cycle_B = accelerator.gather(loss_cycle_B).sum() / accelerator.state.num_processes   
                    err_D_A = accelerator.gather(loss_D_A).sum() / accelerator.state.num_processes
                    err_D_B = accelerator.gather(loss_D_B).sum() / accelerator.state.num_processes
                    err_G_A = accelerator.gather(loss_GAN_AB).sum() / accelerator.state.num_processes
                    err_G_B = accelerator.gather(loss_GAN_BA).sum() / accelerator.state.num_processes                    
            else:
            #'''
                    #print('cB')
                    errD = loss_D.item()
                    errG = loss_G.item()
                    errGAN = loss_GAN.item()
                    errCycle = loss_cycle.item()  
                    errIdentity = loss_identity.item()


                    err_id_A = loss_id_A.item()
                    err_id_B = loss_id_B.item()
                    err_cycle_A = loss_cycle_A.item()
                    err_cycle_B = loss_cycle_B.item()  
                    err_D_A = loss_D_A.item()
                    err_D_B = loss_D_B.item()
                    err_G_A = loss_GAN_AB.item()
                    err_G_B = loss_GAN_BA.item() 
                    
            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = args.num_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()                    
            if accelerator.is_local_main_process:
                    # Print log
                    sys.stdout.write(
                        "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                        % (
                            epoch,
                            args.num_epochs,
                            i,
                            len(dataloader),
                            errD,
                            errG,
                            errGAN,
                            errCycle,
                            errIdentity,
                            #loss_D.item(),
                            #loss_G.item(),
                            #loss_GAN.item(),
                            #loss_cycle.item(),
                            #loss_identity.item(),
                            time_left,
                        )
                    )
                    train_logs = {
                                "epoch": epoch,
                                "total_loss": errG,
                                "loss_discriminator": errD,
                                "loss_generator": errGAN,
                                "loss_cycle": errCycle,
                                "loss_identity": errIdentity,

                                "loss_identity_A": err_id_A,
                                "loss_identity_B": err_id_B,

                                "loss_cycle_A": err_cycle_A,
                                "loss_cycle_B": err_cycle_B,

                                "loss_D_A": err_D_A,
                                "loss_D_B": err_D_B,

                                "loss_G_A": err_G_A,
                                "loss_G_B": err_G_B,                    

                            }

                    #print('d')
                    if args.wandb :
                           wandb.log(train_logs)
                    #print('e')
                    # If at sample interval save image
            if batches_done % args.sample_interval == 0:
                    save_path = sample_images(args, batches_done)
                    if args.wandb and  accelerator.is_local_main_process:
                        try:
                            images = wandb.Image(str(save_path) )
                            wandb.log({'generated_examples': images }  )
                        except:
                            pass
                #print('f')

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), os.path.join(args.output_dir, "saved_models/%s/G_AB_%d.pth" % (args.dataset_name, epoch)) )
            torch.save(G_BA.state_dict(),  os.path.join(args.output_dir,"saved_models/%s/G_BA_%d.pth" % (args.dataset_name, epoch)))
            torch.save(D_A.state_dict(),  os.path.join(args.output_dir,"saved_models/%s/D_A_%d.pth" % (args.dataset_name, epoch)))
            torch.save(D_B.state_dict(),  os.path.join(args.output_dir,"saved_models/%s/D_B_%d.pth" % (args.dataset_name, epoch)))

            '''
            accelerator.wait_for_everyone()
            G_AB_unwrap = accelerator.unwrap_model(G_AB)
            G_AB_unwrap = accelerator.unwrap_model(G_AB)
            G_AB_unwrap = accelerator.unwrap_model(G_AB)
            G_AB_unwrap = accelerator.unwrap_model(G_AB)
            
            accelerator.save(unwrapped_model.state_dict(), filename)
            # Save model checkpoints
            torch.save(G_AB_unwrap.state_dict(), os.path.join(args.output_dir, "saved_models/%s/G_AB_%d.pth" % (args.dataset_name, epoch)) )
            torch.save(G_BA_unwrap.state_dict(),  os.path.join(args.output_dir,"saved_models/%s/G_BA_%d.pth" % (args.dataset_name, epoch)))
            torch.save(D_A_unwrap.state_dict(),  os.path.join(args.output_dir,"saved_models/%s/D_A_%d.pth" % (args.dataset_name, epoch)))
            torch.save(D_B_unwrap.state_dict(),  os.path.join(args.output_dir,"saved_models/%s/D_B_%d.pth" % (args.dataset_name, epoch)))
            
            '''
            
    # Optionally push to hub
    if accelerator.is_main_process and args.push_to_hub:
        save_directory = args.pytorch_dump_folder_path
        if not save_directory.exists():
            save_directory.mkdir(parents=True)

        G_AB.push_to_hub(
            repo_path_or_name=save_directory / args.model_name,
            organization=args.organization_name,
        )

def main():
    args = parse_args()
    print(args)

    # Make directory for saving generated images
    os.makedirs(os.path.join(args.output_dir,"images"), exist_ok=True)

    training_function({}, args)


if __name__ == "__main__":
    main()
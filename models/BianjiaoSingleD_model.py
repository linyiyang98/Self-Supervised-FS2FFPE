import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import torchvision.transforms as transforms
import cv2

class BianjiaoSingleDModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_H', type=float, default=1.0, help='weight for H_loss')
        parser.add_argument('--lambda_texture', type=float, default=1.0, help='weight for texture_loss')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--self_regularization', type=float, default=0.03, help='loss between input and generated image')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--jiaoti_detach',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Use jiaoti_detach")
        parser.add_argument('--img_size', type=int, default=384, help='number of patches per layer')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real_high', 'D_fake_high','D_real_low', 'D_fake_low', 'G', 'NCE']
        self.visual_names = ['real_A_center', 'fake_B_center','real_A_resize', 'fake_B_resize']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        # if opt.nce_idt and self.isTrain:
        #     self.loss_names += ['NCE_Y']
        #     self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.L1_loss = torch.nn.L1Loss().to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data, iter):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_high_loss().backward()                  # calculate gradients for D
            self.compute_D_low_loss().backward()                  # calculate gradients for D
            self.compute_G_loss(iter).backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self, iter):
        # forward
        self.forward()

        # update D_high and D_low
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D_high = self.compute_D_high_loss()
        self.loss_D_low = self.compute_D_low_loss()
        self.loss_D = self.loss_D_high + self.loss_D_low
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss(iter)
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_A_center = input['A_center' if AtoB else 'B_center'].to(self.device)
        self.real_A_resize = input['A_resize' if AtoB else 'B_resize'].to(self.device)

        self.real_B = input['A' if AtoB else 'B'].to(self.device)
        self.real_B_center = input['B_center' if AtoB else 'A_center'].to(self.device)
        self.real_B_resize = input['B_resize' if AtoB else 'A_resize'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        # cv2.imwrite('/data112/linyy/BianjiaoFS2FFPE_tem/real_A.png', util.RGB2BGR(util.tensor2numpy(util.denorm(self.real_A[0])*255)))
        # cv2.imwrite('/data112/linyy/BianjiaoFS2FFPE_tem/real_A_center.png', util.RGB2BGR(util.tensor2numpy(util.denorm(self.real_A_center[0])*255)))
        # cv2.imwrite('/data112/linyy/BianjiaoFS2FFPE_tem/real_A_resize.png', util.RGB2BGR(util.tensor2numpy(util.denorm(self.real_A_resize[0])*255)))
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B_center = self.netG(self.real_A_center)
        self.fake_B_resize = self.netG(self.real_A_resize)

        self.idt_B_center = self.netG(self.real_B_center)
        self.idt_B_resize = self.netG(self.real_B_resize)

    def compute_D_high_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake_high = self.fake_B_center.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake_high)
        self.loss_D_fake_high = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B_center)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real_high = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D_high = (self.loss_D_fake_high + self.loss_D_real_high) * 0.5
        return self.loss_D_high

    def compute_D_low_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake_low = self.fake_B_resize.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake_low)
        self.loss_D_fake_low = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B_resize)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real_low = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D_low = (self.loss_D_fake_low + self.loss_D_real_low) * 0.5
        return self.loss_D_low

    def compute_G_loss(self, iter):
        """Calculate GAN and NCE loss for the generator"""
        fake_high = self.fake_B_center
        fake_low = self.fake_B_resize
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake_high = self.netD(fake_high)
            pred_fake_low = self.netD(fake_low)
            self.loss_G_GAN = (self.criterionGAN(pred_fake_high, True).mean() + self.criterionGAN(pred_fake_low, True).mean()) * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A_resize, self.fake_B_resize)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.lambda_H > 0.0:
            self.loss_H = self.calculate_H_loss(self.real_A_center, self.fake_B_center)
        else:
            self.loss_H = 0.0

        if self.opt.lambda_texture > 0.0:
            if self.opt.jiaoti_detach==False:self.loss_texture = self.calculate_texture_loss(self.fake_B_resize, self.fake_B_center)
            elif self.opt.jiaoti_detach==True:self.loss_texture = self.calculate_texture_loss_jiaotidetach(self.fake_B_resize, self.fake_B_center, iter)

        else:
            self.loss_texture = 0.0

        if self.opt.self_regularization > 0.0:
            self.loss_SR = self.opt.self_regularization * \
                           (self.calculate_SR_loss(self.real_A_center, self.fake_B_center) + self.calculate_SR_loss(self.real_A_resize, self.fake_B_resize))
        else:
            self.loss_SR = 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = (self.calculate_NCE_loss(self.real_B_center, self.idt_B_center) + self.calculate_NCE_loss(self.real_B_resize, self.idt_B_resize))
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_SR + self.loss_H + self.loss_texture
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def calculate_SR_loss(self,src,tgt):
        #rgb_mean_src = torch.mean(src,dim=1) #mean over color channel
        #rgb_mean_tgt = torch.mean(tgt,dim=1)
        diff_chan = src-tgt
        batch_mean = torch.mean(diff_chan,dim=0)
        rgb_sum = torch.sum(batch_mean,0)
        batch_mean2 = torch.mean(torch.mean(rgb_sum,0),0)

        return batch_mean2

    def calculate_H_loss(self, src, tgt):
        src_nucleus = torch.sigmoid(100 * (util.RGB2H_tensor(src, self.device)-0.5))
        tgt_nucleus = torch.sigmoid(100 * (util.RGB2H_tensor(tgt, self.device)-0.5))
        # cv2.imwrite('/data112/linyy/BianjiaoFS2FFPE_tem/src.png', util.RGB2BGR(util.tensor2numpy(util.denorm(src[0])*255)))
        # cv2.imwrite('/data112/linyy/BianjiaoFS2FFPE_tem/tgt.png', util.RGB2BGR(util.tensor2numpy(util.denorm(tgt[0])*255)))
        # cv2.imwrite('/data112/linyy/BianjiaoFS2FFPE_tem/src_nucleus.png', util.RGB2BGR(util.tensor2numpy(util.denorm(src_nucleus[0])*255)))
        # cv2.imwrite('/data112/linyy/BianjiaoFS2FFPE_tem/tgt_nucleus.png', util.RGB2BGR(util.tensor2numpy(util.denorm(tgt_nucleus[0])*255)))
        H_loss = self.L1_loss(src_nucleus, tgt_nucleus) * self.opt.lambda_H
        return H_loss

    def calculate_texture_loss(self, src, tgt):
        origin_size = self.real_A.shape[2]
        n = int((256/origin_size)*256)

        center = transforms.Compose([transforms.CenterCrop(n)])
        resize = transforms.Compose([transforms.Resize(n)])

        texture_loss = self.L1_loss(center(src), resize(tgt)) * self.opt.lambda_texture

        # cv2.imwrite('/data112/linyy/BianjiaoFS2FFPE_tem/center.png', util.RGB2BGR(util.tensor2numpy(util.denorm(center(src)[0])*255)))
        # cv2.imwrite('/data112/linyy/BianjiaoFS2FFPE_tem/resize.png', util.RGB2BGR(util.tensor2numpy(util.denorm(resize(tgt)[0])*255)))

        return texture_loss

    def calculate_texture_loss_jiaotidetach(self, src, tgt, iter):
        origin_size = self.real_A.shape[2]
        n = int((256/origin_size)*256)

        center = transforms.Compose([transforms.CenterCrop(n)])
        resize = transforms.Compose([transforms.Resize(n)])

        if iter%2==0 :
            texture_loss = self.L1_loss(center(src.detach()), resize(tgt)) * self.opt.lambda_texture
        if iter%2==1 :
            texture_loss = self.L1_loss(center(src), resize(tgt.detach())) * self.opt.lambda_texture

        # cv2.imwrite('/data112/linyy/BianjiaoFS2FFPE_tem/center.png', util.RGB2BGR(util.tensor2numpy(util.denorm(center(src)[0])*255)))
        # cv2.imwrite('/data112/linyy/BianjiaoFS2FFPE_tem/resize.png', util.RGB2BGR(util.tensor2numpy(util.denorm(resize(tgt)[0])*255)))

        return texture_loss
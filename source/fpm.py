import torch
import torch.nn as nn

import numpy as np
import scipy as sp
import sys
sys.path.append('./source/')
import utility
import pytorch_complex

mul_c  = pytorch_complex.ComplexMul().apply
div_c  = pytorch_complex.ComplexDiv().apply
abs_c  = pytorch_complex.ComplexAbs().apply
abs2_c = pytorch_complex.ComplexAbs2().apply 
exp_c  = pytorch_complex.ComplexExp().apply
conj  = pytorch_complex.conj

EPS = 1e-7

dtype = torch.float32
np_dtype = np.float32

### Basic Operations
def F(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def iF(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))

class fpm(nn.Module):
    def __init__(self, Np, na, na_illum, na_list, wl, ps, mag, alpha, maxIter, measurements, C_init = None, testFlag = False, device='cpu'):
        super(fpm, self).__init__()

        if C_init is not None:
            self.Nmeas = C_init.shape[0]
            self.Nleds = C_init.shape[1]
            self.C = nn.Parameter(torch.from_numpy(C_init).clone())
        else:
            self.Nleds = na_list.shape[0]
            self.Nmeas = int(self.Nleds//4)
            self.C = nn.Parameter(torch.from_numpy(np.random.rand(self.Nmeas,self.Nleds).astype(np_dtype)))
            
        if testFlag is True:
            self.C.requires_grad = False
    
        self.Np = Np
        self.na = na
        self.na_illum = na_illum
        self.na_list = na_list
        self.wl = wl
        self.ps = ps/mag
        self.mag = mag
        self.alpha = alpha
        self.maxIter = maxIter
        
        # setup sampling (note, might need to put some of this in torch and then gpu)
        self.xx_recon, self.yy_recon, self.uxx_recon, self.uyy_recon = self.setup_sampling()
        
        # generate pupil (for measurements)
        P_tmp = self.genPupil(self.wl,self.na)
        P_tmp = np.fft.ifftshift(P_tmp)
        self.P = torch.stack((torch.from_numpy(P_tmp.real),torch.from_numpy(P_tmp.imag)),dim=2)
        
        # setup illumination (for measurements)
        pupils_tmp, planewaves_tmp, illum_spec_single_tmp = self.setup_illumination()
        planewaves_tmp_real = planewaves_tmp.real.astype(np_dtype)
        planewaves_tmp_imag = planewaves_tmp.imag.astype(np_dtype)
        self.planewaves = torch.stack((torch.from_numpy(planewaves_tmp_real),torch.from_numpy(planewaves_tmp_imag)),dim=len(planewaves_tmp.shape))
        
        pupils_tmp_real = pupils_tmp.real.astype(np_dtype)
        pupils_tmp_imag = pupils_tmp.imag.astype(np_dtype)
        self.pupils = torch.stack((torch.from_numpy(pupils_tmp_real),torch.from_numpy(pupils_tmp_imag)),dim=len(pupils_tmp.shape))
        
        self.generateCrop()
        
        self.measurements = measurements

        self.device = device
    
    
    def step(self, x, device='cpu'):
        g = self.grad(x,device=device)
        return -1 * self.alpha * g
    
    def forward(self, x, device='cpu'):
        return x + self.step(x,device=device)

    def grad(self, field_est, device='cpu'):
        self.measurements = self.measurements.to(self.device)
        self.pupils = self.pupils.to(self.device)
        self.planewaves = self.planewaves.to(self.device)
        self.P = self.P.to(self.device)
        
        multiMeas = torch.matmul(self.measurements.permute(1,2,0),self.C.permute(1,0)).permute(2,0,1)
        multiMeas = torch.abs(multiMeas)
        
        # simulate current estimate of measurements
        y = self.generateMultiMeas(field_est,device=device)

        # compute residual
        sqrty = torch.sqrt(y + EPS)
        residual = sqrty - torch.sqrt(multiMeas + EPS)
        cost = torch.sum(torch.pow(residual,2)).detach()
        Ajx = residual/(sqrty + 1e-10)
        Ajx_c = torch.stack((Ajx,torch.zeros_like(Ajx)),dim=len(Ajx.shape))
        
        # compute gradient
        output = mul_c(self.planewaves,field_est)
        output = torch.fft(output,2)
        output = mul_c(self.P,output)
        output = torch.ifft(output,2)
        
        g = field_est*0.
        for meas_index in range(self.Nmeas):
            output2 = mul_c(Ajx_c[meas_index,...],output)
            output2 = mul_c(conj(self.planewaves),output2)
            output2 = torch.fft(output2,2)
            output2 = mul_c(self.pupils,output2)
            output2 = torch.ifft(output2,2)
            g_tmp = torch.matmul(output2.permute(1,2,3,0),self.C[meas_index,:])
            g = g + g_tmp
#         return -1 * self.alpha * g, cost
        return g
    
    def generateSingleMeas(self,field,device="cpu"):
        output = mul_c(self.planewaves,field)
        output = torch.fft(output,2)
        output = mul_c(self.P, output)
        output = torch.ifft(output,2)
        output = abs2_c(output)
        return output
    
    def generateMultiMeas(self,field,device="cpu"):
        output = mul_c(self.planewaves,field)
        output = torch.fft(output,2)
        output = mul_c(self.P, output)
        output = torch.ifft(output,2)
        output = abs2_c(output)
        multiMeas = torch.matmul(output.permute(1,2,0),self.C.permute(1,0)).permute(2,0,1)
        return multiMeas
    
    def generateCrop(self, ):
        # only really need on-axis crop for effective camera pixel size.
        self.Np_meas = [int(np.ceil(a/self.ps)) for a in self.FOV]
        
        cropxstart = self.Np[0]//2 - self.Np_meas[0]//2
        cropxstop = cropxstart + self.Np_meas[0]
        cropystart = self.Np[1]//2 - self.Np_meas[1]//2
        cropystop = cropystart + self.Np_meas[1]
        self.crops = [cropxstart,cropxstop,cropystart,cropystop]
        
        self.scaling = self.Np[0]*self.Np[1]/self.Np_meas[0]/self.Np_meas[1]

    def cropMeasurements(self, measurements):
        # cropping
        measurements_cropped = torch.zeros(measurements.shape[0],self.Np_meas[0],self.Np_meas[1],2)
        for img_idx in range(measurements.shape[0]):
            fmeas = utility.fftshift2(torch.fft(measurements[img_idx,...],2))
            tmp = fmeas[self.crops[0]:self.crops[1],self.crops[2]:self.crops[3],:]
            measurements_cropped[img_idx,...] = (1/self.scaling) * torch.ifft(utility.ifftshift2(tmp),2)
        return measurements_cropped
    
    def padMeasurements(self, measurements):
        # padding
        measurements_padded = torch.zeros(measurements.shape[0],self.Np[0],self.Np[1],2)
        for ii in range(measurements.shape[0]):
            fmeas = utility.fftshift2(torch.fft(measurements[ii,...],2))
            measurements_padded[ii,self.crops[0]:self.crops[1],self.crops[2]:self.crops[3],:] = fmeas
            measurements_padded[ii,...] = (self.scaling) * torch.ifft(utility.ifftshift2(measurements_padded[ii,...]),2)
        return measurements_padded
    
    def makeField(self,amp,phase,device="cpu"):
        comp = amp * np.exp(1j * phase)
        comp_real_torch = torch.from_numpy(comp.real)
        comp_imag_torch = torch.from_numpy(comp.imag)
        return torch.stack((comp_real_torch,comp_imag_torch),dim=2).to(device)
    
    def setup_sampling(self,):
        self.na_recon = self.na + self.na_illum
        self.ps_recon = self.wl/(2*self.na_recon)
        print('Reconstruction\'s pixel size (um): ' +  str(self.ps_recon))
        print('System\'s pixel size limit (um): ' +  str(self.wl/(2*self.na)))
        print('Camera\'s effective pixel size (um): ' +  str(self.ps))
        self.FOV = [a*self.ps_recon for a in self.Np]
        
        # real space sampling
        x = (np.linspace(0,self.Np[0]-1,self.Np[0]) - int(self.Np[0]//2))*self.ps_recon
        y = (np.linspace(0,self.Np[1]-1,self.Np[1]) - int(self.Np[1]//2))*self.ps_recon
        xx,yy = np.meshgrid(x,y)

        # spatial frequency sampling
        ux = np.linspace(-1/(2*self.ps_recon),1/(2*self.ps_recon), self.Np[0])
        uy = np.linspace(-1/(2*self.ps_recon),1/(2*self.ps_recon), self.Np[1])
        uxx,uyy = np.meshgrid(ux,uy)
        return xx,yy,uxx,uyy
        
    def genPupil(self,wl,na):
        urr = np.sqrt(self.uxx_recon**2 + self.uyy_recon**2)
        pupil = 1. * (urr**2 < (na/wl)**2)
        return pupil.astype(np_dtype)
        
    def setup_illumination(self,):
        illum_spec_single = np.zeros((self.Nleds,self.Np[0],self.Np[1]))
        planewaves = np.zeros((self.Nleds,self.Np[0],self.Np[1]),dtype='complex')
        for ii in range(self.Nleds):
            kspx = int(np.argmin(np.abs(self.uxx_recon[0,:]-self.na_list[ii,0]/self.wl)))
            kspy = int(np.argmin(np.abs(self.uyy_recon[:,0]-self.na_list[ii,1]/self.wl)))
            illum_spec_single[ii,kspx,kspy] = 1
            planewaves[ii,:,:] = F(illum_spec_single[ii,:,:])
            
        pupils = np.zeros((self.Nleds,self.Np[0],self.Np[1]),dtype='float32')
        Pmeas = self.genPupil(self.wl,self.na).astype(np_dtype)
        for ii in range(self.Nleds):
            pupils[ii,:,:] = np.fft.ifftshift(np.abs(F(np.conj(planewaves[ii,:,:]) * iF(Pmeas))))
        return pupils,planewaves,illum_spec_single
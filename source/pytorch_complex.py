import torch

def conj(complex_tensor):
    complex_conj_tensor          = complex_tensor.clone()
    complex_conj_tensor[..., 1] *= -1.0 
    return complex_conj_tensor

def angle(complex_tensor):
    return torch.atan2(complex_tensor[..., 1], complex_tensor[..., 0])

def multiply_complex(complex_tensor1, complex_tensor2):
    complex_tensor_mul_real = complex_tensor1[..., 0]*complex_tensor2[..., 0] -\
                              complex_tensor1[..., 1]*complex_tensor2[..., 1]
    complex_tensor_mul_imag = complex_tensor1[..., 0]*complex_tensor2[..., 1] +\
                              complex_tensor1[..., 1]*complex_tensor2[..., 0]
    return torch.stack((complex_tensor_mul_real, complex_tensor_mul_imag), dim=len(complex_tensor_mul_real.shape))

def division_complex(complex_tensor1, complex_tensor2):
    denominator             = (complex_tensor2**2).sum(-1)
    complex_tensor_mul_real = (complex_tensor1[..., 0]*complex_tensor2[..., 0] + complex_tensor1[..., 1]*complex_tensor2[..., 1])/denominator
    complex_tensor_mul_imag = (complex_tensor1[..., 1]*complex_tensor2[..., 0] - complex_tensor1[..., 0]*complex_tensor2[..., 1])/denominator
    return torch.stack((complex_tensor_mul_real, complex_tensor_mul_imag), dim=len(complex_tensor_mul_real.shape))        
        
class ComplexMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        assert input1.shape[-1]==2, "Complex tensor should have real and imaginary parts."
        assert input2.shape[-1]==2, "Complex tensor should have real and imaginary parts."
        output = multiply_complex(input1, input2)
        
        ctx.save_for_backward(input1, input2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        grad_input1    = multiply_complex(conj(input2), grad_output)
        grad_input2    = multiply_complex(conj(input1), grad_output)
        if len(input1.shape)>len(input2.shape):
            grad_input2 = grad_input2.sum(0)
        elif len(input1.shape)<len(input2.shape):
            grad_input1 = grad_input1.sum(0)            
            
        return grad_input1, grad_input2

class ComplexDiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        assert input1.shape[-1]==2, "Complex tensor should have real and imaginary parts."
        assert input2.shape[-1]==2, "Complex tensor should have real and imaginary parts."
        output = division_complex(input1, input2)
        
        ctx.save_for_backward(input1, input2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2       = ctx.saved_tensors
        denominator          = (input2**2).sum(-1)
        grad_input1          = input2.clone()
        grad_input1[..., 0] /= denominator
        grad_input1[..., 1] /= denominator
        grad_input1          = multiply_complex(grad_input1, grad_output)
        grad_input2          = -1*conj(division_complex(input1, multiply_complex(input2, input2)))
        grad_input2          = multiply_complex(grad_input2, grad_output)

        if len(input1.shape)>len(input2.shape):
            grad_input2 = grad_input2.sum(0)
        elif len(input1.shape)<len(input2.shape):
            grad_input1 = grad_input1.sum(0)            
            
        return grad_input1, grad_input2
    
class ComplexAbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        assert input.shape[-1]==2, "Complex tensor should have real and imaginary parts."
        output         = ((input**2).sum(-1))**0.5
        
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input,         = ctx.saved_tensors
        grad_input     = torch.stack((grad_output, torch.zeros_like(grad_output)), dim=len(grad_output.shape))
        phase_input    = angle(input)
        phase_input    = torch.stack((torch.cos(phase_input), torch.sin(phase_input)), dim=len(grad_output.shape))
        grad_input     = multiply_complex(phase_input, grad_input)
        
        return 0.5*grad_input
    
class ComplexAbs2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        assert input.shape[-1]==2, "Complex tensor should have real and imaginary parts."
        output         = multiply_complex(conj(input), input)
        
        ctx.save_for_backward(input)
        return output[..., 0]

    @staticmethod
    def backward(ctx, grad_output):
        input,         = ctx.saved_tensors
        grad_output_c  = torch.stack((grad_output, torch.zeros_like(grad_output)), dim=len(grad_output.shape))
        grad_input     = multiply_complex(input, grad_output_c)
        
        return grad_input
    
class ComplexExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        assert input.shape[-1]==2, "Complex tensor should have real and imaginary parts."
        output         = input.clone()
        amplitude      = torch.exp(input[..., 0])
        # amplitude      = input[..., 0]
        output[..., 0] = amplitude*torch.cos(input[..., 1])
        output[..., 1] = amplitude*torch.sin(input[..., 1])
        
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output,        = ctx.saved_tensors
        grad_input     = multiply_complex(conj(output), grad_output)
        
        return grad_input
    
class ComplexTensor:
    def __init__(self, real, imag):
        assert type(real)==torch.Tensor and type(imag)==torch.Tensor, "ComplexTensor is based on Tensor in PyTorch."
        assert real.shape==imag.shape, "Real and imaginary parts should have same shape."
        self.real    = real
        self.imag    = imag
        self.complex = torch.stack((real, imag), dim=len(real.shape))
        self.shape   = real.shape
    
    def amplitude(self, ):
        return (self.real.detach()**2+self.imag.detach()**2)**0.5
    
    def phase(self, ):
        return angle(self.complex).detach()
    
    def real_part(self, ):
        return self.real.detach()
    
    def imag_part(self, ):
        return self.imag.detach()
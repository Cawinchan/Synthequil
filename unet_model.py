from random import sample
import torch
import torch.nn as nn
import copy
import gc

class UNet(nn.Module):

    def __init__(self, feature_count_list, kernel_size, activation_type, instruments, sample_block_depth=1, bottleneck_depth=1):
        super().__init__()

        self.feature_count_list = copy.deepcopy(feature_count_list)
        self.kernel_size = kernel_size
        self.activation_type = activation_type
        self.instruments = instruments
        self.sample_block_depth = sample_block_depth
        self.bottleneck_depth = bottleneck_depth

        self.models = nn.ModuleDict()
        for i in instruments:
            self.models[i] = Basic_UNet(feature_count_list,kernel_size,activation_type,sample_block_depth,bottleneck_depth)
        
    def forward(self,input,instrument):
        return self.models[instrument](input)

class Basic_UNet(nn.Module):

    def __init__(self, feature_count_list, kernel_size, activation_type, sample_block_depth=1, bottleneck_depth=1):
        super().__init__()
        
        self.feature_count_list = copy.deepcopy(feature_count_list)
        self.kernel_size = kernel_size
        self.activation_type = activation_type
        self.sample_block_depth = sample_block_depth
        self.bottleneck_depth = bottleneck_depth
        
        self.downsampling_blocks = nn.ModuleList(
            [Downsampling_Block(feature_count_list[i],feature_count_list[i+1],kernel_size,
                activation_type,sample_block_depth) for i in range(len(feature_count_list)-1)]
        )
        self.bottleneck_blocks = nn.ModuleList(
            [Conv1D_Block_With_Activation(feature_count_list[-1],feature_count_list[-1],1,activation_type) for i in range(bottleneck_depth)]
        )
        self.upsampling_blocks = nn.ModuleList(
            [Upsampling_Block(feature_count_list[i],feature_count_list[i-1],kernel_size,
                activation_type,sample_block_depth) for i in range(len(feature_count_list)-1,0,-1)]
        )
        self.output_block = Conv1D_Block_With_Activation(feature_count_list[0],feature_count_list[0],1,"tanh")
    
    def forward(self,input):
        shortcuts = []
        
        intermediate = input
        for i in self.downsampling_blocks:
            intermediate, shortcut = i(intermediate)
            shortcuts.append(shortcut)
        
        for i in self.bottleneck_blocks:
            intermediate = i(intermediate)
        
        output = intermediate
        for i in self.upsampling_blocks:
            shortcut = shortcuts.pop()
            output= i(output,shortcut)

        output = self.output_block(output)
        
        return output

class Downsampling_Block(nn.Module):
    
    def __init__(self, num_input_features, num_output_features, kernel_size, activation_type, depth=1):
        super().__init__()

        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.kernel_size = kernel_size
        self.activation_type = activation_type
        self.depth = depth

        self.num_intermediate_features = self.num_output_features

        self.preshortcut_block = []
        for i in range(depth):
            self.preshortcut_block.append(Conv1D_Block_With_Activation(num_input_features if i==0 else self.num_intermediate_features,
                self.num_intermediate_features,kernel_size,activation_type,stride=kernel_size//2))
        self.preshortcut_block = nn.ModuleList(self.preshortcut_block)
        
        self.postshortcut_block = []
        for i in range(depth):
            self.postshortcut_block.append(Conv1D_Block_With_Activation(self.num_intermediate_features if i==0 else num_output_features,
                num_output_features,kernel_size,activation_type,stride=kernel_size//2))
        self.postshortcut_block = nn.ModuleList(self.postshortcut_block)
    
    def forward(self, input):
        shortcut = input
        for i in self.preshortcut_block:
            shortcut = i(shortcut)

        output = shortcut
        for i in self.postshortcut_block:
            output = i(output)
        return output, shortcut

class Upsampling_Block(nn.Module):

    def __init__(self, num_input_features, num_output_features, kernel_size, activation_type, depth=1):
        super().__init__()

        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.kernel_size = kernel_size
        self.activation_type = activation_type
        self.depth = depth

        self.num_intermediate_features = self.num_input_features

        self.preshortcut_block = []
        for i in range(depth):
            self.preshortcut_block.append(Conv1D_Block_With_Activation(num_input_features if i==0 else self.num_intermediate_features,
                self.num_intermediate_features,kernel_size,activation_type,transpose=True,stride=kernel_size//2))
        self.preshortcut_block = nn.ModuleList(self.preshortcut_block)
        
        self.shortcut_in_block = Conv1D_Block_With_Activation(self.num_intermediate_features,self.num_intermediate_features,1,"tanh")
        
        self.postshortcut_block = []
        for i in range(depth):
            self.postshortcut_block.append(Conv1D_Block_With_Activation(self.num_intermediate_features*2 if i==0 else num_output_features,
                num_output_features,kernel_size,activation_type,transpose=True,stride=kernel_size//2))
        self.postshortcut_block = nn.ModuleList(self.postshortcut_block)

    def forward(self, input, input_shortcut):
        
        intermediate = input
        for i in self.preshortcut_block:
            intermediate = i(intermediate)
        
        shortcut_filter = self.shortcut_in_block(input_shortcut)
        shortcut = torch.mul(input_shortcut,shortcut_filter)

        output = torch.cat([intermediate,shortcut],dim=-2)
        for i in self.postshortcut_block:
            output = i(output)
        
        return output

# Conv1D (or Conv1DTranspose) + Activation
# Available norms: Leaky ReLU, GELU, Tanh
# Padding set to 0 and stride to 1
class Conv1D_Block_With_Activation(nn.Module):
    
    def __init__(self, num_input_features, num_output_features, kernel_size, activation_type, transpose=False, stride=1):
        super().__init__()
        
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(num_input_features,num_output_features,
            kernel_size,stride=stride) if not transpose else nn.ConvTranspose1d(num_input_features,num_output_features,kernel_size,stride=stride)

        assert activation_type in ("leaky_relu", "gelu", "tanh")
        self.activation_type = activation_type
        if activation_type=="leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation_type=="gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()

        self.norm = nn.BatchNorm1d(num_output_features)
        
    def forward(self, input):
        conv_output = self.conv(input)
        if True in torch.isnan(conv_output):
            print(self.num_input_features,self.num_output_features,input,conv_output,torch.mean(torch.mul(input,input)))
            with open("error.txt","w") as f:
                f.write(str(input.cpu().numpy().tolist()))
            raise Exception("Error: nan value found in convolution block: conv")
        norm_output = self.norm(conv_output)
        if True in torch.isnan(conv_output):
            print(self.num_input_features,self.num_output_features,conv_output,norm_output,torch.mean(torch.mul(conv_output,conv_output)))
            with open("error.txt","w") as f:
                f.write(str(conv_output.cpu().numpy().tolist()))
            raise Exception("Error: nan value found in convolution block: norm")
        output = self.activation(norm_output)
        if True in torch.isnan(output):
            print(self.num_input_features,self.num_output_features,norm_output,output,torch.mean(torch.mul(norm_output,norm_output)))
            with open("error.txt","w") as f:
                f.write(str(norm_output.cpu().numpy().tolist()))
            raise Exception("Error: nan value found in convolution block: activation")
        return output
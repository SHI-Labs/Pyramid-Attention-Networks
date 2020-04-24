from model import common
import torch.nn as nn
import torch
from model.attention import ContextualAttention,NonLocalAttention
def make_model(args, parent=False):
    return MSSR(args)

class MultisourceProjection(nn.Module):
    def __init__(self, in_channel,kernel_size = 3, conv=common.default_conv):
        super(MultisourceProjection, self).__init__()
        self.up_attention = ContextualAttention(scale=2)
        self.down_attention = NonLocalAttention()
        self.upsample = nn.Sequential(*[nn.ConvTranspose2d(in_channel,in_channel,6,stride=2,padding=2),nn.PReLU()])
        self.encoder = common.ResBlock(conv, in_channel, kernel_size, act=nn.PReLU(), res_scale=1)
    
    def forward(self,x):
        down_map = self.upsample(self.down_attention(x))
        up_map = self.up_attention(x)

        err = self.encoder(up_map-down_map)
        final_map = down_map + err
        
        return final_map

class RecurrentProjection(nn.Module):
    def __init__(self, in_channel,kernel_size = 3, conv=common.default_conv):
        super(RecurrentProjection, self).__init__()
        self.multi_source_projection_1 = MultisourceProjection(in_channel,kernel_size=kernel_size,conv=conv)
        self.multi_source_projection_2 = MultisourceProjection(in_channel,kernel_size=kernel_size,conv=conv)
        self.down_sample_1 = nn.Sequential(*[nn.Conv2d(in_channel,in_channel,6,stride=2,padding=2),nn.PReLU()])
	#self.down_sample_2 = nn.Sequential(*[nn.Conv2d(in_channel,in_channel,6,stride=2,padding=2),nn.PReLU()])
        self.down_sample_3 = nn.Sequential(*[nn.Conv2d(in_channel,in_channel,8,stride=4,padding=2),nn.PReLU()])
        self.down_sample_4 = nn.Sequential(*[nn.Conv2d(in_channel,in_channel,8,stride=4,padding=2),nn.PReLU()])
        self.error_encode_1 = nn.Sequential(*[nn.ConvTranspose2d(in_channel,in_channel,6,stride=2,padding=2),nn.PReLU()])
        self.error_encode_2 = nn.Sequential(*[nn.ConvTranspose2d(in_channel,in_channel,8,stride=4,padding=2),nn.PReLU()])
        self.post_conv = common.BasicBlock(conv,in_channel,in_channel,kernel_size,stride=1,bias=True,act=nn.PReLU())


    def forward(self, x):
        x_up = self.multi_source_projection_1(x)

        x_down = self.down_sample_1(x_up)
        error_up = self.error_encode_1(x-x_down)
        h_estimate_1 = x_up + error_up
	
        x_up_2 = self.multi_source_projection_2(h_estimate_1)
        x_down_2 = self.down_sample_3(x_up_2)
        error_up_2 = self.error_encode_2(x-x_down_2)
        h_estimate_2 = x_up_2 + error_up_2
        x_final = self.post_conv(self.down_sample_4(h_estimate_2))

        return x_final, h_estimate_2
        

        


class MSSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MSSR, self).__init__()

        #n_convblock = args.n_convblocks
        n_feats = args.n_feats
        self.depth = args.depth
        kernel_size = 3 
        scale = args.scale[0]
        

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        m_head = [common.BasicBlock(conv, args.n_colors, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.PReLU()),
        common.BasicBlock(conv,n_feats, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.PReLU())]

        # define multiple reconstruction module
        
        self.body = RecurrentProjection(n_feats)


        # define tail module
        m_tail = [
            nn.Conv2d(
                n_feats*self.depth, args.n_colors, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)
    def forward(self,input):
        x = self.sub_mean(input)
        x = self.head(x)
        bag = []
        for i in range(self.depth):
            x, h_estimate = self.body(x)
            bag.append(h_estimate)
        h_feature = torch.cat(bag,dim=1)
        h_final = self.tail(h_feature)
        
        return self.add_mean(h_final)

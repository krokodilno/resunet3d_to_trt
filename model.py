import torch
import torch.nn as nn
from torch2trt import torch2trt as t2t

bn_track = True

class ResUnet3d(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv0 = nn.Sequential(
            nn.Sequential(
              nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
              nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
              nn.PReLU(num_parameters=1)
            ),
            nn.Sequential(
              nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
              nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
              nn.PReLU(num_parameters=1)
            ),
            nn.Sequential(
              nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
              nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
              nn.PReLU(num_parameters=1)
            ),
            nn.Sequential(
              nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
              nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
              nn.PReLU(num_parameters=1)
            )
        )
        
        self.res0 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        
        self.conv1 = nn.Sequential(
            nn.Sequential(
              nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
              nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
              nn.PReLU(num_parameters=1)
            ),
            nn.Sequential(
              nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
              nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
              nn.PReLU(num_parameters=1)
            ),
            nn.Sequential(
              nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
              nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
              nn.PReLU(num_parameters=1)
            ),
            nn.Sequential(
              nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
              nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
              nn.PReLU(num_parameters=1)
            )
          )
        
        self.res1 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        
        self.conv2 = nn.Sequential(
                nn.Sequential(
                  nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
                  nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
                  nn.PReLU(num_parameters=1)
                ),
                nn.Sequential(
                  nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                  nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
                  nn.PReLU(num_parameters=1)
                ),
                nn.Sequential(
                  nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                  nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
                  nn.PReLU(num_parameters=1)
                ),
                nn.Sequential(
                  nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                  nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
                  nn.PReLU(num_parameters=1)
                )
              )
        
        self.res2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
            nn.PReLU(num_parameters=1),

            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
            nn.PReLU(num_parameters=1),

            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
            nn.PReLU(num_parameters=1),

            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
            nn.PReLU(num_parameters=1)
        )
        
        self.res3 = nn.Conv3d(128, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        
        self.deconv3_0 = nn.Sequential(
            nn.ConvTranspose3d(384, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
            nn.PReLU(num_parameters=1)
        )

        self.deconv3_1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
            nn.PReLU(num_parameters=1)
        )
        
        self.deconv2_0 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
            nn.PReLU(num_parameters=1),
        )
        
        self.deconv2_1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
            nn.PReLU(num_parameters=1)
        )
        
        self.deconv1_0 = nn.Sequential(
            nn.ConvTranspose3d(64, 6, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=bn_track),
            nn.PReLU(num_parameters=1)
        )
        self.deconv1_1 = nn.Conv3d(6, 6, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
    def forward(self, x):
        
        cx0 = self.conv0(x)
        rx0 = self.res0(x)
        l0 = torch.add(cx0, rx0)
        
        cx1 = self.conv1(l0)
        rx1 = self.res1(l0)
        l1 = torch.add(cx1, rx1)
        
        cx2 = self.conv2(l1)
        rx2 = self.res2(l1)
        l2 = torch.add(cx2, rx2)
        
        cx3 = self.conv3(l2)
        rx3 = self.res3(l2)
        l3 = torch.add(cx3, rx3)
        
        l3d_c = torch.cat((l2, l3), dim=1)
        l3d_0 = self.deconv3_0(l3d_c)
        l3d_1 = self.deconv3_1(l3d_0)
        l3d = torch.add(l3d_0, l3d_1)
        
        l2d_c = torch.cat((l1, l3d), dim=1)
        l2d_0 = self.deconv2_0(l2d_c)
        l2d_1 = self.deconv2_1(l2d_0)
        l2d = torch.add(l2d_0, l2d_1)
        
        l1d_c = torch.cat((l0, l2d), dim=1)
        l1d_0 = self.deconv1_0(l1d_c)
        l1d_1 = self.deconv1_1(l1d_0)
        l1d = torch.add(l1d_0, l1d_1)
        
        return l1d


# CHECK
net = ResUnet3d().cuda()
net.eval()

input = torch.rand(1, 1, 128, 128, 128).cuda()

with torch.no_grad():
    a = net(input)
    print(a.shape)
    
# CONVERTION

net_trt = t2t(net, [input])
torch.save(net_trt.state_dict(), 'net3D-trt.pt')

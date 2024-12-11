from utils_libs import *
import torchvision.models as models
from tiny_vit import tiny_vit_21m_224
from vits import VITs

class client_model(nn.Module):
    def __init__(self, name, pretrained, args=True):
        super(client_model, self).__init__()
        self.name = name
        
        if self.name == 'vit-small':
            self.n_cls = 100
            self.model = VITs()
        
        if self.name == 'tiny_vit':
            self.n_cls = 100 # IN-100
            # Load pretrained weights from model trained on IN-1k
            # self.model = tiny_vit_21m_224(pretrained=True, num_classes=self.n_cls, pretrained_type='1k')
            # Train from scratch
            self.model = tiny_vit_21m_224(pretrained=pretrained, num_classes=self.n_cls, pretrained_type='1k', img_size=args.image_size)
        
        if self.name == 'Linear':
            [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(self.n_dim, self.n_out)
          
        if self.name == 'mnist':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, self.n_cls)
            
        if self.name == 'emnist':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, self.n_cls)
        
        if self.name == 'cifar10' or self.name == 'cifar10c':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) 
            self.fc3 = nn.Linear(192, self.n_cls)
            
        if self.name == 'cifar100':
            self.n_cls = 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) 
            self.fc3 = nn.Linear(192, self.n_cls)
        
        if self.name=='ConvNet':
            self.n_cls = 200
            self.model = ConvNet(channel=3,num_classes=200,net_width=64,net_depth=3,net_act='relu',
                                 net_norm='groupnorm',net_pooling='maxpooling',im_size=(64,64))
            
            self.fc1 = nn.Linear(4096,512)
            self.fc2 = nn.Linear(512, 384)
            self.fc3 = nn.Linear(384, self.n_cls)
            
        if self.name == 'Resnet18':
            resnet18 = models.resnet18()
            resnet18.avgpool = nn.AdaptiveAvgPool2d(1)
            num_filters = resnet18.fc.in_features
            #print("num_filters:",num_filters)
            resnet18.fc = nn.Linear(num_filters, 200)
            resnet18.conv1 = nn.Conv2d(3,64,kernel_size = (3,3),stride =(1,1),padding = (1,1))
            resnet18.maxpool = nn.Sequential()
            # Change BN to GN 
            resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'
            self.model = resnet18


        if self.name == 'shakespeare':
            embedding_dim = 8
            hidden_size = 100
            num_LSTM = 2
            input_length = 80
            self.n_cls = 80
            
            self.embedding = nn.Embedding(input_length, embedding_dim)
            self.stacked_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_LSTM)
            self.fc = nn.Linear(hidden_size, self.n_cls)
              
        
    def forward(self, x):
        if self.name == 'tiny_vit':
            x = self.model(x)
            
        if self.name == 'vit-small':
            x = self.model(x)
        
        if self.name == 'Linear':
            x = self.fc(x)
            
        if self.name == 'mnist':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
  
        if self.name == 'emnist':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        
        if self.name == 'cifar10' or self.name=='cifar10c':
            x = self.conv1(x)
           
            
            x = F.relu(x)

            x = self.pool(x)
            #x = F.relu(self.conv2(x))
            x = self.conv2(x)
        
            x = F.relu(x)

            x = self.pool(x)
            
            x = x.view(-1, 64*5*5)
            #x = F.relu(self.fc1(x))
            x = self.fc1(x)
        
            x = F.relu(x)

            #x = F.relu(self.fc2(x))
            x = self.fc2(x)

            x = F.relu(x)

            x = self.fc3(x)
            

        if self.name == 'cifar100':

            #x = F.relu(self.conv1(x))
            x = self.conv1(x)
            
            x = F.relu(x)

            x = self.pool(x)
            #x = F.relu(self.conv2(x))
            x = self.conv2(x)
        
            x = F.relu(x)

            x = self.pool(x)
            
            x = x.view(-1, 64*5*5)
            #x = F.relu(self.fc1(x))
            x = self.fc1(x)
            x = F.relu(x)

            #x = F.relu(self.fc2(x))
            x = self.fc2(x)
            x = F.relu(x)

            x = self.fc3(x)
            
        if self.name=='ConvNet':
            x = self.model(x)
            #print("x:",x.shape) 
            x = x.view((x.shape[0], -1))

            x = F.relu(self.fc1(x))

            x = F.relu(self.fc2(x))

            x = self.fc3(x)
               
        if self.name == 'Resnet18':
            x = self.model(x)

        if self.name == 'shakespeare':
            x = self.embedding(x)
            x = x.permute(1, 0, 2) # lstm accepts in this style
            output, (h_, c_) = self.stacked_LSTM(x)
            # Choose last hidden layer
            last_hidden = output[-1,:,:]
            x = self.fc(last_hidden)

        return x
    
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, 
                    net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        #out = out.view(out.size(0), -1)
        #out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        # elif net_act == 'swish':
        #     return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat
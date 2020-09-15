import torch.nn as nn
import torch
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
        """
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 3, padding = 1), nn.LeakyReLU(0.2),
                                     nn.Conv2d(32, 32, 3, padding = 1), nn.LeakyReLU(0.2),
                                     nn.MaxPool2d(2, 2),
                                     nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU(0.2),
                                     nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2),
                                     nn.MaxPool2d(2, 2),
                                     nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
                                     nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
                                     nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
                                     nn.MaxPool2d(2, 2),
                                     nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.2),
                                     nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
                                     nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
                                     nn.MaxPool2d(2, 2),
                                     nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
                                     nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
                                     nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
                                     nn.MaxPool2d(2, 2),
                                     )
                                     """
        """
        self.convnet = nn.Sequential(
            # 3 224 128
            nn.Conv2d(3, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 64 112 64
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 128 56 32
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 256 28 16
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 512 14 8
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.avg_pool = nn.AvgPool2d(7)
        #self.classifier = nn.Linear(512, 10)
        self.classifier = nn.Sequential(nn.Linear(512, 1000),
                            nn.PReLU(),
                            nn.Linear(1000, 100),
                            nn.PReLU(),
                            nn.Linear(100, 2)
                            )        
        """
        self.fc = nn.Sequential(#nn.AvgPool2d(7),
                                #nn.Linear(512, 100),
                                #nn.PReLU(),
                                #nn.Linear(100, 2)
                                #nn.Linear(256 * 14 * 14, 256),
                                #nn.PReLU(),
                                #nn.Linear(256, 256),
                                #nn.PReLU(),
                                nn.Linear(1000, 2)
                                #nn.LeakyReLU(),
                                #nn.Linear(10,2)
                                )

    # 512 1 1

    def forward(self, x):
        #output = self.convnet(x)
        #x = self.avg_pool(output)
        #x = x.view(output.size(0), -1)
        #x = self.classifier(x)
        output = self.model(x)
        #output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

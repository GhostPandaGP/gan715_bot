from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import copy


PATH = 'models/vgg11.pth'


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # это константа. Убираем ее из дерева вычеслений
        self.loss = F.mse_loss(self.target, self.target)  # to initialize with something

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    batch_size, h, w, f_map_num = input.size()
    features = input.view(batch_size * h, w * f_map_num)
    G = torch.mm(features, features.t())
    return G.div(batch_size * h * w * f_map_num)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)  # to initialize with something

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class SlowClass:
    def __init__(self):
        self.img_size = 115
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unloader = transforms.ToPILImage()

        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        self.content_layers_default = ['conv_4']
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        self.output = None

    # helps functions

    def image_loader(self, image_name, loader):
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def imshow(self, tensor, title=None):
        image = tensor.cpu().clone()
        image = image.squeeze(0)  # функция для отрисовки изображения
        image = self.unloader(image)
        return image
        # plt.imshow(image)
        # if title is not None:
        #     plt.title(title)
        # plt.pause(0.001)

        # main functions

    def load_images(self, path1, path2):

        loader = transforms.Compose([
            transforms.Resize(self.img_size),  # нормируем размер изображения
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor()])  # превращаем в удобный формат

        style_img = self.image_loader(path1, loader)
        content_img = self.image_loader(path2, loader)

        return [style_img, content_img]

    def get_style_model_and_losses(self, cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=None,
                                   style_layers=None):
        if style_layers is None:
            style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        if content_layers is None:
            content_layers = ['conv_4']

        cnn = copy.deepcopy(cnn)

        normalization = Normalization(normalization_mean, normalization_std).to(self.device)

        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    @staticmethod
    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def run_style_transfer(self, cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=100,
                           style_weight=100000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses(cnn,
                                                                              normalization_mean, normalization_std,
                                                                              style_img, content_img)
        optimizer = self.get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values
                # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                # взвешивание ощибки
                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img

    def run(self, path1, path2):
        style_img, content_img = self.load_images(path1, path2)
        input_img = content_img.clone()

        # cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        cnn = torch.load(PATH)

        output = self.run_style_transfer(cnn, self.cnn_normalization_mean, self.cnn_normalization_std,
                                         content_img, style_img, input_img)

        self.output = output

        return output

    def save(self, path):
        # plt.figure()
        img = self.imshow(self.output, title='Output Image')
        img.save(path)
        # plt.ioff()
        # plt.show()


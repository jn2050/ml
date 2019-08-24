def mse_loss(input, target):
    return ((input - target) ** 2).sum() / input.data.nelement()


class BiasedLoss(torch.nn.Module):

    def __init__(self):
        super(BiasedLoss, self).__init__()

    def forward(self, y2, y):
        """
        :param y2: [batch_size, nclasses], log probabilities predictions, one per class (5 classes)
        :param y: [batch_size, 1], labels
        :return: loss (scalar)
        """

        loss = torch.autograd.Variable(torch.FloatTensor([0.0]))
        print(y)
        print(y.data.view(-1))
        exit(0)
        print(y2)
        msk = [0,1]
        msk = y.type(torch.LongTensor)
        print(msk)
        print(y2[:,msk])
        exit(0)
        z = torch.FloatTensor([1.0, 0.001])
        loss = -torch.log(z)
        print(z)
        print(loss)
        loss = torch.sum(loss)
        print(loss)

        return nn.NLLLoss.forward(y2, y)
        y_var = torch.autograd.Variable(y.data, requires_grad=True)
        _, preds = y2.max(dim=1)
        loss = torch.autograd.Variable(torch.FloatTensor([0.0]))
        for i in range(y_var.data.size()[0]):
            if preds.data[i] < y_var.data[i]:
                loss += 1.0 * (y_var.data[i] - preds.data[i])
            else:
                loss += 2.0 * (preds.data[i] - y_var.data[i])
        #loss = loss / 10.0
        return loss
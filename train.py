def train(training_data_loader, optimizer, model, criterion, epoch):

    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        input, label_x2, label_x4 = Variable(batch[0]), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)

        input = input.cuda()
        label_x2 = label_x2.cuda()
        label_x4 = label_x4.cuda()

        HR_2x, HR_4x = model(input)

        loss_x2 = criterion(HR_2x, label_x2)
        loss_x4 = criterion(HR_4x, label_x4)
        loss = loss_x2 + loss_x4

        optimizer.zero_grad()

        loss_x2.backward(retain_graph=True)

        loss_x4.backward()

        optimizer.step()

        if iteration%100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))

def main():
    pass

if __name__ == "__main__":
    main()
import torch
import models.SEMICON as SEMICON
from data.data_loader import load_data
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':

    query_dataloader, train_dataloader, _ = load_data(
            'food101',
            '../datasets/food-101',
            1000,
            2000,
            16,
            4,
        )


    num_classes, att_size, feat_size = 2000, 3, 2048
    model = SEMICON.semicon(code_length=48, num_classes=num_classes, att_size=att_size, feat_size=feat_size,
                                device='cuda', pretrained=True)

    model = model.to('cuda')
        
    model.load_state_dict(torch.load('./checkpoints/Food101-resume/48/model.pth'))

    model.eval()

    embs = []
    labels = []
    for batch, (data, targets, index) in enumerate(tqdm(train_dataloader)):
        data = torch.tensor(data, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        data, targets = data.to('cuda'), targets.to('cuda')

        hash_code, out = model(data)
        # out = out.view(1024, -1)
        B, C, H, W = out.shape
        out = out.view(B , -1)
        targets = targets.data.cpu().numpy()
        targets = np.argwhere(targets == 1)
        label = []
        for data in targets:
            label.append(data[1])
        # print(targets.shape)

        embs.append(out.data.cpu().numpy())
        labels.append(np.array(label))


    embs = np.concatenate(embs)
    labels = np.concatenate(labels)
    print(embs.shape)
    print(labels.shape)
    print(labels[0])



    import hypertools as hyp
    import matplotlib.pyplot as plt
    hyp.plot(embs,'.',reduce='TSNE',ndims=2, hue=labels)
    plt.title('TSNE')
    plt.savefig('images/food101_48_prev_2d.png', dpi=500)

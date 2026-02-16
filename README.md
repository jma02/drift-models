# My quick implementation of the popular drift model paper by Kaiming's group: [Generative Modeling via Drifting](https://arxiv.org/pdf/2602.04770v1).

- Download your own dataset and place it into `data/`. You can probably use an agent to shape the training code to target your dataset. 

- If you want to try CelebA, you might have to download the files one-by-one from Google Drive, if `gdown` can't automatically do it through Torchvision. Just keep running the script and downloading the files via the given links.

- Waiting for Mingyang to drop the code for their paper. 
    - How to handle gigantic `torch.cdist` computations?
    - How to handle huge softmax ops?

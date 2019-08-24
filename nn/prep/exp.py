

def image_samples(self, data, n=16):
    imgs = np.asarray(data.ds1.imgs)
    rand = np.random.choice(range(imgs.shape[0]), n, replace=False)
    fnames = imgs[rand, 0]
    titles = [data.classes[i] for i in imgs[rand, 1].astype(int)]
    df = pd.DataFrame(list(zip(fnames, titles)))
    df.to_csv(self.path + '/samples.csv', header=False, index=False)
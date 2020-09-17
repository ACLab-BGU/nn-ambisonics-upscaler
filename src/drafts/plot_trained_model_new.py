from src.models.cnn_model import CNN
import matplotlib.pyplot as plt
import src.utils.visualizations as vis
from src.utils.complex_tensors_old import complextorch2numpy


def plot_examples(chkpt_path):
    model = CNN.load_from_checkpoint(chkpt_path, num_workers=0)
    model.setup('fit')
    model.setup('test')
    model.eval()
    print(model.hparams)

    loader = model.test_dataloader()

    for x, R in loader:
        R_hat = model(x)

        x = complextorch2numpy(x[0],dim=0)
        R = complextorch2numpy(R[0],dim=0)
        R_hat = complextorch2numpy(R_hat[0],dim=0)

        R_small_x = (x @ x.T.conj())*(1/x.shape[1])

        vis.compare_covs(R_small_x, R_hat, R)
        vis.compare_power_maps(R_small_x, R_hat, R)

        plt.close('all')
        pass



if __name__ == '__main__':
    PATH = "/Users/ranweisman/PycharmProjects/nn-ambisonics-upscaler/experiments/cnn_10reflections_overfit/version_2/checkpoints/epoch=33.ckpt"
    plot_examples(PATH)


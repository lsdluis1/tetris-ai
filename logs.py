from torch.utils.tensorboard import SummaryWriter


class CustomTensorBoard():
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def set_model(self, model):
        pass

    def hparams(self, hparams: dict, metrics: dict):
        self.writer.add_hparams(hparams, metrics)

    def log(self, step, **stats):
        for key, val in stats.items():
            self.writer.add_scalar(key, val, global_step=step)

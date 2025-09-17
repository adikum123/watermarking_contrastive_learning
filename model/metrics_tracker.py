from model.utils import accuracy


class MetricsTracker:
    def __init__(self, name="train"):
        assert name in {"train", "val"}, "Name must be 'train' or 'val'"
        self.name = name
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_num = 0
        self.total_bits = 0
        self.total_acc_identity = 0.0
        self.total_acc_distorted = 0.0
        self.total_pesq = 0.0

    def update(
        self,
        loss=None,
        pesq=None,
        decoded=None,
        msg=None,
        batch_size=None,
    ):
        """
        Single update call to handle loss, PESQ, and accuracy tracking.

        Args:
            loss (float): batch loss
            pesq (float): PESQ score
            decoded (Tensor): decoded watermark
            msg (Tensor): original watermark
            batch_size (int): batch size
            identity (bool): if True, track identity accuracy
            distorted (bool): if True, track distorted accuracy
        """
        if batch_size is None and msg is not None:
            batch_size = msg.size(0)

        if batch_size is not None:
            self.total_num += batch_size

        if pesq is not None and batch_size is not None:
            self.total_pesq += pesq * batch_size

        if loss is not None and batch_size is not None:
            self.total_loss += loss * batch_size

        if decoded is not None and msg is not None:
            if isinstance(decoded, tuple):
                decoded_distorted, decoded_identity = decoded
                self.total_acc_distorted += accuracy(decoded=decoded_distorted, msg=msg)
                self.total_acc_identity += accuracy(decoded=decoded_identity, msg=msg)
            else:
                self.total_acc_identity += accuracy(decoded=decoded, msg=msg)
            self.total_bits += msg.numel()

    def average_loss(self):
        return self.total_loss / self.total_num

    def average_pesq(self):
        return self.total_pesq / self.total_num

    def avg_acc_identity(self):
        return self.total_acc_identity / self.total_bits

    def avg_acc_distorted(self):
        return self.total_acc_distorted / self.total_bits

    def summary(self):
        if self.name == "train":
            return {
                "loss": round(self.average_loss(), 7),
                "pesq": round(self.average_pesq(), 5),
                "avg_acc_identity": round(self.avg_acc_identity(), 4),
                "avg_acc_distorted": round(self.avg_acc_distorted(), 4),
            }
        return {
            "loss": round(self.average_loss(), 7),
            "pesq": round(self.average_pesq(), 4),
            "avg_acc": round(self.avg_acc_identity(), 4),
        }

    def __str__(self):
        if self.name == "train":
            return (
                f"Loss: {self.average_loss():.7f}, "
                f"PESQ: {self.average_pesq():.4f}, "
                f"Acc (id): {self.avg_acc_identity():.4f}, "
                f"Acc (dist): {self.avg_acc_distorted():.4f}"
            )
        return (
            f"Loss: {self.average_loss():.7f}, "
            f"PESQ: {self.average_pesq():.4f}, "
            f"Acc (id): {self.avg_acc_identity():.4f}"
        )

class MetricCal():
    def __init__(self) -> None:
        self.reset()
    def reset(self):
        self.total_loss = 0.0
        self.correct = 0
        self.total = 0
    def update(self, loss, outputs, targets, type="soft"):
        batch_size = targets.size(0)
        self.total_loss += loss.item() * batch_size
        if type == "soft":
            pred_class = outputs.argmax(dim=1)
            true_class = targets.argmax(dim=1)
        else:
            pred_class = outputs.max(1)
            true_class = targets
        
        self.correct += (pred_class == true_class).sum().item()
        self.total += batch_size

    @property
    def avg_loss(self):
        """Average loss over all accumulated batches."""
        return self.total_loss / self.total if self.total > 0 else 0.0

    @property
    def avg_accuracy(self):
        """Accuracy (%) over all accumulated batches."""
        return 100.0 * self.correct / self.total if self.total > 0 else 0.0

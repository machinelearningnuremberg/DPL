class WrappedDataLoader:
    def __init__(self, dl, dev):
        self.dl = dl
        self.device = dev

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield b[0].to(self.device), b[1].to(self.device), b[2].to(self.device), b[3].to(self.device)

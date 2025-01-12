import yaml
import argparse
import json

from SD_helper import S4T as S
from SD_helper.Diarization_dataset import AMI, DiLibriSM, DiarizationDataset
from SD_helper.losses import pit_loss, batch_pit_loss, eda_loss
from SD_helper.utils import report_diarization_error
from SD_helper.optim import NoamScheduler
from SA_EEND import *
from EEND_EDA import *
from EEND_VC import *

class DiLSSM(S.SDataModule):
    def __init__(self,
                 args):
        super().__init__()
        train_dataset = DiLibriSM(audio_directory = args.train_audio_dir,
                                  annotation_directory = args.train_annotation_dir)
        val_dataset = DiLibriSM(audio_directory = args.val_audio_dir,
                                annotation_directory = args.val_annotation_dir)
        test_dataset = DiLibriSM(audio_directory = args.test_audio_dir,
                                 annotation_directory = args.test_annotation_dir)
        train_speakers = val_speakers = test_speakers = None
        if args.train_speakers:
            train_speakers = json.load(open(args.train_speakers), 'rb')
        if args.val_speakers:
            val_speakers = json.load(open(args.val_speakers, 'rb'))
        if args.test_speakers:
            test_speakers = json.load(open(args.test_speakers, 'rb'))
        self.batch_size = args.batch_size
        self.train_dataset = DiarizationDataset(train_dataset,
                                                all_speakers = train_speakers, 
                                                input_transform = args.input_transform,
                                                n_speakers = args.n_speakers,
                                                chunk_size = args.chunk_size,
                                                context_size = args.context_size,
                                                subsampling = args.subsampling,
                                                win_length = args.win_length,
                                                hop_length = args.hop_length,
                                                use_last_samples = False)
        self.val_dataset = DiarizationDataset(val_dataset,
                                              all_speakers = val_speakers, 
                                              input_transform = args.input_transform,
                                              n_speakers = args.n_speakers,
                                              chunk_size = args.chunk_size,
                                              context_size = args.context_size,
                                              subsampling = args.subsampling,
                                              win_length = args.win_length,
                                              hop_length = args.hop_length,
                                              use_last_samples = False)
        self.test_dataset = DiarizationDataset(test_dataset,
                                               all_speakers = test_speakers, 
                                               input_transform = args.input_transform,
                                               n_speakers = args.n_speakers,
                                               chunk_size = args.chunk_size,
                                               context_size = args.context_size,
                                               subsampling = args.subsampling,
                                               win_length = args.win_length,
                                               hop_length = args.hop_length,
                                               use_last_samples = False)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size = self.batch_size,
                                           shuffle = True,
                                           num_workers = 4,
                                           prefetch_factor = 1,
                                           collate_fn = lambda x: self.collate_fn(x, True))

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size = self.batch_size,
                                           shuffle = False,
                                           num_workers = 4,
                                           prefetch_factor = 1,
                                           collate_fn = lambda x: self.collate_fn(x, False))

    def collate_fn(self, batch, train):
        x_batch, y_batch, lengths, spks, n_spks = [], [], [], [], []
        for x, y, spk in batch:
            x = torch.tensor(x)
            y = torch.tensor(y)
            x_batch.append(x)        # (T, D)
            y_batch.append(y)        # (T, n_speakers)
            lengths.append(len(x))
            if self.train_dataset.all_speakers:
                spks.append(spk)
            else:
                n_spks.append(len(spk))
        x_batch = nn.utils.rnn.pad_sequence(x_batch, batch_first = True, padding_value = -1)
        y_batch = nn.utils.rnn.pad_sequence(y_batch, batch_first = True, padding_value = 0).type(torch.float)
        if self.train_dataset.all_speakers:
            return x_batch, y_batch, torch.tensor(lengths), torch.tensor(spks)
        else:
            return x_batch, y_batch, torch.tensor(lengths), torch.tensor(n_spks)

class SA_EEND_training(SA_EEND, S.SModule):
    def __init__(self,
                 lr: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr
    
    def loss(self, y_hat, y, lengths):
        return batch_pit_loss(y_hat, y, lengths)

    def training_step(self, batch, batch_idx):
        x, y, lengths = batch
        padding_mask = None
        preds = self.forward(x, src_key_padding_mask = padding_mask)
        loss, labels = self.loss(preds, y, lengths)
        avg_der = report_diarization_error(preds, labels, lengths)
        self.log_dict({"train_loss": loss, "train_avg_der": avg_der}, 
                      pbar = True, train_logging = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, lengths = batch
        padding_mask = None
        preds = self.forward(x, src_key_padding_mask = padding_mask)
        loss, labels = self.loss(preds, y, lengths)
        avg_der = report_diarization_error(preds, labels, lengths)
        self.log_dict({"val_loss": loss, "val_avg_der": avg_der}, 
                      pbar = True, train_logging = False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 0, 
                                     betas = (0.9, 0.98),
                                     eps = 1e-9)
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure = optimizer_closure)

        try:
            lr = self.lr.calculate_lr(epoch, batch_idx)
        except:
            lr = self.lr

        for pg in optimizer.param_groups:
            pg['lr'] = lr
        self.log('lr', lr, pbar = True, train_logging = True)

class EEND_EDA_training(EEND_EDA, S.SModule):
    def __init__(self,
                 lr: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr
    
    def loss(self, 
             y_hat: torch.Tensor,
             attractor_logits: torch.Tensor,
             y: torch.Tensor, 
             lengths: Optional[torch.Tensor] = None, 
             n_spks: Optional[torch.Tensor] = None):
        return eda_loss(ys = y_hat, 
                        attractor_logits = attractor_logits, 
                        ts = y, 
                        n_spks = n_spks,
                        lengths = lengths)

    def training_step(self, batch, batch_idx):
        x, y, lengths, n_spks = batch
        padding_mask = None
        di_logits, attr_logits = self.forward(x, src_key_padding_mask = padding_mask)
        loss, labels = self.loss(di_logits, attr_logits, y, None, n_spks)
        avg_der = report_diarization_error(di_logits, labels, None, n_spks)
        self.log_dict({"train_loss": loss, "train_avg_der": avg_der}, 
                      pbar = True, train_logging = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, lengths, n_spks = batch
        padding_mask = None
        di_logits, attr_logits = self.forward(x, src_key_padding_mask = padding_mask)
        loss, labels = self.loss(di_logits, attr_logits, y, None, n_spks)
        avg_der = report_diarization_error(di_logits, labels, None, n_spks)
        self.log_dict({"val_loss": loss, "val_avg_der": avg_der}, 
                      pbar = True, train_logging = False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 0, 
                                     betas = (0.9, 0.98),
                                     eps = 1e-9)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure = optimizer_closure)

        try:
            lr = self.lr.calculate_lr(epoch, batch_idx)
        except:
            lr = self.lr

        for pg in optimizer.param_groups:
            pg['lr'] = lr
        self.log('lr', lr, pbar = True, train_logging = True)

class EEND_VC_training(EEND_VC, S.SModule):
    def __init__(self,
                 lr: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y, lengths, spk_indices = batch
        padding_mask = None
        di_logits, spk_vecs = self.forward(x, src_key_padding_mask = padding_mask)
        di_loss, labels, perms = batch_pit_loss(di_logits, y)
        spk_indices = torch.stack([spk_index[perm] for perm, spk_index in zip(perms, spk_indices)])
        spk_loss = self.spk_loss.calc_spk_loss(spk_vecs, spk_indices)
        loss = (1-0.01)*di_loss + 0.01*spk_loss
        avg_der = report_diarization_error(di_logits, labels)
        self.log_dict({"train_loss": loss, "train_avg_der": avg_der, 'train_di_loss': di_loss, 'train_spk_loss': spk_loss}, 
                      pbar = True, train_logging = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, lengths, spks_indices = batch
        padding_mask = None
        di_logits, spk_vecs = self.forward(x, src_key_padding_mask = padding_mask)
        loss, labels, perms = batch_pit_loss(di_logits, y)
        avg_der = report_diarization_error(di_logits, labels)
        self.log_dict({"val_loss": loss, "val_avg_der": avg_der}, 
                      pbar = True, train_logging = False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 0, 
                                     betas = (0.9, 0.98),
                                     eps = 1e-9)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure = optimizer_closure)

        try:
            lr = self.lr.calculate_lr(epoch, batch_idx)
        except:
            lr = self.lr

        for pg in optimizer.param_groups:
            pg['lr'] = lr
        self.log('lr', lr, pbar = True, train_logging = True)

def train(args):
    ### Setup data for training
    data = DiLSSM(args)

    ### Configure model
    lr = args.lr
    if lr == 'noam':
        lr = NoamScheduler(base_lr = args.base_lr,
                           d_model = args.d_model,
                           warmup_steps = args.warmup_steps,
                           step_per_epoch = len(data.train_dataloader())//args.batch_size + 1)
    if args.model == 'SA_EEND':
        model = SA_EEND_training(input_dim = args.input_dim,
                                 d_model = args.d_model,
                                 nhead = args.nhead,
                                 dim_ffn = args.dim_ffn,
                                 num_layers = args.num_layers,
                                 n_speakers = args.n_speakers,
                                 lr = lr,
                                 rel_attn = False)
    elif args.model == 'EEND_EDA':
        model= EEND_EDA_training(input_dim = args.input_dim,
                                 d_model = args.d_model,
                                 nhead = args.nhead,
                                 dim_ffn = args.dim_ffn,
                                 num_layers = args.num_layers,
                                 max_n_speakers = args.max_n_speakers,
                                 lr = lr,
                                 rel_attn = False)
    elif args.model == 'EEND_VC':
        model = EEND_VC_training(input_dim = args.input_dim,
                                 d_model = args.d_model,
                                 nhead = args.nhead,
                                 dim_ffn = args.dim_ffn,
                                 num_layers = args.num_layers,
                                 dropout = 0.1,
                                 n_speakers = args.n_speakers,
                                 d_spk = args.d_spk,
                                 n_all_speakers = len(data.train_dataset.all_speakers),
                                 lr = lr,
                                 rel_attn = False)
    else:
        raise "The model doesn't exist!!!"
    
    ### Train
    torch.manual_seed(args.seed)
    checkpoint_callback = S.ModelCheckpoint(dirpath = args.ckpt_dir,
                                            save_top_k = 10, monitor = 'val_loss',
                                            mode = 'min',
                                            filename = f'{args.model}-epoch:%02d-val_loss:%.4f')
    trainer = S.Trainer(accelerator = args.device,
                        callbacks = [checkpoint_callback],
                        enable_checkpointing = True,
                        max_epochs = args.max_epochs,
                        gradient_clip_val = args.gradient_clip_val)
    history = trainer.fit(model, data)
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_yaml", type = str)
    args = parser.parse_args()
    args = yaml.load(open(args.config_yaml, 'rb'), Loader = yaml.SafeLoader)
    args = argparse.Namespace(**args)
    train(args)    
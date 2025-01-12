from typing import Optional, Callable, List, Tuple, Any

import torch
from torch import nn
import numpy as np

import time
import sys
import os
import collections

class ModelCheckpoint:
    """
    filename format: "ckpt_name-epoch:%xxd-val_loss:%xxf..."
    monitor only working validation log
    """
    def __init__(self,
                 dirpath: str,
                 save_top_k: int,
                 monitor: str,
                 mode: str = 'min',
                 filename: str = 'model'):
        self.dirpath = dirpath
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.ckpt = {'name': [],
                     'monitor': []}
    
    def _save(self, 
              model,
              monitor,
              epoch,
              history):
        if 'epoch' in self.filename:
            filename = self.filename%(epoch, monitor) + '.ckpt'
        else:
            filename = self.filename%(monitor) + '.ckpt'
        ckpt = {'state_dict': model.state_dict(),
                'last_epoch': epoch + 1,
                'history': history}
        torch.save(ckpt, filename)
        return filename

    def update(self,
               model: Callable,
               dataloader: Callable,
               epoch: int,
               logging: dict,
               history: dict):
        len_dataloader = len(dataloader)
        monitor = sum(logging[self.monitor][-len_dataloader:])/len_dataloader
        if len(self.ckpt['name']) < self.save_top_k:
            self.ckpt['monitor'].append(monitor)
            filename = self._save(model,
                                  monitor,
                                  epoch,
                                  history)
            self.ckpt['name'].append(filename)
        else:
            monitor_log = np.array(self.ckpt['monitor'])
            if self.mode == 'min':
                idx = np.argmax(monitor_log)
                deq = True
            else:
                idx = np.argmin(monitor_log)
                deq = False
            monitor_base = self.ckpt['monitor'][idx]
            if (monitor - monitor_base < 0) == deq:
                os.remove(os.path.join(self.dirpath, self.ckpt['name'][idx]))
                self.ckpt['monitor'][idx] = monitor
                filename = self._save(model, 
                                      monitor,
                                      epoch,
                                      history)
                self.ckpt['name'][idx] = filename

class SDataModule:
    def __init__(self, root = None, num_workers = 4, **kwargs):
        self.root = root

    #def train_dataloader(self):
    #    raise NotImplementedError()

    #def val_dataloader(self):
    #    raise NotImplementedError()

    #def test_dataloader(self):
    #    raise NotImplementedError()

    def _prepare_dataloader(self, name):
        flag = getattr(self, name, None) is not None
        setattr(self, name+'_flag', flag)
        if flag:
            setattr(self, 'self_'+name, getattr(self, name)())

    def _prepare(self):
        self._prepare_dataloader('train_dataloader')
        self._prepare_dataloader('val_dataloader')
        self._prepare_dataloader('test_dataloader')

class SModule(nn.Module):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.logging = collections.defaultdict(list)
        self.pbar = {True: set(), False: set()}
        self.train_logging = {True: set(), False: set()}

    def log(self,
            name: str,
            value: object,
            pbar: bool = False,
            train_logging: bool = True):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.logging[name].append(value)
        self.pbar[pbar].add(name)
        self.train_logging[train_logging].add(name)

    def log_dict(self,
                 values: dict,
                 pbar: bool = False,
                 train_logging: bool = True):
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.logging[k].append(v)
            self.pbar[pbar].add(k)
            self.train_logging[train_logging].add(k)

    def loss(self, y_hat, y):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()

    #def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
    #    raise NotImplementedError()

    #def configure_gradient_clipping(self,):
    #    raise NotImplementedError()

class Trainer:
    def __init__(self,
                 accelerator: str = "gpu",
                 callbacks: Optional[List[Callable]] = None,
                 enable_checkpointing: bool = False,
                 default_root_dir: str = './',
                 max_epochs: int = 200,
                 gradient_clip_val: Optional[float] = None,
                 devices: Optional[List[int]] = None):
        self.accelerator = accelerator
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.enable_checkpointing = enable_checkpointing
        self.default_root_dir = default_root_dir
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, ModelCheckpoint):
                    self.model_ckpt = callback
                    self.model_ckpt_callback = True
        else:
             self.model_ckpt_callback = False
        if devices is None:
            self.device = 'cuda:0' if accelerator == 'gpu' else 'cpu'
            self.devices = [0]
        else:
            self.device = 'cuda:' + str(devices[0])
            self.devices = devices

    def _prepare(self, model, dataloader):
        dataloader._prepare()
        self.optimizer = model.configure_optimizers()
        self.dataloader = dataloader

    def fit(self,
            model: Callable,
            dataloader: Callable,
            ckpt_path: Optional[str] = None):
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, 
                              map_location = next(model.parameters()).device,
                              weights_only = False)
            model.load_state_dict(ckpt['state_dict'])
            start_epoch = ckpt['last_epoch']
            history = ckpt['history']
        else:
            start_epoch = 0
            history = collections.defaultdict(list)
        if len(self.devices) == 1:
            model = model.to(self.device)
        self._prepare(model, dataloader)
        self._loop(model, start_epoch, history)
        return self.history

    def evaluate(self,
                 model: Callable):
        #if self.dataloader.test_dataloader_flag:
        #    self.loop(model, self.dataloader.self_test_dataloader, False)
        #else:
            raise NotImplementedError()

    def _loop(self,
              model: Callable,
              start_epoch: int,
              history: dict):
        self.history = history
        for epoch in range(start_epoch, self.max_epochs):
            # Training
            model.train()
            for batch_idx, batch in enumerate(self.dataloader.self_train_dataloader):
                if len(self.devices) == 1:
                    batch = self._batch_cast(batch, self.device)
                time_start = time.time()
                def closure():
                    self.optimizer.zero_grad()
                    loss = model.training_step(batch, batch_idx)
                    loss.backward()
                    if self.gradient_clip_val:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)
                    return loss
                if getattr(model, "optimizer_step", None) is not None:
                    model.optimizer_step(epoch,
                                         batch_idx,
                                         self.optimizer,
                                         closure)
                else:
                    self.optimizer.step(closure)
                time_stop = time.time()
                ### Train logging
                logging = self._get_logging(model, 
                                            epoch,
                                            train = True)
                self._progress('training...', 
                               epoch, 
                               self.max_epochs, 
                               batch_idx, 
                               len(self.dataloader.self_train_dataloader), 
                               ' - %.2fs/step'% self._time(time_start, time_stop) + logging)

            # Validation
            if self.dataloader.val_dataloader_flag:
                model.eval()
                for batch_idx, batch in enumerate(self.dataloader.self_val_dataloader):
                    if len(self.devices) == 1:
                        batch = self._batch_cast(batch, self.device)
                    time_start = time.time()
                    with torch.no_grad():
                        model.validation_step(batch, batch_idx)
                    time_stop = time.time()
                    ### Val Logging
                    logging = self._get_logging(model, epoch,
                                           train = False)
                    self._progress('validating...', 
                                   epoch, 
                                   self.max_epochs, 
                                   batch_idx, 
                                   len(self.dataloader.self_val_dataloader), 
                                   ' - %.2fs/step'% self._time(time_start, time_stop) + logging)

            self._sumup(model)        

            if self.enable_checkpointing:
                if self.model_ckpt_callback:
                    self.model_ckpt.update(model,
                                           self.dataloader.self_val_dataloader,
                                           epoch,
                                           model.logging,
                                           self.history)
    
    def _batch_cast(self, 
                    batch: List[Any],
                    device: str):
        casted_batch = []
        for em in batch:
            if isinstance(em, torch.Tensor):
                em = em.to(device)
            if isinstance(em, list):
                casted_em = []
                for el in em:
                    if isinstance(el, torch.Tensor):
                        el = el.to(device)
                    if isinstance(el, dict):
                        for k, v in el.items():
                            if isinstance(v, torch.Tensor):
                                v = v.to(device)
                            el[k] = v
                    casted_em.append(el)
                em = casted_em
            casted_batch.append(em)
        return casted_batch
    
    def _progress(self, 
                  state: str, 
                  epoch: int, 
                  max_epochs: int, 
                  count: int, 
                  total_size: int, 
                  text: str):
        moving_arrow_pos = int(count/total_size*20)
        moving_arrow = moving_arrow_pos*['='] + ['>'] + (19-moving_arrow_pos)*['-'] 
        moving_arrow = ''.join(moving_arrow)
        sys.stdout.write(
            f'\r>> Epoch %d/%d: %s %d/%d [{moving_arrow}]%s' %
            (epoch, max_epochs, state, count, total_size, text))
        sys.stdout.flush()
    
    def _get_logging(self,
                     model,
                     epoch,
                     train: bool = True):
        text = ''
        for name in model.train_logging[train]:
            if name in model.pbar[True]:
                text += f' - {name}: %.4f'%(model.logging[name][-1])
        if train and epoch != 0 and self.dataloader.val_dataloader_flag:
            step_per_epoch = len(self.dataloader.self_val_dataloader)
            for name in model.train_logging[not train]:
                if name in model.pbar[True]:
                    text += f' - {name}: %.4f'%(sum(model.logging[name][-step_per_epoch:])/step_per_epoch)
        return text
    
    def _sumup(self,
               model):
        for name in model.train_logging[True]:
            step_per_epoch = len(self.dataloader.self_train_dataloader)
            self.history[name].append(sum(model.logging[name][-step_per_epoch:])/step_per_epoch)
        for name in model.train_logging[False]:
            step_per_epoch = len(self.dataloader.self_val_dataloader)
            self.history[name].append(sum(model.logging[name][-step_per_epoch:])/step_per_epoch)

    def _time(self,
              time_start: float,
              time_stop: float):
        return time_stop - time_start
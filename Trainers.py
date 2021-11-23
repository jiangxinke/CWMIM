import torch
import math
import os
import time
import copy
import numpy as np
import scipy.sparse as sp
from lib.utils import get_logger, evaluation, makedirs
from torch.utils.tensorboard import SummaryWriter
from lib.layers.GAT import GATNet


class StdTrainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader, scaler, args):
        super(StdTrainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        # 如果没有validation，将用test代替
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args

        self.scaler = scaler
        # self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.val_per_epoch = len(val_loader) if self.val_loader else len(test_loader)
        self.best_path = os.path.join(self.args.save_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.save_dir, 'loss.png')

        if self.args.writer and self.args.mode == "train":
            runs_dir = os.path.join(self.args.save_dir, 'runs', time.strftime("%m%d-%H-%M"))
            makedirs(runs_dir)
            self.writer = SummaryWriter(runs_dir)

        self.logger = get_logger(logpath=self.args.log_dir, filepath=os.path.abspath(__file__))
        self.logger.info('Experiment log path in: {}'.format(self.args.log_dir))
        if not args.debug:
            self.logger.info(args)
            parameters = self.return_parameters()
            log_message = "Total parameters is {}.".format(parameters)
            self.logger.info(log_message)
        else:
            self.args.epochs = 10
            self.args.log_freq = 1
            self.args.val_freq = 1
            self.args.early_stop_patience = 5

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch_idx, (x, label) in enumerate(self.train_loader):
                x = x.reshape([self.args.num_nodes, self.args.seq_len])
                label = label.reshape([self.args.num_nodes, self.args.pre_len])
                if self.args.real_value:
                    # 预测真实label
                    label = self.scaler.inverse_transform(label)
                Y_pre = self.model(x)
                eloss = self.loss(input=Y_pre, target=label)

                if not torch.isnan(eloss):
                    total_val_loss += eloss.item()

        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        for batch_idx, (x, label) in enumerate(self.train_loader):
            x = x.reshape([self.args.num_nodes, self.args.seq_len])
            label = label.reshape([self.args.num_nodes, self.args.pre_len])
            self.optimizer.zero_grad()

            Y_pre = self.model(x)
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)

            eloss = self.loss(input=Y_pre, target=label)
            eloss.backward()
            total_loss += eloss.item()

            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            if batch_idx % self.args.log_freq == 0:
                # 每 log_freq 个batch做一次log
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(epoch, batch_idx + 1, self.train_per_epoch, eloss.item()))
        train_epoch_loss = total_loss / self.train_per_epoch
        return train_epoch_loss

    def train(self):
        if self.args.conti:
            check_point = torch.load(self.best_path)
            self.model.load_state_dict(check_point['state_dict'])
            self.args = check_point['args']
            self.model.to(self.args.device)

        best_model = None
        best_loss = float('inf')
        early_stop = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):  # 总 epochs
            # 一个train 一个val交替进行
            torch.cuda.empty_cache()
            epoch_start = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            train_loss_list.append(train_epoch_loss)
            epoch_end = time.time()
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            self.logger.info('Train Epoch {}: averaged Loss: {:.6f} time: {:4}'.format(epoch, train_epoch_loss, epoch_end-epoch_start))

            if self.args.writer:
                self.writer.add_scalar('Train/loss', train_epoch_loss, epoch)

            if epoch % self.args.val_freq == 0:
                # 每 val_freq 个epoch做一次validation
                if not self.val_loader:
                    val_dataloader = self.test_loader
                else:
                    val_dataloader = self.val_loader
                val_epoch_loss = self.val_epoch(epoch, val_dataloader)
                val_loss_list.append(val_epoch_loss)
                if self.args.writer:
                    self.writer.add_scalar('Valid/loss', val_epoch_loss, epoch)

                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    early_stop = 0
                    best_state = True
                elif early_stop < self.args.early_stop_patience:
                    early_stop += 1
                    best_state = False
                else:
                    self.logger.info("Validation performance didn\'t improve for {} epochs.".format(early_stop))
                    break
                    # save the best state
                if best_state:
                    best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        # save the best model to file
        if not self.args.debug:
            self.model.load_state_dict(best_model)
            self.save_checkpoint()
            self.logger.info('Current best model saved!')
            self.test(self.model, self.args, self.test_loader, self.scaler)
            if self.args.writer:
                self.writer.close()

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    def return_parameters(self):
        # log net info
        size = 0
        for name, parameters in self.model.named_parameters():
            size += int(parameters.numel())
        return size

    @staticmethod
    def test(model, args, data_loader, scaler, path=None):
        if path:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['args']
            model.load_state_dict(state_dict)
            model.to(args.device)
            print("Load saved model")
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (x, label) in enumerate(data_loader):
                x = x.reshape([args.num_nodes, args.seq_len])
                label = label.reshape([args.num_nodes, args.pre_len])
                Y_pre = model(x)

                # Save normed values
                if args.real_value:
                    Y_pre = scaler.transform(Y_pre)
                y_pred.append(Y_pre.cpu().numpy())
                y_true.append(label.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        real_y_true = scaler.inverse_transform(y_true)
        real_y_pred = scaler.inverse_transform(y_pred)

        hyper_parameters = "seq{}_pre{}".format(args.seq_len, args.pre_len)
        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        rmse_ts, mae_ts, acc_ts, r2_ts, var_ts = evaluation(y_true, y_pred, args, args.acc_threshold)
        with open(args.save_file, mode='a') as fin:
            result = "\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{}".format(
                rmse_ts, mae_ts, acc_ts, r2_ts, var_ts, time_stamp, "normed_value", args.data, args.log_key)
            fin.write(result)

        rmse_ts, mae_ts, acc_ts, r2_ts, var_ts = evaluation(real_y_true, real_y_pred, args, args.acc_real_threshold)
        with open(args.save_file, mode='a') as fin:
            result = "\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{}".format(
                rmse_ts, mae_ts, acc_ts, r2_ts, var_ts, time_stamp, "real_value", args.data, args.log_key)
            fin.write(result)

        np.save(args.save_dir + r'\{}_true.npy'.format(args.data), y_true)
        np.save(args.save_dir + r'\{}_pred.npy'.format(args.data), y_pred)
        np.save(args.save_dir + r'\{}_real_true.npy'.format(args.data), real_y_true)
        np.save(args.save_dir + r'\{}_real_pred.npy'.format(args.data), real_y_pred)

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))


class GATTrainer(object):
    def __init__(self, model: GATNet, loss, optimizer, train_loader, val_loader, test_loader, scaler, args):
        super(GATTrainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        # 如果没有validation，将用test代替
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args

        self.scaler = scaler
        # self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.val_per_epoch = len(val_loader) if self.val_loader else len(test_loader)
        self.best_path = os.path.join(self.args.save_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.save_dir, 'loss.png')

        if self.args.writer and self.args.mode == "train":
            runs_dir = os.path.join(self.args.save_dir, 'runs', time.strftime("%m%d-%H-%M"))
            makedirs(runs_dir)
            self.writer = SummaryWriter(runs_dir)

        self.logger = get_logger(logpath=self.args.log_dir, filepath=os.path.abspath(__file__))
        self.logger.info('Experiment log path in: {}'.format(self.args.log_dir))
        if not args.debug:
            self.logger.info(args)
            parameters = self.return_parameters()
            log_message = "Total parameters is {}.".format(parameters)
            self.logger.info(log_message)
        else:
            self.args.epochs = 10
            self.args.log_freq = 1
            self.args.val_freq = 1
            self.args.early_stop_patience = 5

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch_idx, (x, label) in enumerate(self.train_loader):
                # x = x.reshape([self.args.num_nodes, self.args.seq_len])
                # label = label.reshape([self.args.num_nodes, self.args.pre_len])
                if self.args.real_value:
                    # 预测真实label
                    label = self.scaler.inverse_transform(label)
                Y_pre = self.model(x)
                eloss = self.loss(input=Y_pre, target=label)

                if not torch.isnan(eloss):
                    total_val_loss += eloss.item()

        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        for batch_idx, (x, label) in enumerate(self.train_loader):
            # x = x.reshape([self.args.num_nodes, self.args.seq_len])
            # label = label.reshape([self.args.num_nodes, self.args.pre_len])
            self.optimizer.zero_grad()

            Y_pre = self.model(x)
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)

            eloss = self.loss(input=Y_pre, target=label)
            eloss.backward()
            total_loss += eloss.item()

            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            if batch_idx % self.args.log_freq == 0:
                # 每 log_freq 个batch做一次log
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(epoch, batch_idx + 1, self.train_per_epoch, eloss.item()))
        train_epoch_loss = total_loss / self.train_per_epoch
        return train_epoch_loss

    def train(self):
        if self.args.conti:
            check_point = torch.load(self.best_path)
            self.model.load_state_dict(check_point['state_dict'])
            self.args = check_point['args']
            self.model.to(self.args.device)

        best_model = None
        best_loss = float('inf')
        early_stop = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):  # 总 epochs
            # 一个train 一个val交替进行
            torch.cuda.empty_cache()
            epoch_start = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            train_loss_list.append(train_epoch_loss)
            epoch_end = time.time()
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            self.logger.info('Train Epoch {}: averaged Loss: {:.6f} time: {:4}'.format(epoch, train_epoch_loss, epoch_end-epoch_start))

            if self.args.writer:
                self.writer.add_scalar('Train/loss', train_epoch_loss, epoch)

            if epoch % self.args.val_freq == 0:
                # 每 val_freq 个epoch做一次validation
                if not self.val_loader:
                    val_dataloader = self.test_loader
                else:
                    val_dataloader = self.val_loader
                val_epoch_loss = self.val_epoch(epoch, val_dataloader)
                val_loss_list.append(val_epoch_loss)
                if self.args.writer:
                    self.writer.add_scalar('Valid/loss', val_epoch_loss, epoch)

                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    early_stop = 0
                    best_state = True
                elif early_stop < self.args.early_stop_patience:
                    early_stop += 1
                    best_state = False
                else:
                    self.logger.info("Validation performance didn\'t improve for {} epochs.".format(early_stop))
                    break
                    # save the best state
                if best_state:
                    best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        # save the best model to file
        if not self.args.debug:
            self.model.load_state_dict(best_model)
            self.save_checkpoint()
            self.logger.info('Current best model saved!')
            self.test(self.model, self.args, self.test_loader, self.scaler)
            if self.args.writer:
                self.writer.close()

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    def return_parameters(self):
        # log net info
        size = 0
        for name, parameters in self.model.named_parameters():
            size += int(parameters.numel())
        return size

    @staticmethod
    def test(model, args, data_loader, scaler, path=None):
        if path:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['args']
            model.load_state_dict(state_dict)
            model.to(args.device)
            print("Load saved model")
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (x, label) in enumerate(data_loader):
                # x = x.reshape([args.num_nodes, args.seq_len])
                # label = label.reshape([args.num_nodes, args.pre_len])
                Y_pre = model(x)

                # Save normed values
                if args.real_value:
                    Y_pre = scaler.transform(Y_pre)
                y_pred.append(Y_pre.cpu().numpy())
                y_true.append(label.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        real_y_true = scaler.inverse_transform(y_true)
        real_y_pred = scaler.inverse_transform(y_pred)

        hyper_parameters = "seq{}_pre{}".format(args.seq_len, args.pre_len)
        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        rmse_ts, mae_ts, acc_ts, r2_ts, var_ts = evaluation(y_true, y_pred, args, args.acc_threshold)
        with open(args.save_file, mode='a') as fin:
            result = "\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{}".format(
                rmse_ts, mae_ts, acc_ts, r2_ts, var_ts, time_stamp, "normed_value", args.data, args.log_key)
            fin.write(result)

        rmse_ts, mae_ts, acc_ts, r2_ts, var_ts = evaluation(real_y_true, real_y_pred, args, args.acc_real_threshold)
        with open(args.save_file, mode='a') as fin:
            result = "\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{}".format(
                rmse_ts, mae_ts, acc_ts, r2_ts, var_ts, time_stamp, "real_value", args.data, args.log_key)
            fin.write(result)

        np.save(args.save_dir + r'\{}_true.npy'.format(args.data), y_true)
        np.save(args.save_dir + r'\{}_pred.npy'.format(args.data), y_pred)
        np.save(args.save_dir + r'\{}_real_true.npy'.format(args.data), real_y_true)
        np.save(args.save_dir + r'\{}_real_pred.npy'.format(args.data), real_y_pred)

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))

import pickle
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from collections import Counter, OrderedDict
import os
import contextlib
import math
from train_utils import AverageMeter
from models.flexmatch.flexmatch import FlexMatch
from models.flexmatch.flexmatch_utils import consistency_loss, Get_Scalar
from train_utils import ce_loss, wd_loss, EMA, Bn_Controller
from sklearn.metrics import *
from copy import deepcopy

class FlexRetrain(FlexMatch):
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, \
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
        super(FlexRetrain,self).__init__(net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, \
                 hard_label, t_fn, p_fn, it, num_eval_iter, tb_log, logger)
        self.teacher = net_builder(num_classes=num_classes)
        self.builder = net_builder
        self.num_classes = num_classes
        self.teacher_loss_ratio = 0.9 ##Hyperparameter

    def schedule(self,iterations,T):
        out = []
        for i in range(iterations):
            out.append(((math.cos(i*(1/iterations)*math.pi)+1)/2)*T)
        return out

    def load_teacher(self,load_path,save_dir,save_name):
        #Loads best model from previous iteration
        if os.path.exists(load_path):
          print(load_path)
          checkpoint = torch.load(load_path)
        else:
          checkpoint = torch.load("{}/{}/model_best.pth".format(save_dir,save_name))
        state_dict = checkpoint["model"]
        if "module" in list(state_dict.keys())[0]:
          new_state_dict = OrderedDict()
          for k,v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        else:
          new_state_dict = state_dict  
        self.teacher.load_state_dict(new_state_dict)
        self.teacher.cuda()
        self.print_fn('teacher loaded')

    def forward_pass(self, x_lb , x_ulb_w , x_ulb_s , x_ulb_idx , args ):
            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            x_ulb_idx = x_ulb_idx.cuda(args.gpu)
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
            # inference and calculate sup/unsup losses
            logits = self.model(inputs)
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
            return logits_x_lb, logits_x_ulb_w, logits_x_ulb_s

    def forward_pass_teacher(self, x_ulb_w , x_ulb_s , x_ulb_idx , args ):
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            x_ulb_w, x_ulb_s = x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            x_ulb_idx = x_ulb_idx.cuda(args.gpu)
            inputs = torch.cat((x_ulb_w, x_ulb_s))
            # inference and calculate sup/unsup losses
            logits = self.model(inputs)
            logits_x_ulb_w, logits_x_ulb_s = logits.chunk(2)
            return logits_x_ulb_w, logits_x_ulb_s

    def forward_pass_combined(self,x_lb, x_ulb_w , x_ulb_s , x_ulb_idx , args , idx ):
            num_ulb = x_ulb_w.shape[0]
            num_lb = x_lb.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            x_ulb_idx = x_ulb_idx.cuda(args.gpu)
            x_ulb_s_t , x_ulb_idx_t =  x_ulb_s[idx] , x_ulb_idx[idx]
            num_t = x_ulb_s_t.shape[0]
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s, x_ulb_s_t))
            # inference and calculate sup/unsup losses
            logits = self.model(inputs)
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:-num_t].chunk(2)
            logits_x_ulb_t = logits[-num_t:]
            return logits_x_lb, logits_x_ulb_w, logits_x_ulb_s, x_ulb_idx, logits_x_ulb_t, x_ulb_idx_t


    def retrain(self, args,teacher,logger=None,lb_eval = None ):
        self.it = 0
        self.num_eval_iter = args.num_eval_iter
        ngpus_per_node = torch.cuda.device_count()
        # EMA Init
        self.model = self.builder(num_classes=self.num_classes)
        self.model.cuda()
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)

        # p(y) based on the labeled examples seen during training
        dist_file_name = r"./data_statistics/" + args.dataset + '_' + str(args.num_labels) + '.json'
        if args.dataset.upper() == 'IMAGENET':
            p_target = None
        else:
            with open(dist_file_name, 'r') as f:
                p_target = json.loads(f.read())
                p_target = torch.tensor(p_target['distribution'])
                p_target = p_target.cuda(args.gpu)
            # print('p_target:', p_target)

        p_model = None

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1
        selected_label = selected_label.cuda(args.gpu)

        classwise_acc = torch.zeros((args.num_classes,)).cuda(args.gpu)
        print("Dataloaders size labeled size : {} , unlabelled size: {}".format(len(self.loader_dict["train_lb"]),len(self.loader_dict["train_ulb"])))

        for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s) in zip(self.loader_dict['train_teacher'],
                                                                  self.loader_dict['train_ulb']):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            #Update teacher loss weighting

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            pseudo_counter = Counter(selected_label.tolist())
            if max(pseudo_counter.values()) < len(self.ulb_dset):  # not all(5w) -1
                if args.thresh_warmup:
                    for i in range(args.num_classes):
                        classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
                else:
                    wo_negative_one = deepcopy(pseudo_counter)
                    if -1 in wo_negative_one.keys():
                        wo_negative_one.pop(-1)
                    for i in range(args.num_classes):
                        classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

            with amp_cm():
                logits_x_lb , logits_x_ulb_w ,logits_x_ulb_s = self.forward_pass( x_lb , x_ulb_w , x_ulb_s , x_ulb_idx , args )
                y_lb = y_lb.cuda(args.gpu)
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')


                # hyper-params for update
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)

                unsup_loss, mask, select, pseudo_lb, p_model = consistency_loss(logits_x_ulb_s,
                                                                                logits_x_ulb_w,
                                                                                classwise_acc,
                                                                                p_target,
                                                                                p_model,
                                                                                'ce', T, p_cutoff,
                                                                                use_hard_labels=args.hard_label,
                                                                                use_DA=args.use_DA)

                if teacher != None:
                    teacher.update(logits_x_ulb_w, x_ulb_idx)

                if lb_eval != None and self.it % 50 == 0 and self.it > self.num_eval_iter:
                    lb_eval.prob_eval(logits_x_ulb_w, x_ulb_idx, eval_dict['eval/top-1-acc'],self.it)

                if x_ulb_idx[select == 1].nelement() != 0:
                    selected_label[x_ulb_idx[select == 1]] = pseudo_lb[select == 1]

                #teacher_loss = teacher.loss(logits_x_ulb_s, x_ulb_idx)


                total_loss = sup_loss + ( self.lambda_u * unsup_loss )
                if self.it % 10 == 0:
                    print("[Losses] superv : {} -- unsup_loss: {}-- Total: {}".format(sup_loss,unsup_loss, total_loss))




            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/unsup_loss'] = unsup_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['train/mask_ratio'] = 1.0 - mask.detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.
            
            if self.it % 50 == 0:
              self.print_fn(pseudo_counter)

            # Save model for each 10K steps and best model for each 1K steps
            if self.it % 10000 == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('latest_model.pth', save_path)

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)
                save_path = os.path.join(args.save_dir, args.save_name)
                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it
                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")

                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    if self.it == best_it:
                        self.save_model('model_best.pth', save_path)
                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()
            #if self.it > 0.8 * args.num_train_iter:
            #    self.num_eval_iter = 1000

        self.save_teacher('model_teacher.pth', save_path)
        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict

    def save_teacher(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        self.ema.restore()
        self.teacher.train()

        torch.save({'model': self.teacher.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'ema_model': ema_model},
                   save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def retrain_teacher_only(self, args,teacher,logger=None,lb_eval = None ):
        self.teacher_loss_schedule = self.schedule(args.num_train_iter,self.teacher_loss_ratio)
        self.it = 0
        self.num_eval_iter = args.num_eval_iter
        ngpus_per_node = torch.cuda.device_count()
        # EMA Init
        self.model = self.builder(num_classes=self.num_classes)
        self.model.cuda()
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)

        # p(y) based on the labeled examples seen during training
        dist_file_name = r"./data_statistics/" + args.dataset + '_' + str(args.num_labels) + '.json'
        if args.dataset.upper() == 'IMAGENET':
            p_target = None
        else:
            with open(dist_file_name, 'r') as f:
                p_target = json.loads(f.read())
                p_target = torch.tensor(p_target['distribution'])
                p_target = p_target.cuda(args.gpu)
            # print('p_target:', p_target)

        p_model = None

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1
        selected_label = selected_label.cuda(args.gpu)

        classwise_acc = torch.zeros((args.num_classes,)).cuda(args.gpu)
        print("Dataloaders size labeled size : {} , unlabelled size: {}".format(len(self.loader_dict["train_lb"]),len(self.loader_dict["train_ulb"])))

        for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s) in zip(self.loader_dict['train_lb'],
                                                                  self.loader_dict['train_ulb']):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            #Update teacher loss weighting
            self.lambda_teacher = self.teacher_loss_schedule[self.it]

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            pseudo_counter = Counter(selected_label.tolist())
            if max(pseudo_counter.values()) < len(self.ulb_dset):  # not all(5w) -1
                if args.thresh_warmup:
                     for i in range(args.num_classes):
                        classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
                else:
                    wo_negative_one = deepcopy(pseudo_counter)
                    if -1 in wo_negative_one.keys():
                        wo_negative_one.pop(-1)
                    for i in range(args.num_classes):
                        classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

            with amp_cm():
                idx = teacher.valid_samples(x_ulb_idx)
                if torch.sum(idx) == 0:
                  print("no valid samples in batch: {}".format(idx.shape))
                  continue
                y_lb = y_lb.cuda(args.gpu)
                logits_x_lb, logits_x_ulb_w, logits_x_ulb_s, x_ulb_idx, logits_x_ulb_t, x_ulb_idx_t = self.forward_pass_combined(x_lb, x_ulb_w , x_ulb_s , x_ulb_idx , args ,idx)
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

                
                # hyper-params for update
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)

                unsup_loss, mask, select, pseudo_lb, p_model = consistency_loss(logits_x_ulb_s,
                                                                                logits_x_ulb_w,
                                                                                classwise_acc,
                                                                                p_target,
                                                                                p_model,
                                                                                'ce', T, p_cutoff,
                                                                                use_hard_labels=args.hard_label,
                                                                                use_DA=args.use_DA)
                

                if lb_eval != None and self.it % 50 == 0 and self.it > self.num_eval_iter:
                    lb_eval.prob_eval(logits_x_ulb_w, x_ulb_idx, eval_dict['eval/top-1-acc'],self.it)

                #if x_ulb_idx[select == 1].nelement() != 0:
                #    selected_label[x_ulb_idx[select == 1]] = pseudo_lb[select == 1]

                teacher_loss = teacher.hard_label_loss(logits_x_ulb_t, x_ulb_idx_t)


                total_loss = sup_loss + teacher_loss +  ( self.lambda_u * unsup_loss )
                if self.it % 10 == 0:
                    print("[Losses] teacher : {} -- Total: {}".format(teacher_loss, total_loss))



            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            #tb_dict['train/sup_loss'] = sup_loss.detach()
            #tb_dict['train/unsup_loss'] = unsup_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            #tb_dict['train/mask_ratio'] = 1.0 - mask.detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.
            
            if self.it % 50 == 0:
              self.print_fn(pseudo_counter)

            # Save model for each 10K steps and best model for each 1K steps
            if self.it % 10000 == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('latest_model.pth', save_path)

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)
                save_path = os.path.join(args.save_dir, args.save_name)
                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it
                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")

                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    if self.it == best_it:
                        self.save_model('model_best.pth', save_path)
                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()
            #if self.it > 0.8 * args.num_train_iter:
            #    self.num_eval_iter = 1000

        self.save_teacher('model_teacher.pth', save_path)
        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict

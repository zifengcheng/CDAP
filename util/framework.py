import os
import numpy as np
import sys
import time
import torch
from torch.nn import functional as F
# from pytorch_pretrained_bert import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import time
import re
from .utils import beam_search_soft_nms, now, get_p_r_f1, metrics_by_entity_tuples
import torch.nn as nn
O_CLASS = 0

class FewShotNERFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, opt):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.opt = opt

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    
    def get_entity_tuples(self, all_spans, span_lens, logits, pred, with_score=True, MAX_LEN=8,pred_token=None,label_token=None):
        """
        all_span: sent_num x [[l, r]]
        logits: all_span_num
        pred: all_span_num

        return tuple (span_l, span_r, span_score, pred)
        """
        assert len(pred) == len(span_lens) and len(logits) == len(span_lens)
        entity_tuples = []  # sentence_num x [tuple]
        idx = 0
        count = 0
        for i,all_span in  enumerate(all_spans):
            one_snt_tuples = []
            max_len = label_token[i].shape[0]
            for span in all_span:
                wrong = 0
                assert span_lens[idx] <= MAX_LEN
                [left,right] = span
                if pred[idx] != O_CLASS:
                    wrong = int((pred_token[count+left: count+right+1] != pred[idx]).long().sum())
                    logits[idx] = logits[idx] - wrong * self.opt.hyper
                    logits[idx] = 0 if logits[idx] < 0 else logits[idx]
                    if with_score:
                        one_snt_tuples.append((span[0], span[1], logits[idx], pred[idx]))
                    else:
                        one_snt_tuples.append((span[0], span[1], pred[idx]))
                idx += 1
            count += max_len
            entity_tuples.append(one_snt_tuples)
        assert len(span_lens) == idx
        return entity_tuples
    
    def get_token_prob(self, all_spans, span_lens, logits, pred, with_score=True, MAX_LEN=8,pred_token=None,label_token=None):
        """
        all_span: sent_num x [[l, r]]
        logits: all_span_num
        pred: all_span_num

        return tuple (span_l, span_r, span_score, pred)
        """
        length = 0
        for i in label_token:
            length += len(i)
        pred_token = torch.zeros(length,logits.shape[1]).cuda()

        assert len(pred) == len(span_lens) and len(logits) == len(span_lens)
        idx = 0
        count = 0

        for i,all_span in  enumerate(all_spans):
            one_snt_tuples = []
            max_len = label_token[i].shape[0]
            for span in all_span:
                [left,right] = span
                if logits[idx].softmax(0).max() > pred_token[count+left:count+right+1,:].softmax(1).max(1).values.min():
                    for i in range(count+left, count+right+1):
                        if logits[idx].softmax(0).max() >= pred_token[i].softmax(0).max():
                            pred_token[i] = logits[idx]
                    idx += 1
                else:
                    idx += 1
                    continue
            count += max_len
        assert len(span_lens) == idx
        #print(pred_token)
        return pred_token
    

    def get_token_prob_snips(self, all_spans, span_lens, logits, pred, with_score=True, MAX_LEN=8,pred_token=None,label_token=None):
        predict_token = []
        idx, count = 0, 0
        index = 0

        pred_temp = torch.zeros(pred_token[0].shape,requires_grad=True).cuda()
        for i,all_span in  enumerate(all_spans):
            temp_len = pred_token[index].shape[0]
            max_len = label_token[i].shape[0]

            for span in all_span:
                [left,right] = span
                if logits[index][idx].softmax(0).max() > pred_temp[count+left:count+right+1,:].softmax(1).max(1).values.min():
                    for i in range(count+left, count+right+1):
                        if logits[index][idx].softmax(0).max() >= pred_temp[i].softmax(0).max():
                            pred_temp[i] = logits[index][idx]
                    idx += 1
                else:
                    idx += 1
                    continue
            count += max_len
            if count >= temp_len:
                idx = 0
                count =0
                index += 1
                predict_token.append(pred_temp)
                if index < len(pred_token):
                    pred_temp = torch.zeros(pred_token[index].shape).cuda()
        return predict_token
    
    
    def train(self,
              model,
              model_name,
              opt,
              save_ckpt=None,
              warmup_step=300):
        
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_model = ['word_encoder']
        named_params = list(model.named_parameters())
    
        parameters_to_optimize = [
            {'params': [p for n, p in named_params if (not any(ll in n for ll in bert_model))], 'weight_decay': 0.0},
            {'params': [p for n, p in named_params if (not any(nd in n for nd in no_decay)) and any(ll in n for ll in bert_model)], 'weight_decay': 0.01, 'lr': opt.bert_lr}, # bert parameter
            {'params': [p for n, p in named_params if any(nd in n for nd in no_decay) and any(ll in n for ll in bert_model)], 'weight_decay': 0.0, 'lr': opt.bert_lr} # bert parameter

        ]
     
        optimizer = AdamW(parameters_to_optimize, lr=opt.lr, betas=(0.9, 0.999),  eps=1e-8, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=50000)
       

        # load model
        if opt.load_ckpt:
            state_dict = self.__load_model__(opt.load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, opt.load_ckpt))
                own_state[name].copy_(param)

        model.train()
        # Training
        best_f1 = 0
        iter_loss = 0.0
        patient = 0
        iter_sample = 0.0
        it = 0
        epoch = 1
        eval_per_epoch = int(len(self.train_data_loader) / 3)
        if self.opt.dataset == 'fewnerd':
            while True:
                for support, query in self.train_data_loader:
                    span_label = []
                    spans = []
                    span_lens = []
                    label = []
                    support['word'] = support['word'].cuda()
                    query['word'] = query['word'].cuda()
                    for span_tag in query['span_tags']: 
                        label.append(torch.tensor(span_tag).long().cuda())

                    res = model(support, query, query_label=label)
                    
                    for span_tag in query['span_tags']:
                        span_label.extend(span_tag)
                    for span in query['spans']:
                        spans.append(span)
                    for lens in query['span_lens']:
                        span_lens.extend(lens)

                    gold_entitys = query['gold_entitys']

                    logits = res['logits']
                    pred = res['pred']

                    assert len(pred.size()) == 1 and len(pred) == len(span_label)

                    pred = pred.cpu().tolist()
                    #logits = torch.max(logits.softmax(dim=-1), 1)[0].cpu().tolist()
                    
                    pred_token = self.get_token_prob(spans, span_lens, logits, pred, with_score=False, MAX_LEN=opt.L,label_token=query['seq_tag_expand'])

                    if opt.consistent_loss == 'KL':
                        kl_loss = nn.KLDivLoss()
                        loss = kl_loss(res['pred_logit'].log_softmax(dim=1),pred_token.softmax(dim=1)) + kl_loss(pred_token.log_softmax(dim=1), res['pred_logit'].softmax(dim=1))
                    elif opt.consistent_loss == 'MSE':
                        kl_loss = nn.MSELoss()
                        loss = kl_loss(res['pred_logit'].softmax(dim=1),pred_token.softmax(dim=1)) + kl_loss(pred_token.softmax(dim=1), res['pred_logit'].softmax(dim=1))
                    elif opt.consistent_loss == 'distill':
                        kl_loss = nn.KLDivLoss()
                        loss = kl_loss(res['pred_logit'].log_softmax(dim=1),(pred_token/opt.temperature).softmax(dim=1)) + kl_loss(pred_token.log_softmax(dim=1), (res['pred_logit']/opt.temperature).softmax(dim=1))
                    loss = loss * self.opt.beta
                    loss = (res['loss']+loss) / float(opt.grad_iter)
                    loss.backward()
                    if it % opt.grad_iter == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    iter_loss += self.item(loss.data)
                    iter_sample += 1
                    if (it + 1) % 50 == 0 or (it + 1) % opt.val_step == 0:
                        sys.stdout.write(
                            f"step: {it+1} | loss: {iter_loss/iter_sample:.6f} \r")

                    if (it + 1) % opt.val_step == 0:
                        _, _, f1 = self.eval(model, L=opt.L)
                        model.train()
                        patient += 1
                        if f1 > best_f1:
                            print('Best checkpoint')
                            torch.save({'state_dict': model.state_dict(), 'opt': opt, 'f1': f1, 'train_step': it}, save_ckpt)
                            best_f1 = f1
                            patient = 0
                        print('[Patient] {} / {}'.format(patient, opt.early_stop))
                        if patient >= opt.early_stop:
                            break
                        iter_loss = 0.
                        iter_sample = 0.

                    it += 1
                if patient >= opt.early_stop:
                    break
        
        else:
            while True:
                print(f'\nEpoch : {epoch}')
                for iteration, (support, query) in tqdm(enumerate(self.train_data_loader)):
                    span_label = []
                    spans = []
                    span_lens = []
                    label = []
                    support['word'] = support['word'].cuda()
                    query['word'] = query['word'].cuda()
                    for span_tag in query['span_tags']: 
                        label.append(torch.tensor(span_tag).long().cuda())

                    res = model(support, query, query_label=label, dataset='snips')
                    
                    for span_tag in query['span_tags']:
                        span_label.extend(span_tag)
                    for span in query['spans']:
                        spans.append(span)
                    for lens in query['span_lens']:
                        span_lens.extend(lens)

                    gold_entitys = query['gold_entitys']

                    logits = res['logits']
                    pred = res['pred']

                    pred_token = self.get_token_prob_snips(spans, span_lens, logits, pred, with_score=False, MAX_LEN=opt.L,pred_token=res['pred_logit'],label_token=query['seq_tag_expand'])
                    if opt.consistent_loss == 'KL':
                        kl_loss = nn.KLDivLoss()
                        loss = kl_loss(res['pred_logit'].log_softmax(dim=1),pred_token.softmax(dim=1)) + kl_loss(pred_token.log_softmax(dim=1), res['pred_logit'].softmax(dim=1))
                    elif opt.consistent_loss == 'MSE':
                        kl_loss = nn.MSELoss()
                        loss = kl_loss(res['pred_logit'].softmax(dim=1),pred_token.softmax(dim=1)) + kl_loss(pred_token.softmax(dim=1), res['pred_logit'].softmax(dim=1))
                    elif opt.consistent_loss == 'distill':
                        kl_loss = nn.KLDivLoss()
                        assert len(res['pred_logit']) == len(pred_token)
                        loss = 0
                        len_loss = 0
                        for i in range(len(pred_token)):
                            len_loss += pred_token[i].shape[0]
                            loss += (kl_loss(res['pred_logit'][i].log_softmax(dim=1),(pred_token[i]/opt.temperature).softmax(dim=1)) + kl_loss(pred_token[i].log_softmax(dim=1), (res['pred_logit'][i]/opt.temperature).softmax(dim=1))) * pred_token[i].shape[0]
                        loss /= len_loss
                    loss = loss * self.opt.beta
                    loss = (res['loss']+loss) / float(opt.grad_iter)
                    loss.backward()
                    if it % opt.grad_iter == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    iter_loss += self.item(loss.data)
                    iter_sample += 1
                    
                    if epoch >= 8 and iteration % eval_per_epoch == 0 and self.opt.dataset == 'snips':
                        _, _, f1 = self.eval(model, L=opt.L,dataset = 'snips')
                        #print('dev performance is',f1,'test performance is',f1_)
                        model.train()
                        patient += 1
                        if f1 > best_f1:
                            print('Best checkpoint')
                            torch.save({'state_dict': model.state_dict(), 'opt': opt, 'f1': f1, 'train_step': epoch}, save_ckpt)
                            best_f1 = f1
                            patient = 0
                        print('[Patient] {} / {}'.format(patient, opt.early_stop))
                        if patient >= opt.early_stop or epoch >= self.opt.max_epoch:
                            break
                if epoch >= 2 and self.opt.dataset == 'ner':
                    _, _, f1 = self.eval(model, L=opt.L,dataset = 'snips')
                    #print('dev performance is',f1,'test performance is',f1_)
                    model.train()
                    patient += 1
                    if f1 > best_f1:
                        print('Best checkpoint')
                        torch.save({'state_dict': model.state_dict(), 'opt': opt, 'f1': f1, 'train_step': epoch}, save_ckpt)
                        best_f1 = f1
                        patient = 0
                    print('[Patient] {} / {}'.format(patient, opt.early_stop))
                    if patient >= opt.early_stop or epoch >= self.opt.max_epoch:
                        break
                print(iteration)
                if patient >= opt.early_stop or epoch >= self.opt.max_epoch:
                    break
                epoch += 1

                sys.stdout.write(
                    f"epoch: {epoch} | loss: {iter_loss/iter_sample:.6f}")
                
                
                iter_loss = 0.
                iter_sample = 0.


        print("\n####################\n")
        print("Finish training " + model_name)


    def eval(self,
             model,
             ckpt=None,
             L=8,
             mode='val',dataset='fewnerd'):
        print("")
        model.eval()
        if ckpt is None:
            if mode == 'val':
                print("Use val dataset")
                eval_dataset = self.val_data_loader
            else:
                print("Use test dataset")
                eval_dataset = self.test_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                checkpoint = self.__load_model__(ckpt)
                state_dict = checkpoint['state_dict']
                dev_f1 = checkpoint['f1']
                train_step = checkpoint.get('train_step', 'xx')
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        iter_sample = 0.0

        iter_precision = 0.0
        iter_precision_beam_soft_nms = 0.0
        
        iter_recall = 0.0
        iter_recall_beam_soft_nms = 0.0

        iter_f1 = 0.0
        iter_f1_beam_soft_nms = 0.0

        all_correct_cnt = 0
        all_pred_cnt = 0
        all_gold_cnt = 0

        beam_soft_nms_all_correct = 0
        beam_soft_nms_all_pred = 0
        beam_soft_nms_all_gold = 0

        with torch.no_grad():
            start_time = time.time()
            for _, (support, query) in tqdm(enumerate(eval_dataset)):
                label = []
                spans = []
                span_lens = []
              
                support['word'] = support['word'].cuda()
                query['word'] = query['word'].cuda()
                res = model(support, query)
                #print(query['seq_tag_expand'])

                for span_tag in query['span_tags']:
                    label.extend(span_tag)
                for span in query['spans']:
                    spans.append(span)
                for lens in query['span_lens']:
                    span_lens.extend(lens)

                gold_entitys = query['gold_entitys']

                logits = res['logits']
                pred = res['pred']

                assert len(pred.size()) == 1 and len(pred) == len(label)

                pred = pred.cpu().tolist()
                logits = torch.max(logits.softmax(dim=-1), 1)[0].cpu().tolist()
                
                pred_entitys_origin = self.get_entity_tuples(spans, span_lens, logits, pred, with_score=False, MAX_LEN=L,pred_token=res['pred_token'],label_token=query['seq_tag_expand'])
                pred_entitys_origin_score = self.get_entity_tuples(spans, span_lens, logits, pred, with_score=True, MAX_LEN=L,pred_token=res['pred_token'],label_token=query['seq_tag_expand'])
        
                # -------------------- Span Refining Module (START) -------------------------
                pred_entitys_beam_soft_nms = beam_search_soft_nms(pred_entitys_origin_score[:], self.opt.beam_size, k=self.opt.soft_nms_k, u=self.opt.soft_nms_u, delta=self.opt.soft_nms_delta)
                # -------------------- Span Refining Module (END) ---------------------------

                correct_cnt, pred_cnt, label_cnt = metrics_by_entity_tuples(gold_entitys, pred_entitys_origin)
                assert label_cnt != 0
                precision, recall, f1 = get_p_r_f1(correct_cnt, pred_cnt, label_cnt)       

                correct_cnt_beam_soft_nms, pred_cnt_beam_soft_nms, label_cnt_beam_soft_nms = metrics_by_entity_tuples(gold_entitys, pred_entitys_beam_soft_nms)
                precision_beam_soft_nms, recall_beam_soft_nms, f1_beam_soft_nms = get_p_r_f1(correct_cnt_beam_soft_nms, pred_cnt_beam_soft_nms, label_cnt_beam_soft_nms)

            
                iter_precision += precision
                iter_recall += recall
                iter_f1 += f1

                iter_precision_beam_soft_nms += precision_beam_soft_nms
                iter_recall_beam_soft_nms += recall_beam_soft_nms
                iter_f1_beam_soft_nms += f1_beam_soft_nms

                all_correct_cnt += correct_cnt
                all_pred_cnt += pred_cnt
                all_gold_cnt += label_cnt

              
                beam_soft_nms_all_correct += correct_cnt_beam_soft_nms
                beam_soft_nms_all_pred += pred_cnt_beam_soft_nms
                beam_soft_nms_all_gold += label_cnt_beam_soft_nms

                iter_sample += 1


            all_p, all_r, all_f1 = get_p_r_f1(all_correct_cnt, all_pred_cnt, all_gold_cnt)
            all_p_beam_soft_nms, all_r_beam_soft_nms, all_f1_beam_soft_nms = get_p_r_f1(beam_soft_nms_all_correct, beam_soft_nms_all_pred, beam_soft_nms_all_gold)

            assert beam_soft_nms_all_gold == all_gold_cnt
            res_string = '''{} ||| [EVAL]
            Batch f1 [ SNIPS ]: [Beam Soft Nms]: ( p: {:.4f}; r: {:.4f}; f1: {:.4f} ) beam_size :{} k:{}, u:{}, delta:{}
            Batch f1 [ SNIPS ]: [Origin       ]: ( p: {:.4f}; r: {:.4f}; f1: {:.4f} )
            All   f1 [FewNERD]: [Beam Soft Nms]: ( p: {:.4f}; r: {:.4f}; f1: {:.4f} ) beam_size :{} k:{}, u:{}, delta:{}
            All   f1 [FewNERD]: [Origin       ]: ( p: {:.4f}; r: {:.4f}; f1: {:.4f} )
            '''.format(now(), 
                    iter_precision_beam_soft_nms / iter_sample, iter_recall_beam_soft_nms / iter_sample, iter_f1_beam_soft_nms / iter_sample, self.opt.beam_size, self.opt.soft_nms_k, self.opt.soft_nms_u, self.opt.soft_nms_delta,
                    iter_precision / iter_sample, iter_recall / iter_sample, iter_f1 / iter_sample,
                    all_p_beam_soft_nms, all_r_beam_soft_nms, all_f1_beam_soft_nms, self.opt.beam_size, self.opt.soft_nms_k, self.opt.soft_nms_u, self.opt.soft_nms_delta,
                    all_p, all_r, all_f1)
            sys.stdout.write(res_string + '\r')
            sys.stdout.flush()
            print("")
        if ckpt is None:
            if dataset == 'fewnerd':
                return all_p_beam_soft_nms, all_r_beam_soft_nms, all_f1_beam_soft_nms
            elif dataset == 'snips':
                return iter_precision_beam_soft_nms / iter_sample, iter_recall_beam_soft_nms / iter_sample, iter_f1_beam_soft_nms / iter_sample
    
        else:
            return "Dev f1: {}; train_step: {} ".format(dev_f1, train_step) + res_string
import time
from copy import deepcopy

import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F

from cope import CoPE
from model_utils import *
from eval_utils import *

from sklearn.metrics import *


def train_one_epoch(model, optimizer, train_dl, delta_coef=1e-5, tbptt_len=20,
                    valid_dl=None, test_dl=None, fast_eval=True, adaptation=False, adaptation_lr=1e-4):
    print("train one epoch parameters") 
    print("model", model)
    print("optimizer", optimizer)
    print("train_dl", train_dl)
    print("delta_coef", delta_coef)
    print("tbptt_len", tbptt_len)
    print("valid_dl", valid_dl)
    print("test_dl", test_dl)
    print("fast_eval", fast_eval)
    print("adaptation", adaptation)
    print("adaptation_lr", adaptation_lr)

    print("IN train_one_epoch Training...")
    last_xu, last_xi = model.get_init_states()
    #gets random tensor for the first time step
    print("last_xu, last_xi", last_xu.shape, last_xi.shape)
    print("last_xu, last_xi", last_xu, last_xi)

    loss_pp = 0.
    loss_norm = 0.
    # Sets gradients of all model parameters to zero.
    optimizer.zero_grad()

    # model.train() tells your model that you are training the model.
    model.train()


    counter = 0
    print("counter", counter)

    print(train_dl, "train_dl")
    

    pbar = tqdm.tqdm(train_dl)

    cum_loss = 0.


    for i, batch in enumerate(pbar):
        

        print("i", i)
        print("batch", batch)

        t, dt, adj, i2u_adj, u2i_adj, users, items = batch


        step_loss, delta_norm, last_xu, last_xi, *_ = model.propagate_update_loss(adj, dt, last_xu, last_xi, i2u_adj, u2i_adj, users, items)
        
        # print("step_loss", step_loss)
        # print("delta_norm", delta_norm)
        # print("last_xu, last_xi", last_xu.shape, last_xi.shape)
        # print("last_xu, last_xi", last_xu, last_xi)

        
        loss_pp += step_loss
        loss_norm += delta_norm
        counter += 1


        if (counter % tbptt_len) == 0 or i == (len(train_dl) - 1):
            total_loss = (loss_pp + loss_norm * delta_coef) / counter
            print("total_loss", total_loss)
            print("loss_pp", loss_pp)
            print("loss_norm", loss_norm)
            print("total_loss", type(total_loss))

            total_loss.backward()
            optimizer.step()
            
            cum_loss += total_loss.item()
            pbar.set_description(f"Loss={cum_loss/i:.4f}")

            last_xu = last_xu.detach()
            last_xi = last_xi.detach()

            optimizer.zero_grad()

            loss_pp = 0.
            loss_norm = 0.
            counter = 0

    pbar.close()
    
    # if fast_eval:
    #     rollout_evaluate_fast(model, valid_dl, test_dl, last_xu.detach(), last_xi.detach())
    # else:

    rollout_evaluate(model, train_dl, valid_dl, test_dl)


# def rollout_evaluate_fast(model, valid_dl, test_dl, train_xu, train_xi):
#     valid_xu, valid_xi, valid_ranks = rollout(valid_dl, model, train_xu, train_xi)
#     print(f"------- Valid MRR: {mrr(valid_ranks):.4f} Recall@10: {recall_at_k(valid_ranks, 10):.4f}")
#     _u, _i, test_ranks = rollout(test_dl, model, valid_xu, valid_xi)
#     print(f"=======  Test MRR: {mrr(test_ranks):.4f} Recall@10: {recall_at_k(test_ranks, 10):.4f}")


def rollout_evaluate(model, train_dl, valid_dl, test_dl):
    print("IN rollout_evaluate")
    train_xu, train_xi, train_ranks, metrics = rollout(train_dl, model, *model.get_init_states())
    print(f"Train MRR: {mrr(train_ranks):.4f} Recall@10: {recall_at_k(train_ranks, 10):.4f}")
    print(f"EXTRA TRAIN MEASURES: {metrics}")

    valid_xu, valid_xi, valid_ranks, metrics = rollout(valid_dl, model, train_xu, train_xi)
    print(f"Valid MRR: {mrr(valid_ranks):.4f} Recall@10: {recall_at_k(valid_ranks, 10):.4f}")
    print(f"EXTRA VALID MEASURES: {metrics}")


    _u, _i, test_ranks, metrics = rollout(test_dl, model, valid_xu, valid_xi)
    print(f"Test MRR: {mrr(test_ranks):.4f} Recall@10: {recall_at_k(test_ranks, 10):.4f}")
    print(f"EXTRA TEST MEASURES: {metrics}")



def rollout(dl, model, last_xu, last_xi):
    print("IN rollout_")
    model.eval()

    pos_scores_list = []
    neg_scores_list = []
    val_ap, val_auc = [], []
    measures_list = []
    ranks = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dl, position=0):
            t, dt, adj, i2u_adj, u2i_adj, users, items = batch
            prop_user, prop_item, last_xu, last_xi = model.propagate_update(adj, dt, last_xu, last_xi, i2u_adj, u2i_adj)



            # # compute performance measures
            mn = None

            yu = torch.cat([prop_user, model.user_states], 1)
            yi = torch.cat([prop_item, model.item_states], 1)

            # positive
            pos_u = F.embedding(users, yu, max_norm=mn)
            pos_i = F.embedding(items, yi, max_norm=mn)
            pos_scores = model.compute_matched_scores(pos_u, pos_i)

            pos_scores_list.append(torch.sigmoid(pos_scores).cpu().detach().numpy())

            # negative
            neg_u_ids = torch.randint(0, model.n_users, size=[model.n_neg_samples//2], device=users.device)
            neg_i_ids = torch.randint(0, model.n_items, size=[model.n_neg_samples//2], device=items.device)
            
            neg_u = F.embedding(neg_u_ids, yu, max_norm=mn)
            neg_i = F.embedding(neg_i_ids, yi, max_norm=mn)

            u_neg_scores = model.compute_pairwise_scores(pos_u, neg_i)
            i_neg_scores = model.compute_pairwise_scores(neg_u, pos_i)

            neg_scores = torch.cat([u_neg_scores, i_neg_scores.T], 1)

            neg_scores_list.append(torch.sigmoid(neg_scores).cpu().detach().numpy())

            print("pos_scores, neg_scores", pos_scores.shape, neg_scores.shape)
            print("pos_scores, neg_scores", pos_scores, neg_scores)

            
            rs = compute_rank(model, prop_user, prop_item, users, items)

            ranks.extend(rs)
            print("ranks_rollout", ranks)


        pos_scores_list = np.concatenate(pos_scores_list, axis=None).ravel()
        neg_scores_list = np.concatenate(neg_scores_list, axis=None).ravel()

        print("pos_scores_list", pos_scores_list)
        print("neg_scores_list", neg_scores_list)

        print("pos_scores_list", len(pos_scores_list))
        print("neg_scores_list", len(neg_scores_list))


        pred_score = np.concatenate([pos_scores_list, neg_scores_list])
        true_label = np.concatenate([np.ones(pos_scores_list.size), np.zeros(neg_scores_list.size)])

        val_ap.append(average_precision_score(true_label, pred_score))
        val_auc.append(roc_auc_score(true_label, pred_score))

        # extra performance measures
        measures_dict = extra_measures(true_label, pred_score)
        measures_list.append(measures_dict)

    metrics = {"P": np.mean(val_ap), 
    "AUC_ALL": np.mean(val_auc), 
    "AP": np.mean(val_ap), 
    "AUC": np.mean(val_auc),
    "measures": measures_list}

    return last_xu, last_xi, ranks, metrics


def compute_rank(model: CoPE, xu, xi, users, items):
    xu = torch.cat([xu, model.user_states], 1)
    xi = torch.cat([xi, model.item_states], 1)
    xu = F.embedding(users, xu)
    scores = model.compute_pairwise_scores(xu, xi)
    ranks = []
    for line, i in zip(scores, items):
        r = (line >= line[i]).sum().item()
        ranks.append(r)
    print("ranks_rollout", ranks)
    return ranks

def extra_measures(y_true, y_pred_score):
    """
    compute extra performance measures
    """
    perf_dict = {}
    # find optimal threshold of au-roc
    perf_dict['ap'] = average_precision_score(y_true, y_pred_score)

    perf_dict['au_roc_score'] = roc_auc_score(y_true, y_pred_score)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_score)
    opt_idx = np.argmax(tpr - fpr)
    opt_thr_auroc = roc_thresholds[opt_idx]
    perf_dict['opt_thr_au_roc'] = opt_thr_auroc
    auroc_perf_dict = get_measures_for_threshold(y_true, y_pred_score, opt_thr_auroc)
    perf_dict['acc_auroc_opt_thr'] = auroc_perf_dict['acc']
    perf_dict['prec_auroc_opt_thr'] = auroc_perf_dict['prec']
    perf_dict['rec_auroc_opt_thr'] = auroc_perf_dict['rec']
    perf_dict['f1_auroc_opt_thr'] = auroc_perf_dict['f1']

    prec_pr_curve, rec_pr_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_score)
    perf_dict['au_pr_score'] = auc(rec_pr_curve, prec_pr_curve)
    # convert to f score
    fscore = (2 * prec_pr_curve * rec_pr_curve) / (prec_pr_curve + rec_pr_curve)
    opt_idx = np.argmax(fscore)
    opt_thr_aupr = pr_thresholds[opt_idx]
    perf_dict['opt_thr_au_pr'] = opt_thr_aupr
    aupr_perf_dict = get_measures_for_threshold(y_true, y_pred_score, opt_thr_aupr)
    perf_dict['acc_aupr_opt_thr'] = aupr_perf_dict['acc']
    perf_dict['prec_aupr_opt_thr'] = aupr_perf_dict['prec']
    perf_dict['rec_aupr_opt_thr'] = aupr_perf_dict['rec']
    perf_dict['f1_aupr_opt_thr'] = aupr_perf_dict['f1']

    # threshold = 0.5
    perf_half_dict = get_measures_for_threshold(y_true, y_pred_score, 0.5)
    perf_dict['acc'] = perf_half_dict['acc']
    perf_dict['prec'] = perf_half_dict['prec']
    perf_dict['rec'] = perf_half_dict['rec']
    perf_dict['f1'] = perf_half_dict['f1']

    return perf_dict

def get_measures_for_threshold(y_true, y_pred_score, threshold):
    """
    compute measures for a specific threshold
    """
    perf_measures = {}
    y_pred_label = y_pred_score > threshold
    perf_measures['acc'] = accuracy_score(y_true, y_pred_label)
    prec, rec, f1, num = precision_recall_fscore_support(y_true, y_pred_label, average='binary', zero_division=1)
    perf_measures['prec'] = prec
    perf_measures['rec'] = rec
    perf_measures['f1'] = f1
    return perf_measures
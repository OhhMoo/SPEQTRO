"""
SSIN model for IR functional group detection.

Vendored from SSIN/method/model.py. The training method (fit) is retained
for completeness but inference entry points are predict_from_jdx and
identify_impt_peaks.
"""

import numpy
import torch.nn
from torch.nn.functional import normalize, sigmoid
from copy import deepcopy
from speqtro.vendors.ssin.chem import topk_points


class SSIN(torch.nn.Module):
    def __init__(self, dim_emb, len_spect, ref_db):
        super(SSIN, self).__init__()
        self.ref_db = ref_db
        self.pos_refs = torch.cat([d.absorbance_savgol.view(1, -1) for d in self.ref_db.data if d.label == 1], dim=0)
        self.dropout_p = 0.5

        self.fc_feat = torch.nn.Sequential(
            torch.nn.Linear(self.pos_refs.shape[0], 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU()
        )
        self.fc_attn = torch.nn.Sequential(
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
        self.fc_seq = torch.nn.Sequential(
            torch.nn.Linear(len_spect, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(1024, dim_emb),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=self.dropout_p)
        )
        self.fc_out = torch.nn.Linear(dim_emb, 1)

    def forward(self, spect, refs):
        _spect = (spect + 1).unsqueeze(2).repeat(1, 1, refs.shape[0])
        _refs = (refs + 1).swapaxes(0, 1).unsqueeze(0).repeat(spect.shape[0], 1, 1)
        x = 1 / (1 + torch.exp((_spect - _refs)**2))
        z = self.fc_feat(x)

        weight = deepcopy(spect)
        attns = weight[:, :z.shape[1]] * torch.exp(self.fc_attn(spect.unsqueeze(2) + z).squeeze(2))
        attns = attns / (torch.sum(attns, dim=1, keepdim=True) + 1e-10)

        z = torch.sum(z, dim=2)
        z = self.fc_seq(attns * (spect + z))
        z = normalize(z, p=2, dim=1)
        out = self.fc_out(z)

        return out, attns, z

    def fit(self, data_loader, optimizer, loss_func, alpha=0.4):
        _alpha = torch.tensor(alpha, dtype=torch.float, requires_grad=False)
        sum_losses = 0

        self.train()
        for spect, y in data_loader:
            spect = spect.cuda()
            refs = self.pos_refs.cuda()
            y = y.view(-1, 1).cuda()

            preds, attns, z = self(spect, refs)
            _, _, z_ref = self(refs, refs)
            loss = loss_func(preds, y)

            _z = z.unsqueeze(1).repeat(1, z_ref.shape[0], 1)
            _z_ref = z_ref.unsqueeze(0).repeat(z.shape[0], 1, 1)
            ref_dists = torch.norm(_z - _z_ref, dim=2)
            loss += torch.mean(y * ref_dists)
            loss += torch.mean((1 - y) * torch.maximum(_alpha, 2 - ref_dists))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_losses += loss.item()

        return sum_losses / len(data_loader)

    def predict(self, data_loader):
        list_preds = list()
        list_attns = list()

        self.eval()
        with torch.no_grad():
            for spect, _ in data_loader:
                preds, attns, _ = self(spect.cuda(), self.pos_refs.cuda())
                list_preds.append(preds)
                list_attns.append(attns)

        return torch.cat(list_preds, dim=0).cpu(), torch.cat(list_attns, dim=0).cpu()


def predict_from_jdx(model, irs):
    model.eval()
    with torch.no_grad():
        absorbance = torch.tensor(irs.absorbance_savgol, dtype=torch.float).unsqueeze(0)
        pred, attns, _ = model(absorbance.cuda(), model.pos_refs.cuda())

    label = 'Not detected' if sigmoid(pred) <= 0.5 else 'Detected'

    return label, attns[0].cpu().numpy()


def identify_impt_peaks(irs, attns, target_fg):
    topk_ind = sorted(attns.argsort()[-topk_points[target_fg]:][::-1], reverse=False)
    topk_wn = irs.wavenumber[topk_ind]
    impt_peaks = cluster_wn(topk_wn)

    return impt_peaks


def cluster_wn(wn):
    peaks = list()
    peak = [wn[0]]

    for i in range(1, wn.shape[0]):
        if wn[i] - wn[i - 1] < 100:
            peak.append(wn[i])
        else:
            if len(peak) == 0:
                peak = [wn[i - 1]]
            peaks.append(peak)
            peak = [wn[i]]
    peaks.append(peak + [wn[-1]])

    for i in range(0, len(peaks)):
        peaks[i] = [numpy.min(peaks[i]), numpy.max(peaks[i])]

    return peaks

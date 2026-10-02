"""PSDINet: patient similarity and drug interactions (PyTorch).

接口沿用参考 models.py: input 是一个患者截至当前就诊的列表，
每次就诊为 [diagnosis_ids, procedure_ids, medication_ids, ...]。
训练返回 (logits, ddi_penalty)，评估返回 logits；概率用 sigmoid(logits)。
论文训练目标仅为 BCE + margin，ddi_penalty 仅兼容原训练接口，默认不加入损失。

实现选择（论文未给出完整源码，不能保证与作者实现完全一致）：
1. 按 Eq.15 使用 Top-1，而非 Fig.2 的 Top-k 加权。
2. 用单层 GRU 实现 RNN；医疗码集合取均值；空集合为零向量。
3. Eq.25 各表示形状未定义，先投影为 d 维再相加。
4. Eq.24 缺少 value 项，补全标准 scaled dot-product attention。
5. 分子图使用五层 GCN + 均值池化；每种药可映射多个分子，取均值。
6. memory 使用训练患者的指定就诊前缀（默认末次就诊）；只能传训练集。
   memory 是显式刷新的缓存，不反传梯度；每个 epoch 开始 refresh_memory()。
7. dropout 默认沿用参考代码 0.4 / GCN 0.3；并非论文报告的超参数。

必需外部数据：interaction_adj（0=无边,1=拮抗,2=协同）、ddi_adj、
已预处理分子图、药物索引到分子索引的映射；本文件不从 SMILES 推测这些数据。

示例：
    model = PSDINet(vocab_size, interaction_adj, ddi_adj, molecular_graphs,
                   drug_to_molecules, atom_dim=atom_feature_dim, emb_dim=64)
    model.set_memory(train_patients, anchor_size=300, seed=0)
    model.train()
    logits, ddi_penalty = model(patient_prefix, patient_index=training_patient_index)
    loss = model.compute_loss(logits, patient_prefix[-1][2], alpha=0.9)
    loss.backward()
    # 每个 epoch 开始、验证/测试之前：model.refresh_memory()

分子图格式：{'node_features': float tensor [atoms, atom_dim],
             'adj': float tensor [atoms, atoms]}。
drug_to_molecules 长度等于药物词表，每项为分子索引列表，必须非空。
原参考 MELORA 的构造参数无法表达新增知识；主类名改为 PSDINet。
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_adj(adj):
    """Eq.16: D^-1/2 (A + I) D^-1/2，支持加权图。"""
    adj = torch.as_tensor(adj, dtype=torch.float32).detach().clone()
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError('adj must be square')
    if not torch.isfinite(adj).all() or (adj < 0).any():
        raise ValueError('adj must be finite and nonnegative')
    adj = adj + torch.eye(adj.shape[0], device=adj.device)
    inv = adj.sum(-1).clamp_min(1e-12).rsqrt()
    return inv[:, None] * adj * inv[None, :]


class GCN(nn.Module):
    """两层药物 GCN；identity 为初始节点特征，与参考文件一致。"""
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu'), dropout=0.3):
        super().__init__()
        adjacency = normalized_adj(adj)
        if adjacency.shape != (voc_size, voc_size):
            raise ValueError('drug adjacency shape does not match medication vocabulary')
        self.register_buffer('adj', adjacency)
        self.register_buffer('x', torch.eye(voc_size))
        self.gcn1 = nn.Linear(voc_size, emb_dim, bias=False)
        self.gcn2 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to(device)

    def forward(self):
        h = self.dropout(F.relu(self.adj @ self.gcn1(self.x)))
        return self.adj @ self.gcn2(h)


class PatientEncoder(nn.Module):
    """Eq.1-8: 诊断、手术、上次药物分别送入 GRU。"""
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu'), dropout=0.4):
        super().__init__()
        self.emb_dim = emb_dim
        self.embeddings = nn.ModuleList([nn.Embedding(v, emb_dim) for v in vocab_size])
        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True)
                                      for _ in range(3)])
        self.dropout = nn.Dropout(dropout)
        self.to(device)

    def mean_emb(self, codes, modality):
        embedding = self.embeddings[modality]
        codes = torch.as_tensor(codes, dtype=torch.long, device=embedding.weight.device)
        if codes.numel() == 0:
            return embedding.weight.new_zeros(1, 1, self.emb_dim)
        return self.dropout(embedding(codes)).mean(0).reshape(1, 1, -1)

    def forward(self, input, current_medications=False):
        if not input:
            raise ValueError('patient visit prefix cannot be empty')
        sequences = [[], [], []]
        for i, visit in enumerate(input):
            if len(visit) < 2:
                raise ValueError('visit requires diagnosis and procedure code lists')
            sequences[0].append(self.mean_emb(visit[0], 0))
            sequences[1].append(self.mean_emb(visit[1], 1))
            # 查询端只读取 i-1 的药；绝不读取待预测就诊的药物标签。
            meds = visit[2] if current_medications else (input[i-1][2] if i else [])
            sequences[2].append(self.mean_emb(meds, 2))
        outputs = [gru(torch.cat(seq, dim=1))
                   for gru, seq in zip(self.encoders, sequences)]
        states = [torch.cat([h[-1], o[:, -1]], dim=-1) for o, h in outputs]
        # r=[hd,hp,od,op]，按 Eq.7 顺序。
        r = torch.cat([outputs[0][1][-1], outputs[1][1][-1],
                       outputs[0][0][:, -1], outputs[1][0][:, -1]], dim=-1)
        return r, states[2]


class MolecularGraph(nn.Module):
    def __init__(self, graph, atom_dim):
        super().__init__()
        x = torch.as_tensor(graph['node_features'], dtype=torch.float32).detach().clone()
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != atom_dim:
            raise ValueError('invalid molecular node features')
        if not torch.isfinite(x).all():
            raise ValueError('molecular features must be finite')
        adj = normalized_adj(graph['adj'])
        if adj.shape[0] != x.shape[0]:
            raise ValueError('molecular adjacency and node count differ')
        self.register_buffer('x', x)
        self.register_buffer('adj', adj)


class MolecularEncoder(nn.Module):
    """Eq.20-22: 五层图聚合，池化得到每种药物的分子表示。"""
    def __init__(self, graphs, drug_to_molecules, atom_dim, emb_dim, dropout=0.3):
        super().__init__()
        if not graphs:
            raise ValueError('molecular_graphs cannot be empty')
        self.graphs = nn.ModuleList([MolecularGraph(g, atom_dim) for g in graphs])
        self.layers = nn.ModuleList([nn.Linear(atom_dim if i == 0 else emb_dim,
                                              emb_dim) for i in range(5)])
        self.dropout = nn.Dropout(dropout)
        mapping = torch.zeros(len(drug_to_molecules), len(graphs))
        for drug, indices in enumerate(drug_to_molecules):
            if not indices or any(i < 0 or i >= len(graphs) for i in indices):
                raise ValueError('each drug needs valid molecular indices')
            indices = sorted(set(indices))
            mapping[drug, indices] = 1.0 / len(indices)
        self.register_buffer('mapping', mapping)

    def forward(self):
        representations = []
        for graph in self.graphs:
            h = graph.x
            for layer in self.layers:
                h = self.dropout(F.relu(graph.adj @ layer(h)))
            representations.append(h.mean(0))
        return self.mapping @ torch.stack(representations)


class PSDINet(nn.Module):
    def __init__(self, vocab_size, interaction_adj, ddi_adj, molecular_graphs,
                 drug_to_molecules, atom_dim, emb_dim=64, device=torch.device('cpu'),
                 dropout=0.4, gcn_dropout=0.3, use_similar=True, use_interaction=True,
                 use_history=True, use_molecular=True):
        super().__init__()
        if len(vocab_size) != 3 or any(v <= 0 for v in vocab_size):
            raise ValueError('vocab_size must be [diagnoses, procedures, medications]')
        self.vocab_size = tuple(vocab_size)
        self.emb_dim = emb_dim
        self.use_similar, self.use_interaction = use_similar, use_interaction
        self.use_history, self.use_molecular = use_history, use_molecular
        interaction_adj = torch.as_tensor(interaction_adj, dtype=torch.float32)
        if not ((interaction_adj == 0) | (interaction_adj == 1) | (interaction_adj == 2)).all():
            raise ValueError('interaction_adj values must be 0, 1, or 2 (Eq.16 input)')
        self.patient_encoder = PatientEncoder(vocab_size, emb_dim, dropout=dropout)
        self.di_gcn = GCN(vocab_size[2], emb_dim, interaction_adj, dropout=gcn_dropout)
        self.ddi_gcn = GCN(vocab_size[2], emb_dim, ddi_adj, dropout=gcn_dropout)
        self.register_buffer('tensor_ddi_adj', torch.as_tensor(ddi_adj, dtype=torch.float32).clone())
        if len(drug_to_molecules) != vocab_size[2]:
            raise ValueError('drug_to_molecules must follow medication vocabulary order')
        self.molecular_encoder = MolecularEncoder(molecular_graphs, drug_to_molecules,
                                                  atom_dim, emb_dim, gcn_dropout)
        self.history_projection = nn.Linear(2 * emb_dim, emb_dim)
        self.similar_projection = nn.Linear(2 * emb_dim, emb_dim)
        self.query = nn.Sequential(nn.Linear(4 * emb_dim, emb_dim), nn.ReLU())
        self.di_query = nn.Linear(4 * emb_dim, emb_dim)
        self.query_projection = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key_projection = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value_projection = nn.Linear(emb_dim, emb_dim, bias=False)
        self.output = nn.Linear(emb_dim, vocab_size[2])
        # 缓存随 state_dict 保存，可变长度通过 load_state_dict hook 处理。
        self.register_buffer('memory_keys', torch.empty(0, 4 * emb_dim))
        self.register_buffer('memory_values', torch.empty(0, 2 * emb_dim))
        self._anchor_records = None
        self._training_patient_count = None
        self.anchor_indices = []
        self.init_weights()
        self.to(device)

    def init_weights(self):
        for embedding in self.patient_encoder.embeddings:
            nn.init.uniform_(embedding.weight, -0.1, 0.1)

    @torch.no_grad()
    def set_memory(self, train_patients, anchor_size=300, seed=0):
        """Eq.9-13: farthest-point sampling。参数必须是训练集患者前缀。

        默认传完整训练患者，则使用每名患者最后一次就诊；也可传指定前缀。
        不自动展开所有前缀，防止把同一患者多次计为多个 anchor。
        train_patients 不保存在 checkpoint；恢复训练后重新调用此方法。
        """
        if not train_patients or anchor_size <= 0:
            raise ValueError('nonempty training patients and positive anchor_size required')
        was_training = self.training
        self.eval()
        try:
            keys = torch.cat([self.patient_encoder(p)[0] for p in train_patients])
            keys = F.normalize(keys, dim=-1)
            count = min(anchor_size, len(train_patients))
            generator = torch.Generator().manual_seed(seed)
            first = int(torch.randint(len(train_patients), (1,), generator=generator))
            selected, used = [first], torch.zeros(len(train_patients), dtype=torch.bool, device=keys.device)
            used[first] = True
            minimum = 1 - keys @ keys[first]
            for _ in range(count - 1):
                index = int(minimum.masked_fill(used, -torch.inf).argmax())
                selected.append(index)
                used[index] = True
                minimum = torch.minimum(minimum, 1 - keys @ keys[index])
            self.anchor_indices = selected
            self._training_patient_count = len(train_patients)
            self._anchor_records = copy.deepcopy([train_patients[i] for i in selected])
            self.refresh_memory()
        finally:
            self.train(was_training)

    @torch.no_grad()
    def refresh_memory(self):
        """模型参数更新后重算 anchor 表示；不重新选择 anchor。"""
        if self._anchor_records is None:
            raise RuntimeError('call set_memory(train_patients) first')
        was_training = self.training
        self.eval()
        try:
            pairs = [self.patient_encoder(p, current_medications=True)
                     for p in self._anchor_records]
            self.memory_keys = torch.cat([r for r, _ in pairs]).detach()
            self.memory_values = torch.cat([m for _, m in pairs]).detach()
        finally:
            self.train(was_training)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in ('memory_keys', 'memory_values'):
            key = prefix + name
            if key in state_dict:
                setattr(self, name, getattr(self, name).new_empty(state_dict[key].shape))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, input, patient_index=None):
        r, historical = self.patient_encoder(input)
        fused = r.new_zeros(1, self.emb_dim)
        if self.use_history and len(input) > 1:
            fused = fused + self.history_projection(historical)
        if self.use_similar:
            if self.memory_keys.shape[0] == 0:
                raise RuntimeError('similar-patient module requires set_memory(train_patients)')
            similarity = F.cosine_similarity(r, self.memory_keys, dim=-1)
            # 训练查询必须给出 train_patients 中的患者索引，以排除自身的
            # anchor（其最后就诊处方可能包含当前/未来标签）。
            if self.training:
                if patient_index is None:
                    raise ValueError('training requires patient_index to exclude the same patient anchor')
                if self._anchor_records is None:
                    raise RuntimeError('after loading, call set_memory before resuming training')
                if not isinstance(patient_index, int) or not 0 <= patient_index < self._training_patient_count:
                    raise ValueError('patient_index must index the train_patients passed to set_memory')
                excluded = torch.tensor([i == patient_index for i in self.anchor_indices],
                                        device=r.device, dtype=torch.bool)
                similarity = similarity.masked_fill(excluded, -torch.inf)
                if not torch.isfinite(similarity).any():
                    raise ValueError('no other patient anchor available; add training patients')
            index = int(similarity.argmax())
            fused = fused + self.similar_projection(self.memory_values[index:index+1])
        if self.use_interaction:
            md = self.di_gcn() - self.ddi_gcn()  # Eq.19，无额外可学习系数。
            weights = F.softmax(self.di_query(r) @ md.t() / math.sqrt(self.emb_dim), dim=-1)
            fused = fused + weights @ md  # 形状对齐的实现选择。
        if self.use_molecular:
            molecules = self.molecular_encoder()
            q = self.query_projection(self.query(r))
            k, v = self.key_projection(molecules), self.value_projection(molecules)
            weights = F.softmax(q @ k.t() / math.sqrt(self.emb_dim), dim=-1)
            fused = fused + weights @ v  # Eq.23-24，补齐 value。
        logits = self.output(fused)
        if self.training:
            probabilities = torch.sigmoid(logits)
            ddi_penalty = (probabilities.t() @ probabilities * self.tensor_ddi_adj).mean()
            return logits, ddi_penalty
        return logits

    def compute_loss(self, logits, medication_ids, alpha=0.9):
        """Eq.26-28，BCE 求和；margin 基于 sigmoid 概率 / |M|。

        alpha 为固定超参数；默认 0.9 是示例设置，论文未指定。
        返回标量；不要再对 logits 先 sigmoid 后调用 BCEWithLogits。
        """
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must be in [0, 1]')
        if logits.shape != (1, self.vocab_size[2]):
            raise ValueError('logits must be [1, num_medications]')
        target = torch.zeros_like(logits)
        ids = torch.as_tensor(medication_ids, dtype=torch.long, device=logits.device)
        target[0, ids] = 1
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='sum')
        probability = torch.sigmoid(logits[0])
        positive, negative = probability[target[0].bool()], probability[~target[0].bool()]
        margin = F.relu(1 - positive[:, None] + negative[None, :]).sum() / self.vocab_size[2]
        return alpha * bce + (1 - alpha) * margin

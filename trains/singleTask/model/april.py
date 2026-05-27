"""
here is the main backbone for APRIL containing feature decoupling and multimodal transformers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
from random import sample


class APRIL(nn.Module):
    def __init__(self, args):
        super(APRIL, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask

        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        output_dim = 1


        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)


        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)




        self.weight_l = nn.Linear(2 * self.d_l, 2 * self.d_l)
        self.weight_v = nn.Linear(2 * self.d_v, 2 * self.d_v)
        self.weight_a = nn.Linear(2 * self.d_a, 2 * self.d_a)


        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)



        self.num_proto_classes = 7


        self.infoNCET = getattr(args, 'infoNCET', 0.02)


        self.tprime_l = self.len_l - args.conv1d_kernel_size_l + 1
        self.tprime_a = self.len_a - args.conv1d_kernel_size_a + 1
        self.tprime_v = self.len_v - args.conv1d_kernel_size_v + 1


        self.register_buffer('proto_text',  torch.zeros(self.num_proto_classes, self.tprime_l, self.d_l))
        self.register_buffer('proto_audio', torch.zeros(self.num_proto_classes, self.tprime_a, self.d_a))
        self.register_buffer('proto_vision',torch.zeros(self.num_proto_classes, self.tprime_v, self.d_v))


        self.register_buffer('proto_ready_text',  torch.zeros(self.num_proto_classes, dtype=torch.bool))
        self.register_buffer('proto_ready_audio', torch.zeros(self.num_proto_classes, dtype=torch.bool))
        self.register_buffer('proto_ready_vision',torch.zeros(self.num_proto_classes, dtype=torch.bool))


        self._init_epoch_caches()


        self.use_personalization = getattr(args, 'use_personalization', False)


        weights_cfg = getattr(args, 'proto_modality_weights', None)
        if isinstance(weights_cfg, dict):
            self.w_text = float(weights_cfg.get('text', 1.0))
            self.w_audio = float(weights_cfg.get('audio', 1.0))
            self.w_vision = float(weights_cfg.get('vision', 1.0))
        elif isinstance(weights_cfg, (list, tuple)) and len(weights_cfg) >= 3:
            self.w_text, self.w_audio, self.w_vision = map(float, weights_cfg[:3])
        else:
            self.w_text = self.w_audio = self.w_vision = 1.0




        if self.use_personalization:

            personal_dropout = getattr(args, 'personal_dropout', 0.2)

            def init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)



            self.t2a_srcproj = nn.Sequential(
                nn.Linear(self.d_l, max(1, self.d_l // 2)),
                nn.LayerNorm(max(1, self.d_l // 2)),
                nn.GELU(),
                nn.Dropout(personal_dropout),
                nn.Linear(max(1, self.d_l // 2), self.d_a)
            )
            self.t2a_step = nn.Linear(2 * self.d_a, self.d_a)
            self.t2a_srcproj.apply(init_weights)
            self.t2a_step.apply(init_weights)

            self.t2v_srcproj = nn.Sequential(
                nn.Linear(self.d_l, max(1, self.d_l // 2)),
                nn.LayerNorm(max(1, self.d_l // 2)),
                nn.GELU(),
                nn.Dropout(personal_dropout),
                nn.Linear(max(1, self.d_l // 2), self.d_v)
            )
            self.t2v_step = nn.Linear(2 * self.d_v, self.d_v)
            self.t2v_srcproj.apply(init_weights)
            self.t2v_step.apply(init_weights)

            self.a2t_srcproj = nn.Sequential(
                nn.Linear(self.d_a, max(1, self.d_a // 2)),
                nn.LayerNorm(max(1, self.d_a // 2)),
                nn.GELU(),
                nn.Dropout(personal_dropout),
                nn.Linear(max(1, self.d_a // 2), self.d_l)
            )
            self.a2t_step = nn.Linear(2 * self.d_l, self.d_l)
            self.a2t_srcproj.apply(init_weights)
            self.a2t_step.apply(init_weights)

            self.a2v_srcproj = nn.Sequential(
                nn.Linear(self.d_a, max(1, self.d_a // 2)),
                nn.LayerNorm(max(1, self.d_a // 2)),
                nn.GELU(),
                nn.Dropout(personal_dropout),
                nn.Linear(max(1, self.d_a // 2), self.d_v)
            )
            self.a2v_step = nn.Linear(2 * self.d_v, self.d_v)
            self.a2v_srcproj.apply(init_weights)
            self.a2v_step.apply(init_weights)

            self.v2t_srcproj = nn.Sequential(
                nn.Linear(self.d_v, max(1, self.d_v // 2)),
                nn.LayerNorm(max(1, self.d_v // 2)),
                nn.GELU(),
                nn.Dropout(personal_dropout),
                nn.Linear(max(1, self.d_v // 2), self.d_l)
            )
            self.v2t_step = nn.Linear(2 * self.d_l, self.d_l)
            self.v2t_srcproj.apply(init_weights)
            self.v2t_step.apply(init_weights)

            self.v2a_srcproj = nn.Sequential(
                nn.Linear(self.d_v, max(1, self.d_v // 2)),
                nn.LayerNorm(max(1, self.d_v // 2)),
                nn.GELU(),
                nn.Dropout(personal_dropout),
                nn.Linear(max(1, self.d_v // 2), self.d_a)
            )
            self.v2a_step = nn.Linear(2 * self.d_a, self.d_a)
            self.v2a_srcproj.apply(init_weights)
            self.v2a_step.apply(init_weights)


            self.personalization_head = getattr(args, 'personalization_head', 'film')
            self.personal_hidden_mult = float(getattr(args, 'personal_hidden_mult', 2.0))

            if self.personalization_head == 'film':

                def build_mlp(dim_in, dim_out):
                    hidden = max(dim_out, int(self.personal_hidden_mult * dim_out))
                    return nn.Sequential(
                        nn.Linear(dim_in, hidden),
                        nn.GELU(),
                        nn.Dropout(personal_dropout),
                        nn.Linear(hidden, dim_out)
                    )


                self.t2a_gamma_mlp = build_mlp(self.d_a, self.d_a)
                self.t2a_beta_mlp  = build_mlp(self.d_a, self.d_a)

                self.t2v_gamma_mlp = build_mlp(self.d_v, self.d_v)
                self.t2v_beta_mlp  = build_mlp(self.d_v, self.d_v)

                self.a2t_gamma_mlp = build_mlp(self.d_l, self.d_l)
                self.a2t_beta_mlp  = build_mlp(self.d_l, self.d_l)

                self.a2v_gamma_mlp = build_mlp(self.d_v, self.d_v)
                self.a2v_beta_mlp  = build_mlp(self.d_v, self.d_v)

                self.v2t_gamma_mlp = build_mlp(self.d_l, self.d_l)
                self.v2t_beta_mlp  = build_mlp(self.d_l, self.d_l)

                self.v2a_gamma_mlp = build_mlp(self.d_a, self.d_a)
                self.v2a_beta_mlp  = build_mlp(self.d_a, self.d_a)


                self.t2a_gamma_mlp.apply(init_weights)
                self.t2a_beta_mlp.apply(init_weights)
                self.t2v_gamma_mlp.apply(init_weights)
                self.t2v_beta_mlp.apply(init_weights)
                self.a2t_gamma_mlp.apply(init_weights)
                self.a2t_beta_mlp.apply(init_weights)
                self.a2v_gamma_mlp.apply(init_weights)
                self.a2v_beta_mlp.apply(init_weights)
                self.v2t_gamma_mlp.apply(init_weights)
                self.v2t_beta_mlp.apply(init_weights)
                self.v2a_gamma_mlp.apply(init_weights)
                self.v2a_beta_mlp.apply(init_weights)
            elif self.personalization_head == 'tconv':
#历史残留，没用到这个tconv啥的，印象中好像是效果不如film
                def build_tconv(in_ch, out_ch):
                    return nn.Sequential(
                        nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
                        nn.GELU(),
                        nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                        nn.Dropout(personal_dropout)
                    )
                self.text_tconv = build_tconv(2 * self.d_l, self.d_l)
                self.audio_tconv = build_tconv(2 * self.d_a, self.d_a)
                self.vision_tconv = build_tconv(2 * self.d_v, self.d_v)

    def _temporal_mean_cosine(self, X_td, P_td):
        """
        X_td: [T, d]
        P_td: [T, d]
        return mean_t cos(X_t, P_t)
        """

        sims = F.cosine_similarity(X_td, P_td, dim=1)
        return sims.mean()



    def _infonce_loss_for_modality(self, proj_x, labels_flat, ready_flags, protos, missing_batch):
        """
        proj_x:   [B, d, T']  (after conv)
        protos:   [C, T', d]  (buffers)
        ready_flags: [C] bool
        If missing_batch is True, return (0,0)
        Returns: (loss_sum, count) over samples contributing to loss
        """
        if missing_batch:
            return proj_x.new_tensor(0.0), 0

        B = proj_x.size(0)
        loss_sum = proj_x.new_tensor(0.0)
        count = 0
        Tprime = proj_x.size(2)


        ready_idx = [c for c in range(self.num_proto_classes) if ready_flags[c].item()]
        if len(ready_idx) <= 1:
            return loss_sum, count

        for i in range(B):
            cls = self._label_to_class(labels_flat[i])
            if not ready_flags[cls].item():
                continue

            x_i = proj_x[i].permute(1, 0)
            x_mean = x_i.mean(dim=0)
            sims = []
            target_pos = None
            for j, c in enumerate(ready_idx):
                P = protos[c]

                p_mean = P.mean(dim=0)
                sim = F.cosine_similarity(x_mean, p_mean, dim=0)
                sims.append(sim)
                if c == cls:
                    target_pos = j
            if target_pos is None:
                continue
            sims_tensor = torch.stack(sims, dim=0) / float(self.infoNCET)

            log_probs = F.log_softmax(sims_tensor, dim=0)
            loss_i = -log_probs[target_pos]
            loss_sum = loss_sum + loss_i
            count += 1

        return loss_sum, count

    def _init_epoch_caches(self):

        self._cache_text = {c: [] for c in range(self.num_proto_classes)}
        self._cache_audio = {c: [] for c in range(self.num_proto_classes)}
        self._cache_vision = {c: [] for c in range(self.num_proto_classes)}

    @staticmethod
    def _label_to_class(y_value):


        if torch.is_tensor(y_value):
            y = float(y_value.item())
        else:
            y = float(y_value)
        idx = round(y + 3.0)
        if idx < 0:
            idx = 0
        if idx > 6:
            idx = 6
        return int(idx)

    @staticmethod
    def _filter_boundary_samples_with_temporal(features, discard_rate=0.05):
        """
        features: Tensor [N, T', d]
        Keep the closest (1-discard_rate) samples to mean center (by Euclidean on per-sample mean over time)
        """
        n = features.size(0)
        if n <= 1:
            return features

        avg_feats = features.mean(dim=1)
        center = avg_feats.mean(dim=0)

        dists = torch.norm(avg_feats - center, dim=1)
        keep_num = max(1, int((1 - discard_rate) * n))
        _, idx = torch.sort(dists)
        idx = idx[:keep_num]
        return features.index_select(dim=0, index=idx)

    def update_prototypes_epoch_end(self, discard_rate=0.05, epoch=None, vis_callback=None, clear_cache=True):
        """
        Aggregate per-epoch caches into prototypes via mean (with boundary filtering).
        Prototypes are stored as buffers and updated with no_grad in-place copy.

        Args:
            discard_rate: boundary sample discard rate.
        """

        cache_counts_text = [len(self._cache_text[c]) for c in range(self.num_proto_classes)]
        cache_counts_audio = [len(self._cache_audio[c]) for c in range(self.num_proto_classes)]
        cache_counts_vision = [len(self._cache_vision[c]) for c in range(self.num_proto_classes)]

        with torch.no_grad():

            for c, lst in self._cache_text.items():
                if len(lst) > 0:
                    stacked = torch.stack(lst, dim=0)
                    filtered = self._filter_boundary_samples_with_temporal(stacked, discard_rate)
                    proto = filtered.mean(dim=0)
                    self.proto_text[c].copy_(proto.to(self.proto_text.device, dtype=self.proto_text.dtype))
                    self.proto_ready_text[c] = True

            for c, lst in self._cache_audio.items():
                if len(lst) > 0:
                    stacked = torch.stack(lst, dim=0)
                    filtered = self._filter_boundary_samples_with_temporal(stacked, discard_rate)
                    proto = filtered.mean(dim=0)
                    self.proto_audio[c].copy_(proto.to(self.proto_audio.device, dtype=self.proto_audio.dtype))
                    self.proto_ready_audio[c] = True

            for c, lst in self._cache_vision.items():
                if len(lst) > 0:
                    stacked = torch.stack(lst, dim=0)
                    filtered = self._filter_boundary_samples_with_temporal(stacked, discard_rate)
                    proto = filtered.mean(dim=0)
                    self.proto_vision[c].copy_(proto.to(self.proto_vision.device, dtype=self.proto_vision.dtype))
                    self.proto_ready_vision[c] = True


        ready_text = int(self.proto_ready_text.sum().item())
        ready_audio = int(self.proto_ready_audio.sum().item())
        ready_vision = int(self.proto_ready_vision.sum().item())
        total_classes = self.num_proto_classes

        print("[Prototype][Text] ready: %d/%d (%.2f%%), cache_total=%d, per_class=%s" % (
            ready_text, total_classes, 100.0 * (ready_text / max(total_classes, 1)),
            sum(cache_counts_text), str(cache_counts_text)))
        print("[Prototype][Audio] ready: %d/%d (%.2f%%), cache_total=%d, per_class=%s" % (
            ready_audio, total_classes, 100.0 * (ready_audio / max(total_classes, 1)),
            sum(cache_counts_audio), str(cache_counts_audio)))
        print("[Prototype][Vision] ready: %d/%d (%.2f%%), cache_total=%d, per_class=%s" % (
            ready_vision, total_classes, 100.0 * (ready_vision / max(total_classes, 1)),
            sum(cache_counts_vision), str(cache_counts_vision)))




        if clear_cache:
            self._init_epoch_caches()

    def _compute_similarity(self, available_modalities, feats_by_modality):
        """
        Strictly follow reference logic:
        - For each available modality m, compute per-class similarity by cosine between temporal means
        - For each modality, pick class with max similarity; across modalities, choose the one with higher max similarity;
          tie-break by lower entropy over softmax(similarities)
        feats_by_modality: { 'text'|'audio'|'vision': Tensor[T', d] }
        Returns: best_class_idx, confidence (1 - entropy/log(C)), best_modality
        """
        best_class_idx = -1
        best_similarity = -float('inf')
        best_entropy = float('inf')
        confidence = 0.0
        best_modality = None

        for modality in available_modalities:
            if modality == 'text':
                protos = self.proto_text
                ready = self.proto_ready_text
            elif modality == 'audio':
                protos = self.proto_audio
                ready = self.proto_ready_audio
            elif modality == 'vision':
                protos = self.proto_vision
                ready = self.proto_ready_vision
            else:
                continue

            X = feats_by_modality[modality].to(protos.device)

            if not torch.any(ready):
                continue

            similarities = torch.full((self.num_proto_classes,), -1e9, device=protos.device, dtype=protos.dtype)
            for c in range(self.num_proto_classes):
                if not ready[c]:
                    continue
                P = protos[c]

                x_mean = X.mean(dim=0)
                p_mean = P.mean(dim=0)
                similarities[c] = F.cosine_similarity(x_mean, p_mean, dim=0)


            sim_probs = torch.softmax(similarities, dim=0)
            entropy = -torch.sum(sim_probs * torch.log(sim_probs + 1e-10))
            max_sim, max_idx = torch.max(similarities, dim=0)
            if (max_sim.item() > best_similarity) or (max_sim.item() == best_similarity and entropy.item() < best_entropy):
                best_class_idx = int(max_idx.item())
                best_similarity = float(max_sim.item())
                best_entropy = float(entropy.item())

                confidence = 1.0 - entropy.item() / float(torch.log(torch.tensor(self.num_proto_classes, dtype=protos.dtype, device=protos.device)).item())
                best_modality = modality

        return best_class_idx, confidence, best_modality

    def _select_best_modality_for_class(self, available_modalities, feats_by_modality, class_idx):
        if class_idx is None or class_idx < 0 or class_idx >= self.num_proto_classes:
            return None

        best_similarity = -float('inf')
        best_modality = None

        for modality in available_modalities:
            if modality == 'text':
                protos = self.proto_text
                ready = self.proto_ready_text
            elif modality == 'audio':
                protos = self.proto_audio
                ready = self.proto_ready_audio
            elif modality == 'vision':
                protos = self.proto_vision
                ready = self.proto_ready_vision
            else:
                continue

            if not ready[class_idx]:
                continue

            X = feats_by_modality[modality].to(protos.device)
            P = protos[class_idx]
            similarity = F.cosine_similarity(X.mean(dim=0), P.mean(dim=0), dim=0)
            if similarity.item() > best_similarity:
                best_similarity = float(similarity.item())
                best_modality = modality

        return best_modality

    def _all_prototypes_ready(self):
        """
        检查是否所有原型都已准备好
        只有当所有模态的所有类别原型都ready时，才返回True
        """
        return (self.proto_ready_text.all() and
                self.proto_ready_audio.all() and
                self.proto_ready_vision.all())

    def _fill_with_prototype(self, proj_x, sample_idx, modality, class_idx):
        """
        proj_x: Tensor [B, d, T']
        Fill sample i with prototype (T', d)^T -> (d, T') if ready, else leave as-is
        """
        if modality == 'text':
            ready = self.proto_ready_text
            proto = self.proto_text
        elif modality == 'audio':
            ready = self.proto_ready_audio
            proto = self.proto_audio
        else:
            ready = self.proto_ready_vision
            proto = self.proto_vision

        if class_idx is not None and class_idx >= 0 and class_idx < self.num_proto_classes and ready[class_idx]:
            with torch.no_grad():
                fill = proto[class_idx].transpose(0, 1).to(proj_x.device, dtype=proj_x.dtype)
                proj_x[sample_idx].copy_(fill)
















    def _get_prototype(self, modality: str, class_idx: int):
        """
        获取指定模态和类别的共性原型
        Returns: Tensor [T', d]
        """
        if modality == 'text':
            return self.proto_text[class_idx]
        elif modality == 'audio':
            return self.proto_audio[class_idx]
        else:
            return self.proto_vision[class_idx]

    def _estimate_class_from_available(self, sample_idx: int, available_modalities: list, proj_x_dict: dict):
        """
        测试期：根据可用模态估计类别
        """
        available_feats = {
            m: proj_x_dict[m][sample_idx].permute(1, 0).detach()
            for m in available_modalities
        }
        best_cls, _, _ = self._compute_similarity(available_modalities, available_feats)
        return best_cls

    def _compute_six_way_reconstruction_loss(self, real_feats_dict: dict, labels):
        """
        Args:
            real_feats_dict: {'text': [B, d_l, T_l'], 'audio': [B, d_a, T_a'], 'vision': [B, d_v, T_v']}
            labels: [B] 或 [B, 1]

        Returns:
            torch.Tensor: 标量，6组损失的平均值
        """
        B = real_feats_dict['text'].size(0)
        labels_flat = labels.view(-1)

        total_loss = 0.0

        for i in range(B):

            cls = self._label_to_class(labels_flat[i])


            sample_losses = []


            c_src_t = real_feats_dict['text'][i].mean(dim=1)
            c_hat_t2a = self.t2a_srcproj(c_src_t)
            proto_a = self._get_prototype('audio', cls)
            T_a = proto_a.size(0)
            if getattr(self, 'personalization_head', 'film') == 'film':
                gamma = self.t2a_gamma_mlp(c_hat_t2a)
                beta  = self.t2a_beta_mlp(c_hat_t2a)
                Gamma = gamma.unsqueeze(0).expand(T_a, -1)
                Beta  = beta.unsqueeze(0).expand(T_a, -1)
                personal_proto_a = Gamma * proto_a + Beta
            else:
                C = c_hat_t2a.unsqueeze(0).expand(T_a, -1)
                Z = torch.cat([proto_a.detach(), C], dim=-1).permute(1, 0).unsqueeze(0)
                Delta = self.audio_tconv(Z).squeeze(0).permute(1, 0)
                personal_proto_a = proto_a + Delta
            real_feat_a = real_feats_dict['audio'][i].permute(1, 0)
            loss_t2a = F.mse_loss(personal_proto_a, real_feat_a)
            sample_losses.append(loss_t2a)


            c_src_t = real_feats_dict['text'][i].mean(dim=1)
            c_hat_t2v = self.t2v_srcproj(c_src_t)
            proto_v = self._get_prototype('vision', cls)
            T_v = proto_v.size(0)
            if getattr(self, 'personalization_head', 'film') == 'film':
                gamma = self.t2v_gamma_mlp(c_hat_t2v)
                beta  = self.t2v_beta_mlp(c_hat_t2v)
                Gamma = gamma.unsqueeze(0).expand(T_v, -1)
                Beta  = beta.unsqueeze(0).expand(T_v, -1)
                personal_proto_v = Gamma * proto_v + Beta
            else:
                C = c_hat_t2v.unsqueeze(0).expand(T_v, -1)
                Z = torch.cat([proto_v.detach(), C], dim=-1).permute(1, 0).unsqueeze(0)
                Delta = self.vision_tconv(Z).squeeze(0).permute(1, 0)
                personal_proto_v = proto_v + Delta
            real_feat_v = real_feats_dict['vision'][i].permute(1, 0)
            loss_t2v = F.mse_loss(personal_proto_v, real_feat_v)
            sample_losses.append(loss_t2v)


            c_src_a = real_feats_dict['audio'][i].mean(dim=1)
            c_hat_a2t = self.a2t_srcproj(c_src_a)
            proto_t = self._get_prototype('text', cls)
            T_l = proto_t.size(0)
            if getattr(self, 'personalization_head', 'film') == 'film':
                gamma = self.a2t_gamma_mlp(c_hat_a2t)
                beta  = self.a2t_beta_mlp(c_hat_a2t)
                Gamma = gamma.unsqueeze(0).expand(T_l, -1)
                Beta  = beta .unsqueeze(0).expand(T_l, -1)
                personal_proto_t = Gamma * proto_t + Beta
            else:
                C = c_hat_a2t.unsqueeze(0).expand(T_l, -1)
                Z = torch.cat([proto_t.detach(), C], dim=-1).permute(1, 0).unsqueeze(0)
                Delta = self.text_tconv(Z).squeeze(0).permute(1, 0)
                personal_proto_t = proto_t + Delta
            real_feat_t = real_feats_dict['text'][i].permute(1, 0)
            loss_a2t = F.mse_loss(personal_proto_t, real_feat_t)
            sample_losses.append(loss_a2t)


            c_src_a = real_feats_dict['audio'][i].mean(dim=1)
            c_hat_a2v = self.a2v_srcproj(c_src_a)
            proto_v = self._get_prototype('vision', cls)
            T_v = proto_v.size(0)
            if getattr(self, 'personalization_head', 'film') == 'film':
                gamma = self.a2v_gamma_mlp(c_hat_a2v)
                beta  = self.a2v_beta_mlp(c_hat_a2v)
                Gamma = gamma.unsqueeze(0).expand(T_v, -1)
                Beta  = beta .unsqueeze(0).expand(T_v, -1)
                personal_proto_v = Gamma * proto_v + Beta
            else:
                C = c_hat_a2v.unsqueeze(0).expand(T_v, -1)
                Z = torch.cat([proto_v.detach(), C], dim=-1).permute(1, 0).unsqueeze(0)
                Delta = self.vision_tconv(Z).squeeze(0).permute(1, 0)
                personal_proto_v = proto_v + Delta
            real_feat_v = real_feats_dict['vision'][i].permute(1, 0)
            loss_a2v = F.mse_loss(personal_proto_v, real_feat_v)
            sample_losses.append(loss_a2v)


            c_src_v = real_feats_dict['vision'][i].mean(dim=1)
            c_hat_v2t = self.v2t_srcproj(c_src_v)
            proto_t = self._get_prototype('text', cls)
            T_l = proto_t.size(0)
            if getattr(self, 'personalization_head', 'film') == 'film':
                gamma = self.v2t_gamma_mlp(c_hat_v2t)
                beta  = self.v2t_beta_mlp(c_hat_v2t)
                Gamma = gamma.unsqueeze(0).expand(T_l, -1)
                Beta  = beta .unsqueeze(0).expand(T_l, -1)
                personal_proto_t = Gamma * proto_t + Beta
            else:
                C = c_hat_v2t.unsqueeze(0).expand(T_l, -1)
                Z = torch.cat([proto_t.detach(), C], dim=-1).permute(1, 0).unsqueeze(0)
                Delta = self.text_tconv(Z).squeeze(0).permute(1, 0)
                personal_proto_t = proto_t + Delta
            real_feat_t = real_feats_dict['text'][i].permute(1, 0)
            loss_v2t = F.mse_loss(personal_proto_t, real_feat_t)
            sample_losses.append(loss_v2t)


            c_src_v = real_feats_dict['vision'][i].mean(dim=1)
            c_hat_v2a = self.v2a_srcproj(c_src_v)
            proto_a = self._get_prototype('audio', cls)
            T_a = proto_a.size(0)
            if getattr(self, 'personalization_head', 'film') == 'film':
                gamma = self.v2a_gamma_mlp(c_hat_v2a)
                beta  = self.v2a_beta_mlp(c_hat_v2a)
                Gamma = gamma.unsqueeze(0).expand(T_a, -1)
                Beta  = beta .unsqueeze(0).expand(T_a, -1)
                personal_proto_a = Gamma * proto_a + Beta
            else:
                C = c_hat_v2a.unsqueeze(0).expand(T_a, -1)
                Z = torch.cat([proto_a.detach(), C], dim=-1).permute(1, 0).unsqueeze(0)
                Delta = self.audio_tconv(Z).squeeze(0).permute(1, 0)
                personal_proto_a = proto_a + Delta
            real_feat_a = real_feats_dict['audio'][i].permute(1, 0)
            loss_v2a = F.mse_loss(personal_proto_a, real_feat_a)
            sample_losses.append(loss_v2a)


            sample_avg_loss = torch.stack(sample_losses).mean()
            total_loss += sample_avg_loss


        return total_loss / B

    def _personalize_and_fill_missing_modality(
        self,
        target_modality: str,
        sample_idx: int,
        available_modalities: list,
        proj_x_dict: dict,
        class_idx: int,

    ):
        """
        Args:
            target_modality: 'text'/'audio'/'vision'
            sample_idx: 样本索引
            available_modalities: 可用模态列表
            proj_x_dict: {'text': proj_x_l, 'audio': proj_x_a, 'vision': proj_x_v}
            class_idx: 该样本的类别
        """
        if len(available_modalities) == 0:
            return None


        available_feats = {
            m: proj_x_dict[m][sample_idx].permute(1, 0).detach()
            for m in available_modalities
        }
        best_mod = self._select_best_modality_for_class(available_modalities, available_feats, class_idx)
        if best_mod is None:
            return None

        src_seq = proj_x_dict[best_mod][sample_idx]
        c_src = src_seq.mean(dim=1).detach()
        if best_mod == 'text':
            if target_modality == 'audio':
                c_hat = self.t2a_srcproj(c_src)
            else:
                c_hat = self.t2v_srcproj(c_src)
        elif best_mod == 'audio':
            if target_modality == 'text':
                c_hat = self.a2t_srcproj(c_src)
            else:
                c_hat = self.a2v_srcproj(c_src)
        else:
            if target_modality == 'text':
                c_hat = self.v2t_srcproj(c_src)
            else:
                c_hat = self.v2a_srcproj(c_src)


        proto = self._get_prototype(target_modality, class_idx)
        T_tgt, d_tgt = proto.size(0), proto.size(1)
        if getattr(self, 'personalization_head', 'film') == 'film':

            if best_mod == 'text':
                if target_modality == 'audio':
                    gamma = self.t2a_gamma_mlp(c_hat)
                    beta  = self.t2a_beta_mlp(c_hat)
                else:
                    gamma = self.t2v_gamma_mlp(c_hat)
                    beta  = self.t2v_beta_mlp(c_hat)
            elif best_mod == 'audio':
                if target_modality == 'text':
                    gamma = self.a2t_gamma_mlp(c_hat)
                    beta  = self.a2t_beta_mlp(c_hat)
                else:
                    gamma = self.a2v_gamma_mlp(c_hat)
                    beta  = self.a2v_beta_mlp(c_hat)
            else:
                if target_modality == 'text':
                    gamma = self.v2t_gamma_mlp(c_hat)
                    beta  = self.v2t_beta_mlp(c_hat)
                else:
                    gamma = self.v2a_gamma_mlp(c_hat)
                    beta  = self.v2a_beta_mlp(c_hat)

            Gamma = gamma.unsqueeze(0).expand(T_tgt, -1)
            Beta  = beta.unsqueeze(0).expand(T_tgt, -1)
            base_out = Gamma * proto + Beta
            Delta = base_out - proto

            personal_proto = proto + Delta
        elif getattr(self, 'personalization_head', 'film') == 'tconv':

            c_rep = c_hat.detach().unsqueeze(0).expand(T_tgt, -1)
            Z = torch.cat([proto.detach(), c_rep], dim=-1)
            Z_c = Z.permute(1, 0).unsqueeze(0)
            if target_modality == 'text':
                Delta = self.text_tconv(Z_c).squeeze(0).permute(1, 0)
            elif target_modality == 'audio':
                Delta = self.audio_tconv(Z_c).squeeze(0).permute(1, 0)
            else:
                Delta = self.vision_tconv(Z_c).squeeze(0).permute(1, 0)

            personal_proto = proto + Delta
        else:

            if best_mod == 'text':
                if target_modality == 'audio':
                    gamma = self.t2a_gamma_mlp(c_hat)
                    beta  = self.t2a_beta_mlp(c_hat)
                else:
                    gamma = self.t2v_gamma_mlp(c_hat)
                    beta  = self.t2v_beta_mlp(c_hat)
            elif best_mod == 'audio':
                if target_modality == 'text':
                    gamma = self.a2t_gamma_mlp(c_hat)
                    beta  = self.a2t_beta_mlp(c_hat)
                else:
                    gamma = self.a2v_gamma_mlp(c_hat)
                    beta  = self.a2v_beta_mlp(c_hat)
            else:
                if target_modality == 'text':
                    gamma = self.v2t_gamma_mlp(c_hat)
                    beta  = self.v2t_beta_mlp(c_hat)
                else:
                    gamma = self.v2a_gamma_mlp(c_hat)
                    beta  = self.v2a_beta_mlp(c_hat)
            Gamma = gamma.unsqueeze(0).expand(T_tgt, -1)
            Beta  = beta.unsqueeze(0).expand(T_tgt, -1)
            personal_proto = Gamma * proto + Beta


        proj_x_dict[target_modality][sample_idx] = personal_proto.transpose(0, 1)



    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, audio, video, num_modal=None, labels=None):
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)


        if self.training and (labels is not None):
            labels_flat = labels.view(-1)
            B = proj_x_l.size(0)

            feats_l = proj_x_l.permute(0, 2, 1).detach().cpu()
            feats_a = proj_x_a.permute(0, 2, 1).detach().cpu()
            feats_v = proj_x_v.permute(0, 2, 1).detach().cpu()
            for i in range(B):
                cls = self._label_to_class(labels_flat[i])
                self._cache_text[cls].append(feats_l[i])
                self._cache_audio[cls].append(feats_a[i])
                self._cache_vision[cls].append(feats_v[i])


            loss_t_sum, cnt_t = self._infonce_loss_for_modality(
                proj_x_l, labels_flat, self.proto_ready_text, self.proto_text, False)
            loss_a_sum, cnt_a = self._infonce_loss_for_modality(
                proj_x_a, labels_flat, self.proto_ready_audio, self.proto_audio, False)
            loss_v_sum, cnt_v = self._infonce_loss_for_modality(
                proj_x_v, labels_flat, self.proto_ready_vision, self.proto_vision, False)


            if (cnt_t + cnt_a + cnt_v) == 0:
                self.prototype_contrastive_loss = proj_x_l.new_tensor(0.0)
            else:
                L_t = loss_t_sum / max(cnt_t, 1)
                L_a = loss_a_sum / max(cnt_a, 1)
                L_v = loss_v_sum / max(cnt_v, 1)

                w_t = proj_x_l.new_tensor(self.w_text)
                w_a = proj_x_l.new_tensor(self.w_audio)
                w_v = proj_x_l.new_tensor(self.w_vision)

                I_t = 1.0 if cnt_t > 0 else 0.0
                I_a = 1.0 if cnt_a > 0 else 0.0
                I_v = 1.0 if cnt_v > 0 else 0.0

                num = w_t * L_t * I_t + w_a * L_a * I_a + w_v * L_v * I_v
                den = w_t * I_t + w_a * I_a + w_v * I_v
                self.prototype_contrastive_loss = num / (den if den.item() > 0 else proj_x_l.new_tensor(1.0))
        else:

            self.prototype_contrastive_loss = proj_x_l.new_tensor(0.0)



        if self.use_personalization and self._all_prototypes_ready() and self.training and labels is not None:
            real_feats_dict = {
                'text': proj_x_l.detach().clone(),
                'audio': proj_x_a.detach().clone(),
                'vision': proj_x_v.detach().clone()
            }
        else:
            real_feats_dict = None

        if num_modal is None:
            num_modal = 3
        modal_idx = [0, 1, 2]
        ava_modal_idx = sample(modal_idx, num_modal)
        if num_modal == 1:
            if ava_modal_idx[0] == 0:
                proj_x_a = torch.zeros_like(proj_x_a)
                proj_x_v = torch.zeros_like(proj_x_v)
            elif ava_modal_idx[0] == 1:
                proj_x_l = torch.zeros_like(proj_x_l)
                proj_x_a = torch.zeros_like(proj_x_a)
            else:
                proj_x_l = torch.zeros_like(proj_x_l)
                proj_x_v = torch.zeros_like(proj_x_v)
        if num_modal == 2:
            if set(modal_idx) - set(ava_modal_idx) == {0}:
                proj_x_l = torch.zeros_like(proj_x_l)
            if set(modal_idx) - set(ava_modal_idx) == {1}:
                proj_x_v = torch.zeros_like(proj_x_v)
            if set(modal_idx) - set(ava_modal_idx) == {2}:
                proj_x_a = torch.zeros_like(proj_x_a)
        if num_modal == 3:
            pass


        B = proj_x_l.size(0)
        proj_x_dict = {'text': proj_x_l, 'audio': proj_x_a, 'vision': proj_x_v}


        missing_text_batch = (proj_x_l.abs().sum() == 0)
        missing_audio_batch = (proj_x_a.abs().sum() == 0)
        missing_vision_batch = (proj_x_v.abs().sum() == 0)



        if self.use_personalization and self._all_prototypes_ready():




            if missing_text_batch:
                available = []
                if not missing_audio_batch:
                    available.append('audio')
                if not missing_vision_batch:
                    available.append('vision')

                for i in range(B):
                    if self.training and labels is not None:
                        cls = self._label_to_class(labels[i])
                    else:
                        cls = self._estimate_class_from_available(i, available, proj_x_dict)

                    self._personalize_and_fill_missing_modality(
                        'text', i, available, proj_x_dict, cls)


            if missing_audio_batch:
                available = []
                if not missing_text_batch:
                    available.append('text')
                if not missing_vision_batch:
                    available.append('vision')

                for i in range(B):
                    if self.training and labels is not None:
                        cls = self._label_to_class(labels[i])
                    else:
                        cls = self._estimate_class_from_available(i, available, proj_x_dict)

                    self._personalize_and_fill_missing_modality(
                        'audio', i, available, proj_x_dict, cls)


            if missing_vision_batch:
                available = []
                if not missing_text_batch:
                    available.append('text')
                if not missing_audio_batch:
                    available.append('audio')

                for i in range(B):
                    if self.training and labels is not None:
                        cls = self._label_to_class(labels[i])
                    else:
                        cls = self._estimate_class_from_available(i, available, proj_x_dict)

                    self._personalize_and_fill_missing_modality(
                        'vision', i, available, proj_x_dict, cls)

        else:

            if self.training and (labels is not None):
                labels_flat = labels.view(-1)

                for i in range(B):
                    cls = self._label_to_class(labels_flat[i])
                    if missing_text_batch:
                        self._fill_with_prototype(proj_x_l, i, 'text', cls)
                    if missing_audio_batch:
                        self._fill_with_prototype(proj_x_a, i, 'audio', cls)
                    if missing_vision_batch:
                        self._fill_with_prototype(proj_x_v, i, 'vision', cls)
            else:

                for i in range(B):
                    available_modalities = []
                    feats = {}
                    if not missing_text_batch:
                        available_modalities.append('text')
                        feats['text'] = proj_x_l[i].permute(1, 0)
                    if not missing_audio_batch:
                        available_modalities.append('audio')
                        feats['audio'] = proj_x_a[i].permute(1, 0)
                    if not missing_vision_batch:
                        available_modalities.append('vision')
                        feats['vision'] = proj_x_v[i].permute(1, 0)

                    best_cls, _, _ = self._compute_similarity(available_modalities, feats) if len(available_modalities) > 0 else (-1, 0.0, None)
                    if missing_text_batch:
                        self._fill_with_prototype(proj_x_l, i, 'text', best_cls)
                    if missing_audio_batch:
                        self._fill_with_prototype(proj_x_a, i, 'audio', best_cls)
                    if missing_vision_batch:
                        self._fill_with_prototype(proj_x_v, i, 'vision', best_cls)


            self.personal_recon_loss = proj_x_l.new_tensor(0.0)




        s_l = proj_x_l.permute(2, 0, 1)
        s_v = proj_x_v.permute(2, 0, 1)
        s_a = proj_x_a.permute(2, 0, 1)





        h_l_with_as = self.trans_l_with_a(s_l, s_a, s_a)
        h_l_with_vs = self.trans_l_with_v(s_l, s_v, s_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]


        h_a_with_ls = self.trans_a_with_l(s_a, s_l, s_l)
        h_a_with_vs = self.trans_a_with_v(s_a, s_v, s_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]


        h_v_with_ls = self.trans_v_with_l(s_v, s_l, s_l)
        h_v_with_as = self.trans_v_with_a(s_v, s_a, s_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]




        last_h_l = torch.sigmoid(self.weight_l(last_h_l))
        last_h_v = torch.sigmoid(self.weight_v(last_h_v))
        last_h_a = torch.sigmoid(self.weight_a(last_h_a))



        last_hs = torch.cat([last_h_l, last_h_v, last_h_a], dim=1)
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)




        if self.use_personalization and self._all_prototypes_ready() and self.training and real_feats_dict is not None:
            self.personal_recon_loss = self._compute_six_way_reconstruction_loss(
                real_feats_dict, labels)
        else:
            self.personal_recon_loss = proj_x_l.new_tensor(0.0)

        res = {
            'output_logit': output
        }
        return res

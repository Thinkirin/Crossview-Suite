"""
Loss functions for CrossViewer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Contrastive) Loss for cross-view matching

    Pulls together matching pairs, pushes apart non-matching pairs
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings_A, embeddings_B):
        """
        Args:
            embeddings_A: [B, K, D] - Ego view embeddings (normalized)
            embeddings_B: [B, K, D] - Exo view embeddings (normalized)

        Returns:
            loss: scalar
            acc: accuracy of positive pairs
        """
        B, K, D = embeddings_A.shape

        # Flatten to [N, D] where N = B * K
        emb_A = embeddings_A.view(B * K, D)
        emb_B = embeddings_B.view(B * K, D)

        N = B * K

        # Compute similarity matrix: [N, N]
        sim_matrix = torch.matmul(emb_A, emb_B.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(N, device=emb_A.device)

        # Cross-entropy loss (A -> B direction)
        loss_A2B = F.cross_entropy(sim_matrix, labels)

        # B -> A direction (symmetric)
        loss_B2A = F.cross_entropy(sim_matrix.T, labels)

        # Total loss
        loss = (loss_A2B + loss_B2A) / 2

        # Compute accuracy (for monitoring)
        with torch.no_grad():
            pred = sim_matrix.argmax(dim=1)
            acc = (pred == labels).float().mean()

        return loss, acc


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Multi-Positive)
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [N, D] normalized embeddings
            labels: [N] integer labels, -1 indicates ignore

        Returns:
            loss: scalar
        """
        if features.numel() == 0:
            return torch.tensor(0.0, device=features.device)

        labels = labels.view(-1)
        valid = labels >= 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        features = features[valid]
        labels = labels[valid]

        N = features.shape[0]
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()

        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        logits_mask = torch.ones_like(mask, dtype=torch.bool)
        logits_mask.fill_diagonal_(False)
        mask = mask & logits_mask

        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        pos_count = mask.sum(dim=1)
        valid_pos = pos_count > 0
        if valid_pos.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        mean_log_prob_pos = (mask.float() * log_prob).sum(dim=1) / (pos_count + 1e-8)
        loss = -mean_log_prob_pos[valid_pos].mean()

        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss: (anchor, positive, negative)
    """

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: [N, D]
            positive: [N, D]
            negative: [N, D]

        Returns:
            loss: scalar
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()


class HardNegativeMining:
    """
    Helper class to mine hard negatives from a batch
    """

    @staticmethod
    def get_hard_negatives(embeddings_A, embeddings_B):
        """
        For each object in A, find the hardest negative in B
        (the non-matching object with highest similarity)

        Args:
            embeddings_A: [B, K, D]
            embeddings_B: [B, K, D]

        Returns:
            hard_negs: [B, K, D] - hardest negative for each object in A
        """
        B, K, D = embeddings_A.shape

        hard_negatives = []

        for b in range(B):
            emb_A = embeddings_A[b]  # [K, D]
            emb_B = embeddings_B[b]  # [K, D]

            # Compute similarity matrix [K, K]
            sim = torch.matmul(emb_A, emb_B.T)

            # Mask out diagonal (positive pairs)
            mask = torch.eye(K, device=sim.device).bool()
            sim_masked = sim.masked_fill(mask, -float('inf'))

            # Get hardest negative for each anchor
            hard_neg_idx = sim_masked.argmax(dim=1)  # [K]

            # Select hard negatives
            hard_neg = emb_B[hard_neg_idx]  # [K, D]

            hard_negatives.append(hard_neg)

        hard_negatives = torch.stack(hard_negatives, dim=0)  # [B, K, D]

        return hard_negatives


class CombinedLoss(nn.Module):
    """
    Combined loss: InfoNCE + Hard Negative Triplet
    """

    def __init__(self,
                 temperature=0.07,
                 margin=0.5,
                 info_nce_weight=1.0,
                 triplet_weight=0.1,
                 use_supcon=True):
        super().__init__()

        self.info_nce = InfoNCELoss(temperature=temperature)
        self.supcon = SupConLoss(temperature=temperature)
        self.triplet = TripletLoss(margin=margin)
        self.info_nce_weight = info_nce_weight
        self.triplet_weight = triplet_weight
        self.use_supcon = use_supcon

    def forward(self, embeddings_A, embeddings_B, valid_mask=None, labels=None):
        """
        Args:
            embeddings_A: [B, K, D]
            embeddings_B: [B, K, D]
            valid_mask: Optional [B, K] - mask for valid objects (1=valid, 0=padded)
            labels: Optional [B, K] - integer labels for SupCon (-1 to ignore)

        Returns:
            total_loss: scalar
            loss_dict: dict of individual losses
        """
        # If mask provided, filter out padded objects
        if valid_mask is not None:
            # Flatten mask
            B, K, D = embeddings_A.shape
            mask_flat = valid_mask.view(-1).bool()  # [B*K]

            # Flatten embeddings and select valid ones
            emb_A_flat = embeddings_A.view(B * K, D)[mask_flat]  # [N_valid, D]
            emb_B_flat = embeddings_B.view(B * K, D)[mask_flat]  # [N_valid, D]
            label_flat = None
            if labels is not None:
                label_flat = labels.view(-1)[mask_flat]
                label_valid = label_flat >= 0
                emb_A_flat = emb_A_flat[label_valid]
                emb_B_flat = emb_B_flat[label_valid]
                label_flat = label_flat[label_valid]

            # Reshape back to batch format for InfoNCE (assuming equal K per sample)
            # For simplicity, compute loss on flattened valid embeddings
            N_valid = emb_A_flat.shape[0]

            if N_valid == 0:
                info_nce_loss = torch.tensor(0.0, device=emb_A_flat.device)
                triplet_loss = torch.tensor(0.0, device=emb_A_flat.device)
                acc = torch.tensor(0.0, device=emb_A_flat.device)
            else:
                if self.use_supcon and label_flat is not None:
                    features = torch.cat([emb_A_flat, emb_B_flat], dim=0)
                    features = F.normalize(features, dim=-1)
                    labels_sup = torch.cat([label_flat, label_flat], dim=0)
                    info_nce_loss = self.supcon(features, labels_sup)

                    with torch.no_grad():
                        sim_matrix = torch.matmul(emb_A_flat, emb_B_flat.T)
                        pred = sim_matrix.argmax(dim=1)
                        acc = (label_flat[pred] == label_flat).float().mean()
                else:
                    # InfoNCE on valid embeddings
                    sim_matrix = torch.matmul(emb_A_flat, emb_B_flat.T) / self.info_nce.temperature
                    labels_idx = torch.arange(N_valid, device=emb_A_flat.device)
                    loss_A2B = F.cross_entropy(sim_matrix, labels_idx)
                    loss_B2A = F.cross_entropy(sim_matrix.T, labels_idx)
                    info_nce_loss = (loss_A2B + loss_B2A) / 2

                    with torch.no_grad():
                        pred = sim_matrix.argmax(dim=1)
                        acc = (pred == labels_idx).float().mean()

                # Triplet loss on valid embeddings
                if self.triplet_weight > 0 and N_valid > 1:
                    # Simple hard negative mining on flattened embeddings
                    sim = torch.matmul(emb_A_flat, emb_B_flat.T)
                    mask_diag = torch.eye(N_valid, device=sim.device).bool()
                    sim_masked = sim.masked_fill(mask_diag, -float('inf'))
                    hard_neg_idx = sim_masked.argmax(dim=1)
                    hard_negs = emb_B_flat[hard_neg_idx]

                    triplet_loss = self.triplet(emb_A_flat, emb_B_flat, hard_negs)
                else:
                    triplet_loss = torch.tensor(0.0, device=emb_A_flat.device)
        else:
            # Original implementation without masking
            if self.use_supcon and labels is not None:
                B, K, D = embeddings_A.shape
                emb_A_flat = embeddings_A.view(B * K, D)
                emb_B_flat = embeddings_B.view(B * K, D)
                label_flat = labels.view(-1)
                label_valid = label_flat >= 0
                emb_A_flat = emb_A_flat[label_valid]
                emb_B_flat = emb_B_flat[label_valid]
                label_flat = label_flat[label_valid]

                if emb_A_flat.numel() == 0:
                    info_nce_loss = torch.tensor(0.0, device=embeddings_A.device)
                    acc = torch.tensor(0.0, device=embeddings_A.device)
                else:
                    features = torch.cat([emb_A_flat, emb_B_flat], dim=0)
                    features = F.normalize(features, dim=-1)
                    labels_sup = torch.cat([label_flat, label_flat], dim=0)
                    info_nce_loss = self.supcon(features, labels_sup)

                    with torch.no_grad():
                        sim_matrix = torch.matmul(emb_A_flat, emb_B_flat.T)
                        pred = sim_matrix.argmax(dim=1)
                        acc = (label_flat[pred] == label_flat).float().mean()
            else:
                info_nce_loss, acc = self.info_nce(embeddings_A, embeddings_B)

            # Hard negative mining
            hard_negs = HardNegativeMining.get_hard_negatives(embeddings_A, embeddings_B)

            # Triplet loss
            B, K, D = embeddings_A.shape
            anchor = embeddings_A.view(B * K, D)
            positive = embeddings_B.view(B * K, D)
            negative = hard_negs.view(B * K, D)

            triplet_loss = self.triplet(anchor, positive, negative)

        # Total loss
        total_loss = (self.info_nce_weight * info_nce_loss +
                      self.triplet_weight * triplet_loss)

        loss_dict = {
            'total': total_loss.item(),
            'info_nce': info_nce_loss.item(),
            'triplet': triplet_loss.item() if isinstance(triplet_loss, torch.Tensor) else 0.0,
            'accuracy': acc.item()
        }

        return total_loss, loss_dict


if __name__ == "__main__":
    # Test losses
    print("Testing InfoNCELoss...")
    loss_fn = InfoNCELoss()
    emb_A = F.normalize(torch.randn(4, 5, 256), dim=-1)  # 4 images, 5 objects each
    emb_B = F.normalize(torch.randn(4, 5, 256), dim=-1)
    loss, acc = loss_fn(emb_A, emb_B)
    print(f"✓ InfoNCE Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")

    print("\nTesting TripletLoss...")
    triplet_fn = TripletLoss()
    anchor = F.normalize(torch.randn(20, 256), dim=-1)
    positive = F.normalize(torch.randn(20, 256), dim=-1)
    negative = F.normalize(torch.randn(20, 256), dim=-1)
    loss = triplet_fn(anchor, positive, negative)
    print(f"✓ Triplet Loss: {loss.item():.4f}")

    print("\nTesting CombinedLoss...")
    combined_fn = CombinedLoss()
    total_loss, loss_dict = combined_fn(emb_A, emb_B)
    print(f"✓ Combined Loss: {loss_dict}")

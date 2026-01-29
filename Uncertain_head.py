import torch


def _span_visual_text(key_position: dict, K: int):
    """
    Unify indices into half-open intervals:
      visual keys: [v_s, v_e)
      sys keys:    [0, img_s)
      text keys:   [t_s, t_e)  (Default from img_e to end)
    Note: Handing Shikra's specific indexing vs standard models.
    """
    img_s = int(key_position.get("image_start", 0))
    img_e = int(key_position.get("image_end", -1))

    # Shikra specific position markers
    is_shikra = (img_s == 33 and img_e == 290)

    if img_e >= 0:
        if is_shikra:
            v_s = max(0, min(img_s + 1, K))
            v_e = max(v_s, min(img_e, K))  # [img_s+1, img_e)
        else:
            v_s = max(0, min(img_s, K))
            v_e = max(v_s, min(img_e + 1, K))  # [img_s, img_e+1)
    else:
        v_s, v_e = 0, 0

    t_s = max(0, min((img_e + 1) if img_e >= 0 else 0, K))
    t_e = K
    return v_s, v_e, t_s, t_e, img_s


def _entropy_from_probs(p: torch.Tensor, eps: float = 1e-12):
    # p: [..., N], non-negative, performs subset normalization internally
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    return -(p * (p + eps).log()).sum(dim=-1)


@torch.no_grad()
def compute_fusion_stage_metrics_vaf(
        attn_probs: torch.Tensor,  # [B, H, Q, K] after softmax
        key_position: dict,
        layer_idx: int,
        tail_n: int = 1,  # Focus on the last tail_n text queries after image
        eps: float = 1e-12,
):
    """
    VAF/CS style statistics:
    - Focuses on decision-related queries: text-tail and prefill-last.
    - Returns [B, H] tensors for head-wise analysis (AUC / effect size).
    """
    if attn_probs is None or attn_probs.dim() != 4:
        return None

    B, H, Q, K = attn_probs.shape
    v_s, v_e, t_s, t_e, img_s = _span_visual_text(key_position, K)

    if v_e <= v_s:
        return None

    # ---------- Select Query Sets ----------
    # tail: Take the last tail_n tokens of the text section following image_end
    q_tail_s = max(t_s, t_e - tail_n)
    q_tail_e = t_e

    # Fallback for extremely short text
    if q_tail_e <= q_tail_s:
        q_tail_s = max(0, Q - 1)
        q_tail_e = Q

    # last: The very last query of the prefill (last prompt token)
    q_last_s = Q - 1
    q_last_e = Q

    def _agg(qs, qe):
        # Clamp to query dimensions to avoid empty slices during decoding (Q=1)
        qs = int(qs)
        qe = int(qe)

        qs = max(0, min(qs, Q))
        qe = max(0, min(qe, Q))

        if qe <= qs:
            qs = max(0, Q - 1)
            qe = Q

        if qs >= Q:
            return None

        A = attn_probs[:, :, qs:qe, :]  # [B, H, Qq, K] where Qq >= 1

        # Mass calculations
        mV = A[:, :, :, v_s:v_e].sum(dim=-1).mean(dim=-1)
        mSys = A[:, :, :, 0:img_s].sum(dim=-1).mean(dim=-1)

        # Visual entropy: Normalized subset entropy averaged over queries
        pV = A[:, :, :, v_s:v_e]
        mV_q = pV.sum(dim=-1, keepdim=True)
        pV_norm = pV / (mV_q + eps)
        HVis = _entropy_from_probs(pV_norm, eps=eps).mean(dim=-1)

        return mV, mSys, HVis

    # Aggregate for both query sets
    res_tail = _agg(q_tail_s, q_tail_e)
    res_last = _agg(q_last_s, q_last_e)

    if res_tail is None or res_last is None:
        return None

    mV_tail, mSys_tail, HVis_tail = res_tail
    mV_last, mSys_last, HVis_last = res_last

    # Risk metric: higher reliance on system prompt vs visual tokens
    risk_tail = mSys_tail - mV_tail
    risk_last = mSys_last - mV_last

    return {
        "layer_idx": layer_idx,
        "mV_tail": mV_tail,  # [B, H]
        "mSys_tail": mSys_tail,  # [B, H]
        "HVis_tail": HVis_tail,  # [B, H]
        "risk_tail": risk_tail,  # [B, H]

        "mV_last": mV_last,  # [B, H]
        "mSys_last": mSys_last,  # [B, H]
        "HVis_last": HVis_last,  # [B, H]
        "risk_last": risk_last,  # [B, H]
    }
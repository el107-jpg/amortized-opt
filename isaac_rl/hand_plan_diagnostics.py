"""Read-only entropy and norm diagnostics for supervised hand-plan ranking."""
import numpy as np
import torch


def prediction_diagnostics(scores, observed, rewards):
    """Entropy of temperature-one softmax over legal, physically labeled plans.

    These are diagnostics of ranker logits, not calibrated success probabilities.
    Normalized entropy excludes singleton banks, whose choice is forced. Target
    entropy uses the same .02 near-best tolerance as the supervised objective.
    """
    scores=np.asarray(scores,dtype=np.float64);observed=np.asarray(observed,bool)
    if scores.shape!=observed.shape or not observed.any(1).all():
        raise ValueError('Each case must have a nonempty labeled plan bank')
    if not np.isfinite(scores[observed]).all():raise ValueError('Nonfinite legal scores')
    logits=np.where(observed,scores,-np.inf)
    shifted=logits-logits.max(1,keepdims=True)
    probabilities=np.exp(shifted);probabilities/=probabilities.sum(1,keepdims=True)
    log_probabilities=np.zeros_like(probabilities)
    np.log(probabilities,out=log_probabilities,where=probabilities>0)
    entropy=-(probabilities*log_probabilities).sum(1)
    count=observed.sum(1);multi=count>1
    normalized=entropy[multi]/np.log(count[multi])
    best=np.where(observed,rewards,-np.inf).max(1,keepdims=True)
    winners=observed&(rewards>=best-.02)
    return dict(prediction_entropy_nats=float(entropy.mean()),
        prediction_entropy_std_nats=float(entropy.std()),
        prediction_entropy_normalized=float(normalized.mean()) if multi.any() else 0.,
        prediction_entropy_normalized_std=float(normalized.std()) if multi.any() else 0.,
        entropy_choice_cases=int(multi.sum()),
        top_plan_probability=float(probabilities.max(1).mean()),
        near_best_target_entropy_nats=float(np.log(winners.sum(1)).mean()),
        near_best_probability_mass=float((probabilities*winners).sum(1).mean()))


@torch.inference_mode()
def weight_diagnostics(model):
    """Absolute-magnitude entropy is scale invariant; L2 reports weight scale."""
    squared=0.;count=0;weighted_entropy=0.;matrix_count=0;raw_statistic=0.;result={}
    for name,parameter in model.named_parameters():
        values=parameter.detach().double().flatten()
        squared+=float(values.square().sum());count+=values.numel()
        if parameter.ndim<2:continue
        magnitudes=values.abs();total=float(magnitudes.sum())
        positive_magnitudes=magnitudes[magnitudes>0]
        raw=float(-(positive_magnitudes*positive_magnitudes.log()).sum())
        result[f'absolute_weight_statistic_raw/{name}']=raw;raw_statistic+=raw
        if total>0:
            probabilities=magnitudes/total;positive=probabilities[probabilities>0]
            entropy=float(-(positive*positive.log()).sum())
        else:entropy=0.
        normalized=entropy/np.log(values.numel()) if values.numel()>1 else 0.
        result[f'absolute_weight_entropy_normalized/{name}']=normalized
        weighted_entropy+=normalized*values.numel();matrix_count+=values.numel()
    result.update(parameter_l2=float(np.sqrt(squared)),parameter_rms=float(np.sqrt(squared/max(1,count))),
                  absolute_weight_entropy_normalized=weighted_entropy/max(1,matrix_count),
                  absolute_weight_statistic_raw=raw_statistic)
    return result

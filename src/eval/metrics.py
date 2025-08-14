from typing import List, Dict
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def bleu4(hypotheses: List[List[str]], references: List[List[List[str]]]) -> float:
    # hypotheses: list of [tokenized caption]; references: list of list of [tokenized ref]
    smoothie = SmoothingFunction().method3
    return corpus_bleu(references, hypotheses, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie) * 100.0

def cider(preds: Dict[int, str], gts: Dict[int, List[str]]) -> float:
    # Use pycocoevalcap CIDEr
    from pycocoevalcap.cider.cider import Cider
    # Convert to COCOEvalCap format
    gts_tok = {k: [v.strip() for v in vals] for k, vals in gts.items()}
    res_tok = {k: [v.strip()] for k, v in preds.items()}
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts_tok, res_tok)
    return score * 100.0

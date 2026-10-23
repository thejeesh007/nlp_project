"""
Shared temperature-scaling wrapper for the binary toxicity pipeline.

A fine-tuned transformer's raw softmax output tends to be overconfident --
clearly-toxic inputs often score 99.9%+ even though the model's actual
accuracy on the held-out test set is ~97%, not 99.9%. That gap (confidence
higher than true accuracy) is textbook overconfidence, not evidence of
overfitting by itself -- see src/calibrate_temperature.py, which measures
it directly via Expected Calibration Error and fixes it with temperature
scaling (Guo et al., ICML 2017): dividing the logits by a single learned
scalar T > 1 before softmax. This only rescales confidence; it does not
change which class is predicted (same logit ranking, same argmax), so
accuracy/F1 at a re-tuned threshold are essentially unchanged.

app.py and fairness_audit.py both import build_calibrated_pipeline() so the
displayed confidence numbers and the audit's measured score gaps reflect
the calibrated, not raw, probabilities -- and the fairness audit's
decision thresholds are tuned against the same calibrated scores.
"""

import json
import os

from transformers import TextClassificationPipeline


class TemperatureScaledPipeline(TextClassificationPipeline):
    def __init__(self, *args, temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, **kwargs):
        if self.temperature != 1.0:
            model_outputs = dict(model_outputs)
            model_outputs["logits"] = model_outputs["logits"] / self.temperature
        return super().postprocess(model_outputs, function_to_apply=function_to_apply, top_k=top_k, **kwargs)


def load_temperature(metrics_path: str) -> float:
    if not os.path.isfile(metrics_path):
        return 1.0
    with open(metrics_path) as f:
        metrics = json.load(f)
    return float(metrics.get("calibration", {}).get("temperature", 1.0))


def build_calibrated_pipeline(model, tokenizer, metrics_path: str, **pipeline_kwargs):
    temperature = load_temperature(metrics_path)
    return TemperatureScaledPipeline(
        model=model, tokenizer=tokenizer, temperature=temperature, **pipeline_kwargs
    )

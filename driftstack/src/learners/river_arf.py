
from river.ensemble import AdaptiveRandomForestClassifier
from river import metrics

class OnlineARF:
    def __init__(self, **params):
        self.model = AdaptiveRandomForestClassifier(**params)
        self.metric = metrics.Accuracy()

    def learn_predict(self, x_dict, y_true):
        # x_dict: {f0: v0, f1: v1, ...}
        y_pred = self.model.predict_one(x_dict)
        self.metric = self.metric.update(y_true, y_pred)
        self.model = self.model.learn_one(x_dict, y_true)
        return y_pred

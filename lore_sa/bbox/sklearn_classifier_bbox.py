from lore_sa.bbox.bbox import AbstractBBox

__all__ = ["AbstractBBox","sklearnBBox"]
class sklearnBBox(AbstractBBox):
    def __init__(self, classifier,map = None, transformer = None, custom_scaler=None):
        self.bbox = classifier
        if transformer:
            self.transformer = transformer
        if custom_scaler:
            self.custom_scaler = custom_scaler
        if map is not None:
            self.map = map

    def predict(self, X):
        data = self.custom_scaler.transform(X) if hasattr(self, 'custom_scaler') else X
        data = self.transformer.transform(data) if hasattr(self, 'transformer') else data
        predictions = self.bbox.predict(data)
        return self.map[predictions] if hasattr(self, 'map') else predictions

    def predict_proba(self, X):
        data = self.custom_scaler.transform(X) if hasattr(self, 'custom_scaler') else X
        data = self.transformer.transform(data) if hasattr(self, 'transformer') else data
        return self.bbox.predict_proba(data)
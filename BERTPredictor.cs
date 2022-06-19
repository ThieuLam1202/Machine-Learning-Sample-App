using Microsoft.ML;

namespace SampleApp
{
    internal class BERTPredictor
    {
        private MLContext _mlContext;
        private PredictionEngine<BERTInput, BERTOutput> _predictionEngine;

        public BERTPredictor(ITransformer trainedModel)
        {
            _mlContext = new MLContext();
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<BERTInput, BERTOutput>(trainedModel);
        }

        public BERTOutput Predict(BERTInput encodedInput)
        {
            return _predictionEngine.Predict(encodedInput);
        }
    }
}

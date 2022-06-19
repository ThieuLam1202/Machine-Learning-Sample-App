using BERTTokenizers;

namespace SampleApp
{
    public class BERT
    {
        private BERTPredictor _predictor;
        private BertUncasedBaseTokenizer _tokenizer;

        enum Label { LABEL_1, LABEL_3, LABEL_0 };

        public BERT(string bertModelPath)
        {
            _tokenizer = new BertUncasedBaseTokenizer();

            var trainer = new BERTTrainer();
            var trainedModel = trainer.Trainer(bertModelPath, false);
            _predictor = new BERTPredictor(trainedModel);
        }

        private static double[] SoftMax(double[] hoSums)
        {
            double max = hoSums[0];
            for (int i = 0; i < hoSums.Length; ++i)
                if (hoSums[i] > max) max = hoSums[i];
            double scale = 0.0;
            for (int i = 0; i < hoSums.Length; ++i)
                scale += Math.Exp(hoSums[i] - max);
            double[] result = new double[hoSums.Length];
            for (int i = 0; i < hoSums.Length; ++i)
                result[i] = Math.Exp(hoSums[i] - max) / scale;
            return result;
        }

        public void Predict(string text)
        {
            var tokenizer = new BertUncasedBaseTokenizer();

            var encoded = tokenizer.Encode(512, text);

            var input = new BERTInput()
            {
                InputIds = encoded.Select(t => t.InputIds).ToArray(),
                AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
                TokenTypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
            };

            var output = _predictor.Predict(input);

            var modifiedOutput = Array.ConvertAll(output.Output0, x => (double)x);
            var percentage = SoftMax(modifiedOutput);
            float maxValue = output.Output0.Max();
            int maxIndex = output.Output0.ToList().IndexOf(maxValue);
            Console.WriteLine("Label: " + Enum.GetName(typeof(Label), maxIndex) + ", percentage: " + percentage.Max());
        }
    }
}

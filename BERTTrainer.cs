using Microsoft.ML;

namespace SampleApp
{
    internal class BERTTrainer
    {
        private readonly MLContext _mlContext;

        public BERTTrainer()
        {
            _mlContext = new MLContext(11);
        }

        public ITransformer Trainer(string modelPath, bool gpu)
        {
            var pipeline = _mlContext.Transforms.ApplyOnnxModel
                (modelFile: modelPath,
                shapeDictionary: new Dictionary<string, int[]>
                {
                    {"input_ids", new[] {1, 512} },
                    {"attention_mask", new[] {1, 512} },
                    {"token_type_ids", new[] {1, 512} },
                    {"output_0", new[] {1, 3}},
                },
                inputColumnNames: new[] { "input_ids",
                                           "attention_mask",
                                           "token_type_ids"},
                outputColumnNames: new[] { "output_0" },
                gpuDeviceId: gpu ? 0 : (int?)null, fallbackToCpu: true);
            return pipeline.Fit(_mlContext.Data.LoadFromEnumerable(new List<BERTInput>()));
        }
    }
}

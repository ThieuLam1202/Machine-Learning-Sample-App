using Microsoft.ML.Data;

namespace SampleApp
{
    public class BERTOutput
    {
        [VectorType(1, 512)]
        [ColumnName("output_0")]
        public float[] Output0 { get; set; }
    }
}

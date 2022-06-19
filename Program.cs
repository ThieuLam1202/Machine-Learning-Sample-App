using SampleApp;

BERT model;
string modelPath = @"C:\Users\lam12\source\repos\SampleApp\Model\bert-base-uncased-snli.onnx";

model = new BERT(modelPath);

string input;

do
{
    Console.Write("Input the sentence pair. Input nothing to end: ");
    input = Console.ReadLine();
    if (input != "")
        model.Predict(input);
} while (input != "");
    
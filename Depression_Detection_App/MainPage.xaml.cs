using System.Threading.Tasks;
using Microsoft.VisualBasic;
using  Python.Runtime;
using System.IO;

namespace Depression_Detection_App;


public class AIManager
{

	dynamic depressionDetector;
	string model = "model_1";

	public AIManager()
	{
        Runtime.PythonDLL = @"C:\Users\dylan\anaconda3\python312.dll";
        PythonEngine.Initialize();
        using (Py.GIL())
        {
            string currentDirectory = Directory.GetCurrentDirectory();
            dynamic sys = Py.Import("sys");
			sys.path.append(currentDirectory);
			sys.path.append(@"Depression_Detection_App");

			dynamic aiManager = Py.Import("ai_manager");
			depressionDetector = aiManager.DepressionDetector();
        }
    }

	public double getDepressionPercentage(string text)
	{
		dynamic probability = depressionDetector.forward(model, text);
		double percentage = Convert.ToDouble(probability*100);
		percentage = Math.Round(percentage, 2);
		return percentage;
	}
}

public partial class MainPage : ContentPage
{
	AIManager DepressionAI;

	public MainPage()
	{
		InitializeComponent();
		DepressionAI = new AIManager();
	}

	private void InputDepression(object sender, EventArgs e)
	{
		double depressionPercentage = DepressionAI.getDepressionPercentage(DepressedTextHarvester.Text);
		TheFunny.HeightRequest += 1;
		DetectDepression.Text = $"Depression Percentage: {depressionPercentage}%";
		SemanticScreenReader.Announce(DetectDepression.Text);
	}
}


using System.Text;
using Keras;
using Keras.Models;
using Numpy;
using Sudoku.Shared;
using System.Diagnostics;


namespace NeuralNetworkSolverCSharp
{
	public class NeuralNetworkSolverCSharp:PythonSolverBase
	{
		
        private static string archiModelPath = GetFullPath(@"..\..\..\..\Sudoku.NeuralNetworkCSharp\Models\model_architecture.json");
        private static string weightsPath = GetFullPath(@"..\..\..\..\Sudoku.NeuralNetworkCSharp\Models\model.weights.h5");
		private static BaseModel model;

        public override SudokuGrid Solve(SudokuGrid s)
        {
            return NeuralNetHelper.SolveSudoku(s, model);
        }

		protected override void InitializePythonComponents()
		{
			//declare your pip packages here
			InstallPipModule("numpy");
			InstallPipModule("tensorflow");
			base.InitializePythonComponents();

			// Load the model
			model = NeuralNetHelper.LoadModel(archiModelPath, weightsPath);
		}

		private static string GetFullPath(string relativePath)
        {
            return Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, relativePath));
        }
    }
	public class NeuralNetHelper
	{

		static NeuralNetHelper()
		{
			Setup.UseTfKeras();
		}

		public static BaseModel LoadModel(string archpath, string weightsPath)
        {

            Debug.WriteLine(archpath);
            Debug.WriteLine(weightsPath);
            var json = File.ReadAllText(archpath);
            Debug.WriteLine(json);
            var loaded_model = Sequential.ModelFromJson(json);
            loaded_model.LoadWeight(weightsPath);
            return loaded_model;
        }

		public static NDarray GetFeatures(SudokuGrid objSudoku)
		{
			return Normalize(np.array(objSudoku.Cells).reshape(9, 9));
		}

		public static NDarray Normalize(NDarray features)
		{
			return (features / 9) - 0.5;
		}

		public static NDarray DeNormalize(NDarray features)
		{
			return (features + 0.5) * 9;
		}



		public static SudokuGrid SolveSudoku(SudokuGrid s, BaseModel model)
		{
			var features = GetFeatures(s);
			while (true)
			{
				var output = model.Predict(features.reshape(1, 9, 9, 1));
				output = output.squeeze();
                output = output.reshape(9, 9, 9);

                
				var prediction = np.argmax(output, axis: 2).reshape(9, 9) + 1;
				var proba = np.around(np.max(output, axis: new[] { 2 }).reshape(9, 9), 2);

				features = DeNormalize(features);
				var mask = features.@equals(0);
				if (((int)mask.sum()) == 0)
				{
					break;
				}

				var probNew = proba * mask;
				var ind = (int)np.argmax(probNew);
				var (x, y) = (ind / 9, ind % 9);
				var val = prediction[x][y];
				features[x][y] = val;
				features = Normalize(features);

			}

            string sudokuString = ConvertToSudokuString(features);
            Console.WriteLine("Sudoku string: " + sudokuString);

            return SudokuGrid.ReadSudoku(sudokuString);
        }

        private static string ConvertToSudokuString(NDarray features)
        {
            StringBuilder sb = new StringBuilder();
            double[] flatArray = features.GetData<double>();
            double[,] array = new double[9, 9];
            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    array[i, j] = flatArray[i * 9 + j];
                }
            }

            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    int value = (int)array[i, j];
                    sb.Append(value == 0 ? '0' : value.ToString()[0]);
                }
            }

            return sb.ToString();
        }
	}
}

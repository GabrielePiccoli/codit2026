# Import delle librerie necessarie
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, confusion_matrix
import scipy.stats as stats
import os
import json

###################################################################################

class MetricCalculator:
    """Classe per il calcolo delle metriche di validazione e superamenti di soglia."""
    def __init__(self, Y_true_ND, Y_pred_ND, Y_true, Y_pred):
        self.Y_true_ND = Y_true_ND
        self.Y_pred_ND = Y_pred_ND
        self.Y_true = Y_true
        self.Y_pred = Y_pred

    def calculate_metrics(self):
        mae = mean_absolute_error(self.Y_true_ND, self.Y_pred_ND)
        me = np.mean(self.Y_true_ND - self.Y_pred_ND)
        nme = me / np.mean(self.Y_true_ND)
        nmae = mae / np.mean(self.Y_true_ND)
        correlation =  np.corrcoef(self.Y_true_ND, self.Y_pred_ND)[0, 1]
        media_vera = np.mean(self.Y_true_ND)
        media_prevista = np.mean(self.Y_pred_ND)

        return [mae, nmae, me, nme, correlation, media_vera, media_prevista]

    def calculate_exceedances(self, threshold):
        A = np.sum((self.Y_true >= threshold) & (self.Y_pred >= threshold))
        B = np.sum((self.Y_true >= threshold) & (self.Y_pred < threshold))
        C = np.sum((self.Y_true < threshold) & (self.Y_pred >= threshold))
        
        HR = A / (A + B) if (A + B) > 0 else 0
        FAR = C / (A + C) if (A + C) > 0 else 0
        CSI = A / (A + C + B) if (A + C + B) > 0 else 0

        return [A, B, C, HR, FAR, CSI]

    @staticmethod
    def matrice_contingenza(Y_true, Y_pred, threshold, image_path):
        Y_true_class = (Y_true >= threshold).astype(int)
        Y_pred_class = (Y_pred >= threshold).astype(int)
        cm = confusion_matrix(Y_true_class, Y_pred_class)
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["< Threshold", "> Threshold"], 
                    yticklabels=["< Threshold", "> Threshold"])
        plt.xlabel("Prediction")
        plt.ylabel("Observed")
        #plt.title(f"Contingency matrix (Threshold {threshold} μg/m³)")
        plt.savefig(os.path.join(image_path, f"MatriceContingenza_{threshold}.png"), dpi=300)
        plt.close()

###################################################################################

class DataImporter:
    """Classe per la gestione dell'importazione dei dataset."""
    @staticmethod
    def import_dataset(year):
        nameCval_ND = f'C_val_{year}_norm.xlsx'
        C_val_ND = pd.read_excel(os.path.join('01-ESPORTAZIONI/04-DATA_NORM_INDIPENDENTE', nameCval_ND), header=None)
        C_val_ND = np.array(C_val_ND)

        nameEval = f'E_val_{year}_norm.pkl'
        E_val_ND = joblib.load(os.path.join('01-ESPORTAZIONI/04-DATA_NORM_INDIPENDENTE', nameEval))

        nameCval = f'C_val_{year}.xlsx'
        C_val = pd.read_excel(os.path.join('00-DATI', nameCval), header=None).iloc[:, 1].values

        return C_val_ND, E_val_ND, C_val

###################################################################################

class ModelEvaluator:
    """Classe principale per la gestione della valutazione del modello."""
    def __init__(self, model_name, hp_name, validation_years, thresholds, val_path):
        self.model_name = model_name
        self.hp_name = hp_name
        self.validation_years = validation_years
        self.thresholds = thresholds
        self.val_path = val_path
        self.metrics_labels = ['MAE', 'NMAE', 'ME', 'NME', 'CORR', 'M_TRUE', 'M_PRED']
        self.exceedances_labels = ['A', 'B', 'C', 'HR', 'FAR', 'CSI']
        self.metrics_data = {}
        self.exceedances_data = {}
        self.model = keras.models.load_model(self.model_name, compile=False)

    def evaluate(self):
        for valYear in self.validation_years:
            print(f"Validazione {valYear}")
            valYearPath = os.path.join(self.val_path, valYear)
            imagePath = os.path.join(valYearPath, '00-IMMAGINI')
            os.makedirs(imagePath, exist_ok=True)
            
            C_val_ND, E_val_ND, C_val = DataImporter.import_dataset(valYear)
            Y_true_ND = C_val_ND[1:]
            Y_pred_ND = self.model.predict([C_val_ND[:-1], E_val_ND[:-1]])

            Y_true = C_val[1:]

            Y_true = C_val[1:]
            max = np.max(C_val)
            min = np.min(C_val)
            
            Y_pred = np.zeros(Y_pred_ND.shape)
            for idx, data in enumerate(Y_pred_ND):
                Y_pred[idx] = (max-min)*data + min

            calculator = MetricCalculator(Y_true.flatten(), Y_pred.flatten(), Y_true.flatten(), Y_pred.flatten())
            self.metrics_data[valYear] = calculator.calculate_metrics()

            self.print_metrics(valYear, self.metrics_data[valYear])
            
            for threshold in self.thresholds:
                self.exceedances_data[(valYear, threshold)] = calculator.calculate_exceedances(threshold)
                MetricCalculator.matrice_contingenza(Y_true, Y_pred, threshold, imagePath)

                self.print_exceedances(self.exceedances_data[(valYear, threshold)], threshold)

            self.save_predictions(Y_pred_ND.flatten(), Y_pred.flatten(), Y_true.flatten(), valYear, valYearPath, imagePath)

        self.save_results()
        
    def print_metrics(self, valYear, metrics):
        print(f"\n=== Metriche di validazione {valYear} ===")
        for index, row in enumerate(self.metrics_labels):
            print(f"{row} = {metrics[index]}")

    def print_exceedances(self, exceedances, threshold):
        print(f"\n=== Soglia {threshold} μg/m³ ===")
        for index, row in enumerate(self.exceedances_labels):
            print(f"{row} = {exceedances[index]}")

    def save_predictions(self, Y_pred_ND, Y_pred, Y_true, valYear, valYearPath, imagePath):
        # Salvataggio delle previsioni
        csv_filename = f"Y_pred_{valYear}.csv"
        csvPath = os.path.join(valYearPath, csv_filename)
        pd.DataFrame([Y_pred, Y_pred_ND]).to_csv(csvPath, header=False, index=False)

        name_image = f"Confronto_{valYear}.png"

        #Y_true_ND = C_val_ND[1:]
        # Plot dei risultati
        plt.figure(figsize=(14, 10))
        plt.plot(Y_true, label='Observed value', color='blue')
        plt.plot(Y_pred, label='Prediction', color='red')

        # Linee di soglia
        for threshold in self.thresholds:
            plt.axhline(y=threshold, linestyle="--", label=f"Threshold {threshold} μg/m³")

        #plt.title('Observed values Vs Predicted values', fontsize=14)
        plt.xlabel('Day', fontsize=14)
        plt.ylabel('NO₂ (μg/m³)', fontsize=14)

        # Imposta font size anche per i tick sugli assi
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.legend(fontsize=12)  # puoi aumentare anche qui se vuoi
        plt.savefig(os.path.join(imagePath, name_image), dpi=300)
        plt.close()

        print(f"Files salvati: {valYearPath}")         

    def save_results(self):
        df_exceedances = pd.DataFrame(
            index=self.exceedances_labels,
            columns=pd.MultiIndex.from_product([self.validation_years, self.thresholds], names=["Year", "Threshold"])
        )

        for (valYear, threshold), values in self.exceedances_data.items():
            df_exceedances.loc[:, (valYear, threshold)] = values

        # Carica gli iperparametri dal file JSON
        with open(self.hp_name, 'r') as f:
            best_hps = json.load(f)

        output_path = os.path.join(self.val_path, "IndiciPrestazione.xlsx")

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_metrics = pd.DataFrame(self.metrics_data, index=self.metrics_labels)
            df_metrics.to_excel(writer, sheet_name="Metrics")
            df_exceedances.to_excel(writer, sheet_name="Exceedances")

            df_best_hps = pd.DataFrame(list(best_hps.items()), columns=["Hyperparameter", "Value"])
            df_best_hps.to_excel(writer, sheet_name="HyperParameter")

        print(f"File Excel salvato: {output_path}")

###################################################################################

if __name__ == "__main__":
    path_names = "02-MODELLI/00-NORM_INDIPENDENTE/03-NORMALIZZATI_DATI"
    names = ['HCNN_NormoDati_20Trials', 'HCNN_NormoDati_40Trials', 'HCNN_NormoDati_60Trials', 'HCNN_NormoDati_100Trials']

    for name in names:
        model_name =  os.path.join(path_names, name + '.h5')
        val_path = os.path.join('01-ESPORTAZIONI/02-VALIDAZIONE/01-NORM_INDIPENDENTE/02-DENORM_METRICHE_ND_NDR', name)
        hp_name = os.path.join(path_names,'bestHP_' + name)
        validation_years = ['2020_2022', '2020', '2021', '2022']
        thresholds = [25, 50]
        
        evaluator = ModelEvaluator(model_name, hp_name, validation_years, thresholds, val_path)
        evaluator.evaluate()

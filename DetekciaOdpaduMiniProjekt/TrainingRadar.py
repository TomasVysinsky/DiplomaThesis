import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import numpy as np
from typing import List, Tuple
from CustomPrinter import CustomTextObj, CustomPrinter
from WindowedLeBasedDataset import WindowedLeBasedDataset
from IPython.display import display, HTML


class TrainingMetrics:
    def __init__(self, precision, recall, f1):
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        
        self.confusion_matrix_data_frame = None

    def print(self, should_print, text_to_append : str):
        CustomPrinter.custom_print(
            f"Precision {self.precision:.3f}  "
            f"Recall {self.recall:.3f} "
            f"F1 {self.f1:.3f}  ",
            should_print,
            text_to_append
        )
        print(self.confusion_matrix_data_frame)
        


class TrainingRadar:
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        epochs: int = 1000,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        batch_train: int = 512,
        batch_test: int = 256,
        device: str = None,
        seed: int = 42,
        class_weight_b1_multiplier: float = 2.0,
        class_weight_b2_multiplier: float = 3.0,
    ):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.wd = weight_decay
        self.batch_train = batch_train
        self.batch_test = batch_test
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.model_name: str = model_name
        self.class_weight_b1_multiplier = class_weight_b1_multiplier
        self.class_weight_b2_multiplier = class_weight_b2_multiplier

    def initialize_and_wrap_datasets(
        self,
        train_dataset: WindowedLeBasedDataset,
        test_dataset: WindowedLeBasedDataset
    ) -> Tuple[DataLoader, DataLoader, int, WindowedLeBasedDataset]:

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_train, shuffle=True
        )

        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_test, shuffle=False
        )

        train_len = len(train_dataset.samples)

        return train_loader, test_loader, train_len, train_dataset

    def train(
        self,
        train_dataset: WindowedLeBasedDataset,
        test_dataset: WindowedLeBasedDataset,
        num_classes: int,
        train: bool = True,
        patience: int = 10,
        labels : List[str] = [ 'L', 'R', 'B' ]
    ):
        """
        num_classes – počet tried (výstupný rozmer modelu = num_classes)
        y v datasetoch sú one-hot vektory dĺžky num_classes, napr. [1,0,0,0]
        """
        torch.manual_seed(self.seed)
        save_model_file_name = f"{self.model_name}.pt"
        save_text_file_name = f"{self.model_name}.txt"
        should_print = True

        if train:
            result_text = CustomTextObj()

            train_loader, test_loader, train_len, train_ds = self.initialize_and_wrap_datasets(
                train_dataset, test_dataset
            )

            # ------ init model / loss / optim --
            self.model.to(self.device)

            # pos_weight z one-hot ground truth v train sete (tvar (N, C))
            y_train = torch.stack([y for _, y in train_ds]).float()  # (N, C)
            # počet negatívnych / pozitívnych príkladov pre každú triedu
            count_total = len(train_ds)
            if False:
                
                w0 = sum(y_train[:,0]).item() / count_total
                w1 = sum(y_train[:,1]).item() / count_total
                w2 = sum(y_train[:,2]).item() / count_total
                if num_classes == 4:
                    w3 = sum(y_train[:,3]).item() / count_total
                    class_weights = torch.tensor([w0, w1, w2, w3], dtype=torch.float32).to(self.device)
                else:
                    class_weights = torch.tensor([w0, w1, w2], dtype=torch.float32).to(self.device)
                
            counts = y_train.sum(dim=0)
            class_weights = count_total / counts
            # Zvýšiť váhy pre problematické triedy B1 a B2
            class_weights[2] = class_weights[2] * self.class_weight_b1_multiplier
            class_weights[3] = class_weights[3] * self.class_weight_b2_multiplier
            class_weights = class_weights / class_weights.sum()  # normovanie
            print(f"Class weights: L={class_weights[0]:.4f}, R={class_weights[1]:.4f}, B1={class_weights[2]:.4f}, B2={class_weights[3]:.4f}")
            class_weights = class_weights.to(self.device)

            criterion = nn.CrossEntropyLoss(weight=class_weights)

            optimiser = optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.wd
            )

            # early stopping
            best_f1 = 0.0
            patience_counter = 0
            best_model_state = None

            history = []
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                running = 0.0

                for xb, yb in train_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device).float()  # one-hot vektory

                    optimiser.zero_grad()
                    logits = self.model(xb)  # (batch, num_classes)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimiser.step()

                    running += loss.item() * xb.size(0)

                train_loss = running / train_len

                metrics_test = self.evaluate(
                    test_loader, num_classes, 'test', should_print, result_text, labels
                )
                metrics_train = self.evaluate(
                    train_loader, num_classes, 'train', should_print, result_text, labels
                )

                CustomPrinter.custom_print(f"Epoch: {epoch:02d}", should_print, result_text)
                CustomPrinter.custom_print(f"Train loss: {train_loss:.6f}", should_print, result_text)
                #CustomPrinter.custom_print("Test metrics:", should_print, result_text)
                #metrics_test.print(should_print, result_text)
                #CustomPrinter.custom_print("Train metrics:", should_print, result_text)
                #metrics_train.print(should_print, result_text)
                #CustomPrinter.custom_print("-----------", should_print, result_text)

                # Early stopping podľa test F1
                if metrics_test.f1 > best_f1:
                    best_f1 = metrics_test.f1
                    best_model_state = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    CustomPrinter.custom_print(
                        f"⏳  Early stop counter: {patience_counter} / {patience}",
                        should_print,
                        result_text
                    )
                    if patience_counter >= patience:
                        
                        
                        CustomPrinter.custom_print(
                            "🛑  Early stopping triggered.",
                            should_print,
                            result_text
                        )
                        
                        
                        break

                history.append({
                    "epoch": epoch,
                    "loss": train_loss,
                    "metrics": metrics_test
                })

            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

            torch.save(self.model.state_dict(), save_model_file_name)

            with open(save_text_file_name, "w", encoding="utf-8") as f:
                f.write(str(result_text.text))

            CustomPrinter.custom_print(
                f"Tréning dokončený, uložené do {save_model_file_name}",
                should_print,
                result_text
            )
            
            print("Test metrics:")
            metrics_test.print(True, result_text)
            print("Train metrics:")
            metrics_train.print(True, result_text)

            return {
                "model": self.model,
                "history": history,
                "test_metrics": history[-1] if history else None
            }
        else:
            self.model.load_state_dict(
                torch.load(save_model_file_name, weights_only=True)
            )
            self.model.to(self.device)
            return {"model": self.model}

    # --------------------------- pomocné ------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        num_classes: int,
        desc: str,
        should_print: bool,
        text_to_append: CustomTextObj,
        labels : List[str]
    ) -> TrainingMetrics:
        """
        Vyhodnotenie pre multi-class single-label klasifikáciu.
        Model vracia logits tvaru (batch, num_classes).
        Labely sú one-hot vektory dĺžky num_classes.
        """
        self.model.eval()
        y_true_oh_list = []   # one-hot ground truth
        y_prob_list = []      # predikované pravdepodobnosti

        for xb, yb in loader:
            xb = xb.to(self.device)
            logits = self.model(xb)  # (batch, num_classes)
            probs = torch.softmax(logits, 1).cpu().numpy()  # (batch, num_classes)

            y_prob_list.append(probs)
            y_true_oh_list.append(yb.numpy())

        if not y_true_oh_list:
            return TrainingMetrics(
                precision=0.0, recall=0.0, f1=0.0
            )

        y_true_oh = np.concatenate(y_true_oh_list, axis=0)  # (N, C)
        y_prob = np.concatenate(y_prob_list, axis=0)        # (N, C)

        # index triedy z one-hot ground truth
        y_true = np.argmax(y_true_oh, axis=1)               # (N,)
        # predikovaná trieda: argmax z pravdepodobností
        y_pred = np.argmax(y_prob, axis=1)                  # (N,)

        #accuracy = accuracy_score(y_true, y_pred)
        #f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        precision = precision_score(
            y_true,
            y_pred,
            average='macro',
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average='macro',
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average='macro',
            zero_division=0
        )
        
        # Konfúzna matica
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        #CustomPrinter.custom_print(f"{desc} Confusion matrix:", should_print, text_to_append)
        #CustomPrinter.custom_print(str(cm), should_print, text_to_append)
        
        confusion_matrix_df = self.make_confusion_df(cm, labels)
        

        training_metrics = TrainingMetrics(precision, recall, f1)
        training_metrics.confusion_matrix_data_frame = confusion_matrix_df
        return training_metrics


    def make_confusion_df(self, cm: np.ndarray, class_names=None):

        if class_names is None:
            # ak neboli zadané názvy tried, spravíme z indexov stringy
            n_classes = cm.shape[0]
            class_names = [str(i) for i in range(n_classes)]

        df = pd.DataFrame(
            cm.T,
            index=pd.Index(class_names, name="Predikovaná trieda"),
            columns=pd.Index(class_names, name="Skutočná trieda")
        )
        return df
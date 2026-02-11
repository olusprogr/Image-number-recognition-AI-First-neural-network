from fileinput import filename
import numpy as np
import datasetloader as dl
import threading
import activation as act

class NeuralNetwork():
    """
    Ein vollständig verbundenes neuronales Netzwerk (Feedforward Neural Network) 
    für Multi-Klassen-Klassifikation.
    
    Das Netzwerk unterstützt:
    - Beliebig viele Hidden Layers mit konfigurierbarer Größe
    - ReLU-Aktivierung für Hidden Layers
    - Softmax-Aktivierung für Output Layer
    - Stochastic Gradient Descent (SGD) mit Mini-Batches
    - Dynamische Learning Rate mit exponentiellem Decay
    - He-Initialisierung für optimale Gewichtsverteilung
    
    Architektur:
        Input Layer → Hidden Layers (ReLU) → Output Layer (Softmax)
    """
    
    def __init__(self, num_input: int, hidden_layers: dict[str, int], num_output: int):
        """
        Initialisiert die Netzwerk-Architektur.
        
        Args:
            num_input (int): Anzahl der Input-Features (z.B. 784 für 28x28 Bilder)
            hidden_layers (dict[str, int]): Dictionary mit Layer-Namen als Keys und 
                                           Neuronenzahl als Values
                                           Beispiel: {'hidden1': 256, 'hidden2': 128}
            num_output (int): Anzahl der Output-Klassen (z.B. 62 für EMNIST byclass)
        
        Attributes:
            weights (list): Liste von Gewichtsmatrizen für jeden Layer
            biases (list): Liste von Bias-Vektoren für jeden Layer
            layer_names (list): Namen aller Layer inklusive Output-Layer
            layer_sizes (list): Neuronenzahl pro Layer
        """
        self.num_input = num_input
        self.hidden_layers = hidden_layers
        self.num_outputs = num_output
        self.weights = []  # Wird in initialize_adaptive_weights_and_biases() gefüllt
        self.biases = []   # Wird in initialize_adaptive_weights_and_biases() gefüllt
        self.forward: any = None  # Placeholder für zukünftige Funktionalität
        self.dynamic_learning_rate: bool = False

        # Extrahiere Layer-Informationen aus dem hidden_layers Dictionary
        # Beispiel: ['hidden1', 'hidden2', 'output']
        self.layer_names = list(self.hidden_layers.keys()) + ['output']
        # Beispiel: [256, 128, 62]
        self.layer_sizes = list(self.hidden_layers.values()) + [self.num_outputs]

        print(self.layer_names)
        print(self.layer_sizes)

    def get_network_configuration(self):
        """
        Gibt die aktuelle Netzwerk-Konfiguration zurück.
        
        Returns:
            dict: Dictionary mit num_input, hidden_layers und num_output
        
        Verwendung:
            Nützlich für Logging, Debugging oder zum Speichern der Architektur
        """
        return {
            "num_input": self.num_input,
            "hidden_layers": self.hidden_layers,
            "num_output": self.num_outputs
        }

    def initialize_adaptive_weights_and_biases(self):
        """
        Initialisiert Gewichte und Biases für alle Layer mit He-Initialisierung.
        
        He-Initialisierung (Kaiming Initialization):
            - Optimiert für ReLU-Aktivierungsfunktionen
            - Gewichte: W ~ N(0, sqrt(2/n_in)) 
              wobei n_in = Anzahl Input-Neuronen zum aktuellen Layer
            - Verhindert Vanishing/Exploding Gradients
        
        Biases:
            - Werden mit Nullen initialisiert (Standard-Praxis)
        
        Gewichts-Dimensionen:
            - weights[i]: (neurons_current_layer, neurons_previous_layer)
            - Beispiel Layer 1: (256, 784) für Input → Hidden1
            - Beispiel Layer 2: (128, 256) für Hidden1 → Hidden2
        
        Bias-Dimensionen:
            - biases[i]: (neurons_current_layer,)
            - Beispiel: (256,) für Hidden1
        """
        input_size = self.num_input

        for layer_name, num_neurons in zip(self.layer_names, self.layer_sizes):
            print(f"Creating layer: {layer_name} with {num_neurons} neurons... Input size: {input_size}")

            # He-Initialisierung: W ~ N(0, sqrt(2/n_in))
            # np.random.randn erzeugt Normalverteilung mit mean=0, std=1
            # Multiplikation mit sqrt(2/input_size) skaliert die Standardabweichung
            self.weights.append(
                np.random.randn(num_neurons, input_size) * np.sqrt(2 / input_size)
            )

            # Biases mit Nullen initialisieren
            # Shape: (num_neurons,) - ein Bias pro Neuron
            self.biases.append(np.zeros(num_neurons))

            # Für nächsten Layer: Output des aktuellen Layers wird Input
            input_size = num_neurons

    def implement_forward(self, x):
        """
        Führt einen Forward Pass durch das gesamte Netzwerk aus.
        
        Unterstützt sowohl einzelne Samples als auch Batches:
            - Einzelnes Sample: x.shape = (784,)
            - Batch: x.shape = (batch_size, 784)
        
        Args:
            x (np.ndarray): Input-Daten
                           Shape: (features,) oder (batch_size, features)
        
        Returns:
            list: Liste von Aktivierungen für jeden Layer
                  - activations[0]: Output von Layer 1 (nach ReLU)
                  - activations[-1]: Output von Output Layer (nach Softmax)
                  
        Mathematik pro Layer:
            1. Linear Transformation: z = W·x + b
               - Batch: z = x·W^T + b
               - Single: z = W·x + b
            2. Activation Function:
               - Hidden Layers: a = ReLU(z) = max(0, z)
               - Output Layer: a = Softmax(z)
        
        Beispiel-Shapes (Batch-Modus mit batch_size=32):
            Input x: (32, 784)
            Layer 1: z = (32, 784) @ (784, 256)^T + (256,) → (32, 256)
                     a = ReLU(z) → (32, 256)
            Layer 2: z = (32, 256) @ (256, 128)^T + (128,) → (32, 128)
                     a = ReLU(z) → (32, 128)
            Output:  z = (32, 128) @ (128, 62)^T + (62,) → (32, 62)
                     a = Softmax(z) → (32, 62)
        """
        activations = []
        is_batch = x.ndim == 2  # True wenn Input 2D ist (Batch)
        
        for w, b, name in zip(self.weights, self.biases, self.layer_names):
            # Linear Transformation z = W·x + b
            if is_batch:
                # Batch-Verarbeitung: (batch, input) @ (input, neurons)
                # w.T transponiert die Gewichtsmatrix für korrekte Matrixmultiplikation
                z = np.dot(x, w.T) + b  
            else:
                # Einzelnes Sample: (neurons, input) @ (input,) → (neurons,)
                z = np.dot(w, x) + b
            
            # Aktivierungsfunktion anwenden
            if name != 'output':
                # Hidden Layers: ReLU-Aktivierung
                # ReLU(z) = max(0, z) - setzt negative Werte auf 0
                x = act.Operations.relu(z)
            else:
                # Output Layer: Softmax-Aktivierung
                # Softmax wandelt Logits in Wahrscheinlichkeitsverteilung um
                # Sum(Softmax(z)) = 1.0
                x = act.Operations.softmax(z)
            
            # Speichere Aktivierung für diesen Layer (für Backpropagation benötigt)
            activations.append(x)
        
        return activations
    
    def train(self, train_inputs, train_labels, learning_rate=0.001, epochs=20, 
              dynamic_learning_rate=False, decay_epochs=4, batch_size=32):
        """
        Öffentliche Methode zum Starten des Trainings.
        
        Args:
            train_inputs (np.ndarray): Training-Daten
                                      Shape: (num_samples, num_features)
                                      Beispiel: (697932, 784) für EMNIST
            train_labels (np.ndarray): Training-Labels (Integer-Klassen)
                                      Shape: (num_samples,)
                                      Beispiel: [5, 23, 41, ...] für Klassen 0-61
            learning_rate (float): Initiale Learning Rate (Schrittgröße)
                                  Standard: 0.001
                                  Empfohlen: 0.001-0.01
            epochs (int): Anzahl der Trainings-Epochen
                         Eine Epoch = einmal durch alle Trainings-Daten
                         Standard: 20
            dynamic_learning_rate (bool): Aktiviert Learning Rate Decay
                                         True → LR wird reduziert während Training
                                         False → LR bleibt konstant
            decay_epochs (int): Nach wie vielen Epochen die LR reduziert wird
                               Bei decay_epochs=4: Reduktion bei Epoch 4, 8, 12, ...
                               Standard: 4
            batch_size (int): Anzahl Samples pro Mini-Batch
                             Größere Batches → stabileres Training, schneller
                             Kleinere Batches → häufigere Updates, bessere Konvergenz
                             Standard: 32, Empfohlen: 64-256
        
        Speichert die Parameter als Instanz-Variablen und startet learn().
        """
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dynamic_learning_rate = dynamic_learning_rate
        self.decay_epochs = decay_epochs
        self.batch_size = batch_size

        # Starte den eigentlichen Lernprozess
        self.learn(learning_rate, epochs, dynamic_learning_rate, decay_epochs)

    def learn(self, learning_rate, epochs, dynamic_learning_rate, decay_epochs):
        """
        Haupttrainings-Loop - implementiert Stochastic Gradient Descent (SGD).
        
        Trainings-Ablauf pro Epoch:
            1. Shuffle der Trainingsdaten (verhindert Bias)
            2. Mini-Batch Verarbeitung:
               a) Forward Pass → Vorhersagen berechnen
               b) Loss berechnen (Cross-Entropy)
               c) Backpropagation → Gradienten berechnen
               d) Gewichte updaten (Gradient Descent)
            3. Metriken ausgeben (Loss, Accuracy)
            4. Learning Rate anpassen (optional)
        
        Args:
            learning_rate (float): Initiale Learning Rate
            epochs (int): Anzahl Trainings-Epochen
            dynamic_learning_rate (bool): Aktiviert exponentiellen LR Decay
            decay_epochs (int): Frequenz der LR-Reduktion
        
        Learning Rate Decay:
            new_lr = initial_lr * (0.85 ^ (epoch // decay_epochs))
            Beispiel bei initial_lr=0.01, decay_epochs=4:
                Epoch 0-3: 0.01
                Epoch 4-7: 0.01 * 0.85 = 0.0085
                Epoch 8-11: 0.01 * 0.85^2 = 0.007225
                usw.
        
        Loss Function (Cross-Entropy):
            L = -1/N * Σ(y_true * log(y_pred))
            wobei:
                - N = batch_size
                - y_true = One-Hot encoded Labels
                - y_pred = Softmax Output
        
        Gradient Descent Update Rule:
            W_new = W_old - learning_rate * ∂L/∂W
            b_new = b_old - learning_rate * ∂L/∂b
        """
        initial_lr = learning_rate

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs} - Learning Rate: {learning_rate:.6f}")

            # Shuffle Trainings-Daten für bessere Generalisierung
            # Verhindert, dass das Model die Reihenfolge der Daten "lernt"
            indices = np.arange(len(self.train_inputs))
            np.random.shuffle(indices)
            train_inputs = self.train_inputs[indices]
            train_labels = self.train_labels[indices]

            # Learning Rate Decay anwenden
            if dynamic_learning_rate and epoch % decay_epochs == 0 and epoch != 0:
                # Exponentieller Decay: lr = initial_lr * (0.85^k)
                # wobei k = epoch // decay_epochs
                learning_rate = initial_lr * (0.85 ** (epoch // decay_epochs))

            # Metriken für diese Epoch
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            # Mini-Batch Training Loop
            # (Start, End, Step) → iteriert über Daten in Schritten von batch_size
            # ITERATION 1: start=0, ITERATION 2: start=128, ...
            for start in range(0, len(train_inputs), self.batch_size):
                end = start + self.batch_size

                # Extrahiere aktuellen Mini-Batch von Start bis End
                x = train_inputs[start:end]        # Shape: (batch_size, 784) - Trainings Daten (784 Pixel pro Bild)
                # Visualisierung:
                # x = [[0.2, 0.1, 0.8, ...],  ← Bild 1 (784 Pixel-Werte)
                #      [0.3, 0.5, 0.1, ...],  ← Bild 2 (784 Pixel-Werte)
                #      [0.1, 0.9, 0.2, ...],  ← Bild 3 (784 Pixel-Werte)
                #      ...                    
                #      [0.4, 0.2, 0.7, ...]]  ← Bild 128 (784 Pixel-Werte)

                y_true = train_labels[start:end]   # Shape: (batch_size,) - Richtige Klassen (0-61)
                batch_len = len(x)                 # Aktuelle Batch-Größe (kann am Ende kleiner sein)
                total_samples += batch_len         # Gesamtzahl der bisher verarbeiteten Samples (Bilder)

                # Progress-Ausgabe alle 50 Batches
                # ITERATION 1: start=0 → Ausgabe, ITERATION 50: start=6400 → Ausgabe, ...
                # Batch-Size wird immer um 128 erhöht, daher Ausgabe alle 6400 Samples
                if start % (self.batch_size * 50) == 0 and start != 0:
                    print(f"Epoch {epoch+1}/{epochs} - Step {start}/{len(train_inputs)} - "
                          f"Learning Rate: {learning_rate:.6f} - "
                          f"Loss: {total_loss/(start+batch_len):.4f} - "
                          f"Accuracy: {correct_predictions/total_samples:.4f}")

                # ===== FORWARD PASS =====
                # Berechne Vorhersagen für aktuellen Batch von Bildern x bzw 128 Bilder
                activations = self.implement_forward(x)
                output_a = activations[-1] # Liefert Liste mit 5 Elementen (4 Hidden Layers + 1 Output Layer)
                                           # Nehmen wir das letzte Element (Output Layer) → Shape: (128, 62) - Wahrscheinlichkeiten für 62 Klassen

                # ===== ACCURACY BERECHNUNG =====
                # Wähle Klasse mit höchster Wahrscheinlichkeit
                # Pro ZEILE den höchsten Index finden aus output_a, welches die vorhergesagte Klasse ist mit der höchsten Wahrscheinlichkeit
                # Beispiel: output_a[0] = [0.01, 0.02, ..., 0.85, ...] 
                #           → argmax findet Index 10 (höchster Wert 0.85)
                #           → predicted[0] = 10 (Klasse "A")
                predicted = np.argmax(output_a, axis=1)  # Shape: (batch_size,)
                # Zähle korrekte Vorhersagen
                # Ergebnis:
                # result = [True, True, True, False, True]  # Boolean-Array
                # Wie viele True-Werte gibt es?
                correct_predictions += np.sum(predicted == y_true)

                # ===== ONE-HOT ENCODING =====
                # Wandle Integer-Labels in One-Hot Vektoren um
                # Beispiel: Label 5 → [0,0,0,0,0,1,0,...,0]
                y_onehot = np.zeros((batch_len, self.num_outputs))  # (batch_size, 62)
                # Setze Einsen an den richtigen Positionen die aus dem Label Array kommen.
                # Sie geben die korrekte Klasse für jedes Bild an.
                y_onehot[np.arange(batch_len), y_true] = 1

                # ===== LOSS BERECHNUNG =====
                # Cross-Entropy Loss: -Σ(y_true * log(y_pred))
                # Epsilon (1e-9) verhindert log(0) = -inf
                # Liefert Liste mit allen Wahrscheinlichkeiten für die korrekten Klassen, z.B. [0.85, 0.02, 0.01, ...]
                # Wendet für jedes Element de Logarithmus an und multipliziert mit -1, um den Loss zu berechnen
                # Multipliziert mit y_onehot, um nur die Wahrscheinlichkeiten der korrekten Klassen (1) aus dem Label zu berücksichtigen
                # Rechts wird durch batch_len geteilt, um den Durchschnitts-Loss pro Sample zu erhalten
                loss = -np.sum(y_onehot * np.log(output_a + 1e-9)) / batch_len
                total_loss += loss

                # --------------------------------------

                # ===== BACKPROPAGATION =====
                # Berechne Gradienten für alle Layer (rückwärts durch Netzwerk)
                
                # Output Layer Error (Gradient der Loss-Funktion)
                # Für Softmax + Cross-Entropy vereinfacht sich zu: error = y_pred - y_true
                error = output_a - y_onehot   # Shape: (batch_size, 62)

                # Initialisiere Listen für Gradienten
                grad_weights = [None] * len(self.weights)
                grad_biases = [None] * len(self.biases)

                # Rückwärts durch alle Layer iterieren
                for layer_idx in reversed(range(len(self.weights))):
                    # Input zu diesem Layer (Output vom vorherigen Layer)
                    input_to_layer = x if layer_idx == 0 else activations[layer_idx - 1]
                    # Shape: (batch_size, neurons_prev)

                    # ===== GRADIENT BERECHNUNG =====
                    # Gradient bzgl. Gewichte: ∂L/∂W = error^T @ input
                    # error: (batch_size, neurons_current)
                    # input: (batch_size, neurons_prev)
                    # Ergebnis: (neurons_current, neurons_prev)
                    grad_weights[layer_idx] = error.T @ input_to_layer / batch_len
                    
                    # Gradient bzgl. Biases: ∂L/∂b = Σ(error) über alle Samples
                    # Summiere über Batch-Dimension → (neurons_current,)
                    grad_biases[layer_idx] = np.sum(error, axis=0) / batch_len

                    # ===== ERROR PROPAGATION ZUM VORHERIGEN LAYER =====
                    if layer_idx > 0:
                        # Propagiere Error rückwärts: error_prev = error @ W
                        # (batch_size, neurons_current) @ (neurons_current, neurons_prev)
                        # → (batch_size, neurons_prev)
                        error = error @ self.weights[layer_idx]
                        
                        # ReLU-Gradient: Multipliziere mit Ableitung von ReLU
                        # ReLU'(x) = 1 wenn x > 0, sonst 0
                        # (activations[layer_idx - 1] > 0) erstellt Boolean-Maske
                        # .astype(float) wandelt True→1.0, False→0.0
                        error = error * (activations[layer_idx - 1] > 0).astype(float)

                # ===== PARAMETER UPDATE (Gradient Descent) =====
                # Update-Regel: W_new = W_old - learning_rate * gradient
                for layer_idx in range(len(self.weights)):
                    self.weights[layer_idx] -= learning_rate * grad_weights[layer_idx]
                    self.biases[layer_idx] -= learning_rate * grad_biases[layer_idx]

            # ===== EPOCH METRIKEN =====
            # Durchschnittlicher Loss über alle Batches
            # HINWEIS: Diese Berechnung ist fehlerhaft (teilt zweimal)
            # Korrekt wäre: total_loss / len(train_inputs)
            avg_loss = total_loss / (len(train_inputs) / self.batch_size)
            
            # Finale Accuracy für diese Epoch
            accuracy = correct_predictions / total_samples

            print(f"Epoch {epoch+1} done - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

        # Speichere trainiertes Model nach allen Epochen
        self.save()

    def save(self, filename="trained_model.npz"):
        """
        Speichert die trainierten Gewichte und Biases in einer .npz Datei.
        
        Args:
            filename (str): Pfad zur Speicher-Datei
                           Standard: "trained_model.npz"
        
        Dateiformat:
            NumPy .npz (komprimiertes Archiv mit mehreren Arrays)
            
        Gespeicherte Arrays:
            - weight_0, weight_1, ..., weight_n: Gewichtsmatrizen aller Layer
            - bias_0, bias_1, ..., bias_n: Bias-Vektoren aller Layer
        
        Beispiel-Struktur:
            {
                'weight_0': array([[...]]),  # (256, 784)
                'bias_0': array([...]),       # (256,)
                'weight_1': array([[...]]),  # (128, 256)
                'bias_1': array([...]),       # (128,)
                ...
            }
        
        Verwendung:
            Ermöglicht späteres Laden des Models ohne erneutes Training
        """
        save_dict = {}
        
        # Speichere alle Gewichtsmatrizen
        for i, w in enumerate(self.weights):
            save_dict[f'weight_{i}'] = w
        
        # Speichere alle Bias-Vektoren
        for i, b in enumerate(self.biases):
            save_dict[f'bias_{i}'] = b

        # Speichere als komprimiertes NumPy-Archiv
        np.savez(filename, **save_dict)
    
    def predict(self, img: np.ndarray, top_k: int = 5) -> tuple:
        """
        Macht eine Vorhersage für ein einzelnes Bild.
        
        Args:
            img (np.ndarray): Input-Bild
                             Shape: (28, 28) oder (784,)
                             Werte: 0.0-1.0 (normalisiert)
            top_k (int): Anzahl der Top-Vorhersagen die ausgegeben werden
                        Standard: 5
        
        Returns:
            tuple: (predicted_class, confidence)
                   - predicted_class (int): Vorhergesagte Klasse (0-61)
                   - confidence (float): Wahrscheinlichkeit (0.0-1.0)
        
        Ablauf:
            1. Bild flattening (28x28 → 784)
            2. Forward Pass → Wahrscheinlichkeiten berechnen
            3. Sortiere Klassen nach Wahrscheinlichkeit
            4. Gebe Top-K Vorhersagen aus
            5. Returniere beste Vorhersage
        
        Ausgabe (Konsole):
            Image shape: (28, 28)
            Min / Max pixel values: 0.0 1.0
            Sample pixels (5x5 top-left):
            [[0.2 0.3 ...]]
            ...
            Top-5 predictions:
            Class 23 (N): 0.8521
            Class 12 (M): 0.0823
            ...
        """
        # Debug-Info: Zeige Bild-Eigenschaften
        print("Image shape:", img.shape)
        print("Min / Max pixel values:", img.min(), img.max())
        print("Sample pixels (5x5 top-left):\n", img[:5,:5])

        # Flatten Bild zu 1D-Vektor
        flat_img = img.flatten()  # (28, 28) → (784,)
        print("Flattened shape:", flat_img.shape)
        print("Flattened sample (first 10):", flat_img[:10])

        # Forward Pass durch Netzwerk
        # WICHTIG: Softmax wird bereits in implement_forward() angewendet!
        activations = self.implement_forward(flat_img)
        output = activations[-1]  # Shape: (62,)
                                 # Bereits Softmax-Wahrscheinlichkeiten!

        # Sortiere Klassen nach Wahrscheinlichkeit (absteigend)
        # np.argsort gibt Indizes in aufsteigender Reihenfolge
        # [-top_k:] nimmt die letzten k Elemente (höchste Werte)
        # [::-1] dreht die Reihenfolge um (höchste zuerst)
        top_indices = np.argsort(output)[-top_k:][::-1]

        # Erstelle Liste von (Klasse, Wahrscheinlichkeit) Tupeln
        results = []
        for idx in top_indices:
            results.append((idx, output[idx]))

        # Gebe Top-K Vorhersagen aus
        print("Top-{} predictions:".format(top_k))
        for idx, conf in results:
            # Wandle numerische Klasse in EMNIST-Zeichen um
            print(f"Class {idx} ({dl.DatasetLoader.number_to_emnist_class(idx)}): {conf:.4f}")

        # Returniere beste Vorhersage (höchste Wahrscheinlichkeit)
        predicted_class, confidence = results[0]
        return predicted_class, confidence

    def load(self, filename="trained_model.npz"):
        """
        Lädt trainierte Gewichte und Biases aus einer .npz Datei.
        
        Args:
            filename (str): Pfad zur gespeicherten Model-Datei
                           Standard: "trained_model.npz"
        
        Voraussetzungen:
            - Die Datei muss mit der save() Methode erstellt worden sein
            - Die Netzwerk-Architektur (layer_sizes) muss übereinstimmen
        
        Lädt:
            - weights: Liste von Gewichtsmatrizen
            - biases: Liste von Bias-Vektoren
        
        Verwendung:
            Ermöglicht Inference ohne erneutes Training
            
        Beispiel:
            nn_model = NeuralNetwork(784, {'h1': 256, 'h2': 128}, 62)
            nn_model.load('trained_model.npz')
            prediction = nn_model.predict(test_image)
        
        Fehlerbehandlung:
            Wirft FileNotFoundError wenn Datei nicht existiert
            Wirft KeyError wenn Datei nicht das erwartete Format hat
        """
        # Lade .npz Datei (Dictionary-artiger Zugriff auf Arrays)
        data = np.load(filename)

        # Rekonstruiere weights Liste
        # Lädt weight_0, weight_1, ..., weight_n
        self.weights = [data[f'weight_{i}'] for i in range(len(self.layer_sizes))]
        
        # Rekonstruiere biases Liste
        # Lädt bias_0, bias_1, ..., bias_n
        self.biases = [data[f'bias_{i}'] for i in range(len(self.layer_sizes))]

        print(f"Model loaded from {filename}")
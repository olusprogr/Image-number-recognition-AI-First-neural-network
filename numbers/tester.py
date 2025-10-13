import numpy as np
from PIL import Image

# === 1. Bild laden und vorbereiten ===
image_path = "Unbenannt.png"
img = Image.open(image_path).convert("L")      # Graustufen
img = img.resize((28, 28))                     # Größe anpassen
img = Image.eval(img, lambda x: 255 - x)       # Invertieren
img_array = np.array(img).astype(np.float32) / 255.0
img_flat = img_array.flatten()  # (784,)

# === 2. Modell laden ===
data = np.load("trained_model.npz")
print("Gespeicherte Arrays:", list(data.keys()))

# weights_input_hidden1 = data["weights_input_hidden1"]
# bias_hidden1 = data["bias_hidden1"]
# weights_hidden1_hidden2 = data["weights_hidden1_hidden2"]
# bias_hidden2 = data["bias_hidden2"]
# weights_hidden2_output = data["weights_hidden2_output"]
# bias_output = data["bias_output"]

weights_input_hidden1   = data["weight_0"]
bias_hidden1            = data["bias_0"]

weights_hidden1_hidden2 = data["weight_1"]
bias_hidden2            = data["bias_1"]

weights_hidden2_output  = data["weight_2"]
bias_output             = data["bias_2"]



# === 3. Aktivierungsfunktionen ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def relu_derivative(x):
    return (x > 0).astype(np.float32)


# === 4. Forward-Funktion ===
def forward(x):
    hidden1_z = np.dot(weights_input_hidden1, x) + bias_hidden1    # (64,)
    hidden1_a = relu(hidden1_z)

    hidden2_z = np.dot(weights_hidden1_hidden2, hidden1_a) + bias_hidden2  # (64,)
    hidden2_a = relu(hidden2_z)

    output_z = np.dot(weights_hidden2_output, hidden2_a) + bias_output    # (62,)
    output_a = softmax(output_z)

    return hidden1_a, hidden2_a, output_a

a1, a2, output_a = forward(img_flat)



# === 5. Vorhersage ===

# Mapping von Label zu Zeichen (EMNIST ByClass – 62 Klassen)
emnist_classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z'
]

for i, prob in enumerate(output_a):
    print(f"Wahrscheinlichkeit für '{emnist_classes[i]}': {prob:.4f}")

digit = np.argmax(output_a)
print(f"Vorhergesagtes Zeichen: '{emnist_classes[digit]}'")

true_char = input("Was war das tatsächliche Zeichen? (z. B. A, b, 3): ")
true_label = emnist_classes.index(true_char)


if digit != true_label:
    print("❌ Falsch erkannt. Lerne aus dem Fehler...")

    learning_rate = 0.01
    y_true = np.zeros(62)
    y_true[true_label] = 1

    error_output = output_a - y_true
    grad_w3 = np.outer(error_output, a2)
    grad_b3 = error_output

    error_hidden2 = np.dot(weights_hidden2_output.T, error_output) * relu_derivative(a2)
    grad_w2 = np.outer(error_hidden2, a1)
    grad_b2 = error_hidden2

    error_hidden1 = np.dot(weights_hidden1_hidden2.T, error_hidden2) * relu_derivative(a1)
    grad_w1 = np.outer(error_hidden1, img_flat)
    grad_b1 = error_hidden1

    weights_hidden2_output -= learning_rate * grad_w3
    bias_output -= learning_rate * grad_b3

    weights_hidden1_hidden2 -= learning_rate * grad_w2
    bias_hidden2 -= learning_rate * grad_b2

    weights_input_hidden1 -= learning_rate * grad_w1
    bias_hidden1 -= learning_rate * grad_b1

    np.savez("trained_model.npz",
             weight_0=weights_input_hidden1,
             bias_0=bias_hidden1,
             weight_1=weights_hidden1_hidden2,
             bias_1=bias_hidden2,
             weight_2=weights_hidden2_output,
             bias_2=bias_output)

    print("✅ Fehler gelernt und Modell gespeichert.")
else:
    print("✅ Korrekt erkannt. Kein Lernen nötig.")
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import ltn

# --- BLOCO DE SEGURANÇA DE IMPORT (A correção para o erro) ---
# Tenta importar Connective, Quantifier, etc. do ltn.core
try:
    Connective = ltn.core.Connective
    Quantifier = ltn.core.Quantifier
    Predicate = ltn.core.Predicate
    Variable = ltn.core.Variable
except AttributeError:
    # Se falhar (por conta de versões muito antigas ou inesperadas), tenta o caminho direto
    try:
        Connective = ltn.Connective
        Quantifier = ltn.Quantifier
        Predicate = ltn.Predicate
        Variable = ltn.Variable
    except AttributeError:
        print("ERRO DE IMPORT: Não foi possível encontrar as classes. Verifique se o restart foi feito.")
        raise
# --------------------------------------------------------------------

# 1. DADOS (Gatos vs Cachorros sintético)
amount_data = 100
data_class_0 = torch.randn(amount_data, 2)          # Gatos (Centro 0,0)
data_class_1 = torch.randn(amount_data, 2) + 4      # Cachorros (Centro 4,4)

# 2. MODELO (Predicado)
class ClassifierModel(nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.dense1 = nn.Linear(2, 16)
        self.dense2 = nn.Linear(16, 16)
        self.dense3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.elu(self.dense1(x))
        x = self.elu(self.dense2(x))
        return self.sigmoid(self.dense3(x))

# Definindo os símbolos LTN
P = Predicate(ClassifierModel())
var_0 = Variable("var_0", data_class_0)
var_1 = Variable("var_1", data_class_1)

# Operadores Lógicos
Not = Connective(ltn.fuzzy_ops.NotStandard())
Forall = Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()

# 3. TREINAMENTO
optimizer = torch.optim.Adam(P.model.parameters(), lr=0.001)
epochs = 1000

print("Iniciando treinamento...")
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Axiomas:
    # "Para todo x da classe 1, P(x) é verdade"
    axiom1 = Forall(var_1, P(var_1))
    # "Para todo x da classe 0, P(x) NÃO é verdade"
    axiom2 = Forall(var_0, Not(P(var_0)))
    
    # Satisfação total
    sat = SatAgg(axiom1, axiom2)
    loss = 1. - sat
    
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Sat: {sat.item():.4f}")

# 4. VISUALIZAÇÃO (Omitido para brevidade, mas o código completo irá rodar)
# ... (código de visualização aqui) ...

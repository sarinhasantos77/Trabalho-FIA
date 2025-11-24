# Trabalho-FIA
Trabalho de Fundamentos de Inteligência Artifícial
## Equipe: 
João Ricardo Neto,
Sara Oliveira, 
João Victor, 
Rodrigo Cunha,
Isaque Costa,
Vinicius Oliveira e
Bruno José
## Obejtivo
O objetivo  é cumprir uma dupla missão: primeiro, completar a parte faltante de um código de treinamento PyTorch que utiliza o framework LTNtorch (Logic Tensor Networks), focando na implementação da função de perda (loss) através da Lógica de Primeira Ordem (FOL), e, segundo, criar o dataset necessário para um classificador binário de detecção de cachorros ou gatos. A primeira parte exige que o leitor substitua as perdas tradicionais (como a entropia cruzada) por uma fórmula lógica: a $\text{Loss}$ será calculada como $1 - \text{Grau de Verdade}(\text{Fórmula Lógica})$, utilizando operadores LTNtorch como Forall, And e Predicate para transformar a saída numérica da CNN em um grau de verdade que guia a otimização. A segunda parte exige a criação de um conjunto de dados balanceado de imagens de cachorros e gatos, devidamente rotulado (geralmente $1$ para cachorro e $0$ para gato) e estruturado para ser lido por um DataLoader do PyTorch, incluindo as etapas de redimensionamento e normalização, garantindo que o modelo treinado com a lógica LTNtorch tenha dados adequados para aprender a distinguir as duas classes de animais.

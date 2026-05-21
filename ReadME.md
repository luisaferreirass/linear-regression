# Regressão Linear Simples com Gradiente Descendente

Este projeto implementa uma **Regressão Linear Simples** do zero utilizando Python e Gradiente Descendente. O objetivo é prever salários com base nos anos de experiência profissional.

O modelo foi desenvolvido manualmente, sem utilizar a classe pronta de regressão linear do Scikit-learn, permitindo compreender melhor o funcionamento interno do treinamento, da atualização dos parâmetros e da avaliação do modelo.

## Objetivo

O objetivo principal do projeto é aplicar os conceitos de Regressão Linear Simples e Gradiente Descendente para construir um modelo capaz de prever o salário de uma pessoa a partir da quantidade de anos de experiência.

A variável independente utilizada foi:

- `YearsExperience`: anos de experiência profissional.

A variável dependente utilizada foi:

- `Salary`: salário correspondente.

## Dataset

O dataset utilizado foi o **Salary Dataset**, baixado diretamente do Kaggle por meio da biblioteca `kagglehub`.

```python
path = kagglehub.dataset_download("rsadiq/salary")
```

O arquivo CSV é carregado com a biblioteca Pandas:

```python
df = pd.read_csv(csv_files[0])
```

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- KaggleHub

## Estrutura do modelo

O modelo foi implementado por meio da classe `LinearRegression`.

Essa classe possui os seguintes atributos principais:

| Atributo | Descrição |
|---|---|
| `weight` | Peso da regressão, equivalente ao coeficiente angular da reta |
| `bias` | Viés da regressão, equivalente ao intercepto da reta |
| `learning_rate` | Taxa de aprendizado usada no Gradiente Descendente |
| `max_iterations` | Número máximo de iterações do treinamento |
| `error_tolerance` | Critério de parada baseado na diferença entre erros |

A equação utilizada pelo modelo é:

```txt
y = weight * x + bias
```

## Funcionamento do treinamento

O treinamento do modelo é realizado com **Gradiente Descendente**.

A cada iteração, o algoritmo executa os seguintes passos:

1. Calcula os valores previstos pelo modelo.
2. Calcula a diferença entre os valores previstos e os valores reais.
3. Calcula o erro utilizando MSE.
4. Atualiza o `weight` e o `bias`.
5. Verifica se a diferença entre o erro atual e o erro anterior é menor que a tolerância definida.
6. Encerra o treinamento caso o erro tenha estabilizado ou caso o número máximo de iterações seja atingido.

A função de erro utilizada foi o **MSE**, ou Erro Quadrático Médio. Como o MSE eleva os erros ao quadrado, erros maiores possuem mais impacto no treinamento. Por isso, outliers podem influenciar mais fortemente o ajuste do modelo.

## Normalização dos dados

Antes do treinamento, as variáveis foram normalizadas utilizando média e desvio padrão:

```python
x_norm = (x - x.mean()) / x.std()
y_norm = (y - y.mean()) / y.std()
```

A normalização foi importante para tornar o treinamento mais estável, principalmente porque o modelo utiliza Gradiente Descendente.

Sem normalização, a escala dos valores de salário poderia dificultar a convergência do algoritmo.

## Separação entre treino e teste

Os dados foram divididos em conjunto de treino e conjunto de teste com a função `train_test_split`:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x_norm,
    y_norm,
    test_size=0.3,
    random_state=0
)
```

A divisão utilizada foi:

| Conjunto | Porcentagem |
|---|---:|
| Treino | 70% |
| Teste | 30% |

## Parâmetros utilizados

Os principais parâmetros definidos no modelo foram:

| Parâmetro | Valor |
|---|---:|
| `learning_rate` | `0.1` |
| `max_iterations` | `1000` |
| `error_tolerance` | `0.00001` |

O `learning_rate` foi definido experimentalmente como `0.1`. Como os dados foram normalizados, esse valor permitiu uma convergência estável, sem grandes oscilações no erro.

O modelo precisou de poucas iterações para convergir, indicando que o valor escolhido para a taxa de aprendizado foi adequado para esse conjunto de dados.

## Avaliação do modelo

Após o treinamento, as previsões foram feitas utilizando os dados de teste:

```python
y_pred_norm = model.predict(x_test)
```

Como o modelo foi treinado com os dados normalizados, foi necessário desnormalizar os valores previstos e reais antes da avaliação:

```python
y_real = y_test * y.std() + y.mean()
y_pred = y_pred_norm * y.std() + y.mean()
```

As métricas utilizadas foram:

| Métrica | Descrição |
|---|---|
| R² | Mede quanto da variação da variável alvo é explicada pelo modelo |
| MAE | Mede o erro absoluto médio entre os valores reais e previstos |
| MSE | Mede o erro quadrático médio |
| RMSE | Mede a raiz do erro quadrático médio |

## Resultados

O modelo apresentou um bom desempenho no conjunto de teste.

O valor de R² foi de aproximadamente `0.97`, indicando que o modelo conseguiu explicar cerca de 97% da variação dos salários a partir dos anos de experiência.

Além disso, o RMSE foi de aproximadamente `4.717`, o que significa que, em média, as previsões se afastam dos valores reais em cerca de 4.717 dólares.

## Exemplo de saída

Durante o treinamento, o modelo exibe informações como:

```txt
Quantidade de iterações: 22
MSE final: 0.04
RMSE Final: 0.20
```

Após a avaliação, são exibidas métricas como:

```txt
R²:  0.97
MAE:  ...
MSE:  ...
RMSE:  ...
```

## Possíveis melhorias futuras

Algumas melhorias que podem ser implementadas futuramente são:

* **Cálculo automático do learning rate**

Implementar uma estratégia para testar diferentes valores de learning_rate automaticamente e escolher aquele que gera menor erro com convergência estável.

* **Visualização da convergência**

Armazenar o MSE a cada iteração e gerar um gráfico da curva de perda. Isso permitiria visualizar melhor se o modelo está convergindo, oscilando ou aprendendo lentamente.

* **Comparação com o modelo do scikit-learn**

Comparar os resultados da regressão linear implementada manualmente com a LinearRegression do sklearn, verificando se os pesos, o bias e as métricas ficaram próximos

## Como executar o projeto

Primeiro, instale as dependências necessárias:

```bash
pip install pandas numpy scikit-learn kagglehub
```

Depois, execute o notebook ou arquivo Python contendo o código do projeto.

O dataset será baixado automaticamente por meio da biblioteca `kagglehub`.

## Conclusão

Este projeto demonstra, de forma prática, como uma Regressão Linear Simples pode ser implementada manualmente utilizando Gradiente Descendente.

A implementação ajuda a compreender conceitos importantes de Machine Learning, como normalização, função de erro, atualização de parâmetros, learning rate, convergência e métricas de avaliação.

O modelo apresentou bom desempenho para o dataset utilizado, mostrando que a relação entre anos de experiência e salário pode ser bem representada por uma regressão linear simples.

# Reporte Técnico del Candidato (`REPORT.md`)

> **Instrucciones:** Responde de forma concisa (máximo 1 página en total) basándote exclusivamente en tu código y en los resultados medidos al ejecutar `./build/benchmark_cli data/train.csv data/test.csv`.

## 1. Diagnóstico del Código Base (Etapa 1)
1. **¿Qué defecto de POO/memoria existía en `IDistanceMetric`? ¿En qué escenarios concretos provoca problemas al destruir una `FisherWeightedL2Metric` a través de un puntero a la clase base, y se manifestaba con la forma en que este código gestiona la propiedad de las métricas? Justifica.**
   * *Respuesta:*
2. **¿Cuántas alocaciones en el heap por consulta hacía `FlatIndex::search` originalmente (exprésalo en función de *N*) y qué estructura utilizaste sobre `SearchResult` para reducirlas a `0`?**
   * *Respuesta:*

## 2. Diagnóstico de los Datos y Machine Learning (Etapa 2)
1. **¿Cuántas filas corruptas descartó `is_valid_row` en `data/train.csv` y qué hubiera ocurrido en `IVFIndex::build` (*k*-Means++ y Lloyd) si no se hubieran filtrado las filas con `NaN` y con `-9999.0`? Distingue ambos casos.**
   * *Respuesta:*
2. **¿Qué valores de peso aprendió `FisherWeightedL2Metric` para las dimensiones `f0..f23` frente a `f24..f31`? ¿Por qué fallaba la distancia Euclídea sin ponderar?**
   * *Respuesta:*
3. **Compara el Macro-F1 (Flat) de cuatro reglas de decisión: mayoría simple, solo prior (π<sub>c</sub>)<sup>−α</sup>, solo kernel Gaussiano y la regla completa. ¿Qué componente aporta más y por qué?**
   * *Respuesta:*

## 3. Tabla de Resultados (`benchmark_cli`)
Pega aquí la salida generada por `./build/benchmark_cli data/train.csv data/test.csv`:

```text
(Pegar salida de benchmark_cli aquí)
```

## 4. Diseño de C++: Despacho Virtual en el *Hot Loop*
* **En el cálculo de distancias entre un `query` y los vectores de una lista invertida, ¿cómo evitaste o mitigaste el sobrecosto de llamadas virtuales repetidas (`metric_->compute(...)`)?**
  * *Respuesta:*

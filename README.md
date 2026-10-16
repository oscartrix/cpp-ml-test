# Prueba Técnica C++17 y Machine Learning: `VecClassify`

## Tiempo Estimado y Plazo
* **Esfuerzo estimado:** 10 a 14 horas efectivas.
* **Plazo máximo de entrega:** 5 días naturales.
* **Restricción de dependencias:** Únicamente la librería estándar de C++17 (STL). No se permite añadir librerías externas (Eigen, FAISS, OpenCV, Boost, etc.).

---

## Contexto del Problema

Recibes un conjunto de datos proveniente de un arreglo de sensores industriales de 32 canales (`data/train.csv` con 12,000 muestras y `data/test.csv` con 2,500 muestras, distribuidas en $C = 5$ estados operativos o clases).

Un prototipo previo (`FlatIndex` con distancia Euclídea simple y votación mayoritaria) presenta tres problemas graves en producción:
1. **Baja calidad predictiva (Macro-F1 ≈ 0.2):** Algunas dimensiones del arreglo de sensores contienen ruido ambiental de gran amplitud no correlacionado con la clase, ciertos registros de entrenamiento provienen de sensores desconectados, y la distribución de las 5 clases está fuertemente desbalanceada.
2. **Latencia elevada:** Comparar cada consulta contra todos los vectores de entrenamiento mediante fuerza bruta no escala.
3. **Defectos de memoria y rendimiento en C++:** El código inicial realiza alocaciones dinámicas innecesarias dentro del bucle crítico de inferencia y tiene defectos en su diseño orientado a objetos.

---

## Objetivos de la Prueba

### Etapa 1: Auditoría de C++ y Memoria en el *Hot Path*
1. Audita `include/vec_classify/distance.hpp` y `src/index.cpp` (`FlatIndex::search`).
2. Corrige el defecto de ciclo de vida polimórfico en `IDistanceMetric`.
3. Refactoriza `FlatIndex::search` para eliminar copias innecesarias y mantener los $k$ vecinos más cercanos directamente sobre los buffers pre-reservados de `SearchResult` (`out.indices` y `out.distances`), logrando **0 alocaciones en el heap** durante la consulta (ni `new` ni `malloc`; los tests instrumentan `operator new` y el resto se revisa en el código).

### Etapa 2: Limpieza de Datos y Métrica Supervisada (`FisherWeightedL2Metric`)
1. Implementa `AdaptiveKNNClassifier::is_valid_row` en `src/classifier.cpp` para descartar filas de entrenamiento con lecturas corruptas (`NaN`, `Inf` o valores centinela de desconexión de sensor como `-9999.0`).
2. Implementa `CosineMetric::compute` y `FisherWeightedL2Metric::fit_weights` / `compute` en `src/distance.cpp`.
   Para cada dimensión $j \in \lbrace 0, \dots, D-1 \rbrace$, el puntaje de relevancia de Fisher se define como la razón entre la varianza inter-clase y la varianza intra-clase regularizada sobre las $N_{\text{valid}}$ muestras válidas:

   ```math
   \text{num}_j = \sum_{c=0}^{C-1} \frac{N_c}{N_{\text{valid}}} (\mu_{c,j} - \mu_j)^2
   ```

   ```math
   \text{den}_j = \frac{1}{N_{\text{valid}}} \sum_{c=0}^{C-1} \sum_{i \in c} (x_{i,j} - \mu_{c,j})^2 + \gamma
   ```

   ```math
   s_j = \frac{\text{num}_j}{\text{den}_j}, \qquad w_j = D \cdot \frac{s_j}{\sum_{l=0}^{D-1} s_l + 10^{-12}}
   ```

### Etapa 3: Índice de Archivo Invertido (`IVFIndex`) con $k$-Means++
Implementa `IVFIndex::build` e `IVFIndex::search` en `src/index.cpp`:
1. **Entrenamiento (`build`)**:
   * Inicializa los `n_lists` centroides usando el algoritmo **$k$-Means++** (probabilidad proporcional a `metric_->compute` al centroide más cercano ya elegido; las métricas L2 de este proyecto ya devuelven la distancia **al cuadrado**, no la vuelvas a elevar) con semilla fija `seed_`.
   * Ejecuta hasta `max_kmeans_iters_` iteraciones de asignación y actualización de centroides (criterio de parada anticipada si ninguna asignación cambia).
   * Agrupa los vectores de cada cluster $c$ de forma contigua en `list_vectors_[c]` y guarda sus índices originales en `list_ids_[c]`.
2. **Búsqueda sin alocaciones (`search`)**:
   * Evalúa la distancia de `query` a los `n_lists` centroides usando `out.probe_scratch` (evitando alocar en el heap) y selecciona las `n_probe` celdas más cercanas.
   * Recorre los vectores de esas `n_probe` celdas y devuelve los `k` más cercanos ordenados de menor a mayor distancia en `out.indices[0..count-1]` y `out.distances[0..count-1]`, manteniendo **0 alocaciones de heap**.

### Etapa 4: Clasificador $k$-NN Ponderado y Reporte Técnico
1. Implementa la regla de votación en `AdaptiveKNNClassifier::predict_one` (`src/classifier.cpp`) combinando un kernel Gaussiano de distancia y compensación por prior empírico de clase $\pi_c = N_c / N_{\text{valid}}$:

   ```math
   \text{Score}(c \mid \mathbf{q}) = (\pi_c)^{-\alpha} \sum_{i \in \mathcal{N}_k(\mathbf{q}),\, y_i = c} \exp\left(-\frac{d_{\mathbf{w}}(\mathbf{q}, \mathbf{x}_i)}{2\tau^2}\right)
   ```

   Usa los valores por defecto de `ClassifierConfig` ($\tau = 5.0$, $\alpha = 0.6$, $k = 11$).

2. Ejecuta `./benchmark_cli` y completa las preguntas concisas en `REPORT_TEMPLATE.md` (renombrándolo o guardándolo como `REPORT.md`).

---

## Cómo Compilar y Ejecutar

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j

# Ejecutar suite de pruebas unitarias y auditoría de memoria
./test_suite

# Ejecutar benchmark sobre el dataset (desde la raíz del repositorio)
cd ..
./build/benchmark_cli data/train.csv data/test.csv
```

Para compilar con `AddressSanitizer` y `UndefinedBehaviorSanitizer` durante el desarrollo:
```bash
cmake -B build_asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
cmake --build build_asan -j
./build_asan/test_suite
```

Para verificar todos los criterios automáticos de una vez (Release + ASan/UBSan + benchmark):
```bash
./scripts/run_acceptance.sh
```

---

## Criterios de Aceptación Automáticos
1. `./test_suite` pasa el **100% de las pruebas**, tanto en Release como bajo `ENABLE_ASAN=ON` (incluye 0 alocaciones en `FlatIndex::search`, `IVFIndex::search` y `AdaptiveKNNClassifier::predict_one`).
2. En `./benchmark_cli data/train.csv data/test.csv`, con `nprobe = 8` (de `n_lists = 64`):
   * **Macro-F1 $\ge 0.87$**, **Recall@10 $\ge 0.80$**, **Speedup vs Flat $\ge 3\times$** y **Heap Allocs/q = `0.0`**.
   * El benchmark imprime `CUMPLE` / `NO CUMPLE` al final y sale con código distinto de 0 si no se cumplen. El speedup depende del hardware: mídelo en Release.
3. `REPORT.md` completado con la tabla de resultados y la justificación técnica de tus decisiones.

---

## Entrega

1. Pulsa **"Use this template" → "Create a new repository"** en GitHub y crea el repositorio como **privado**. No hagas *fork*: un fork de un repositorio público es siempre público.
2. Trabaja en tu repositorio y sube tu solución junto con `REPORT.md`.
3. En **Settings → Collaborators**, invita como colaborador a `oscartrix` antes de la fecha límite.

Si no puedes usar GitHub, envía un `git bundle` (`git bundle create entrega.bundle --all`) o un `.zip` del repositorio **sin** las carpetas `build*/`.

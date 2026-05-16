import tensorflow as tf


class MultiDense(tf.keras.layers.Layer):
    """Capa densa con múltiples funciones de activación aplicadas por particiones.

    Generaliza la capa `Dense` estándar permitiendo asignar una función de
    activación distinta a cada subconjunto (partición) de neuronas de salida.
    Dentro de cada partición el comportamiento es idéntico al de una capa Dense
    convencional: se computa Z = X·W + b y luego se aplica la activación
    correspondiente al bloque de columnas asociado.

    Matemáticamente, si la capa tiene particiones de tamaños [n₁, n₂, …, nₖ]
    con activaciones [ϕ₁, ϕ₂, …, ϕₖ], la salida es:

        A = concat(ϕ₁(Z[:, :n₁]), ϕ₂(Z[:, n₁:n₁+n₂]), …)

    donde Z = X·W + b se calcula una sola vez con la matriz de pesos completa.

    Args:
        units: Lista de enteros positivos con el número de neuronas de cada
            partición. El total de neuronas de la capa es sum(units).
        activations: Lista de funciones de activación, una por partición.
            Cada elemento puede ser un identificador de cadena (p. ej. 'relu',
            'sigmoid', 'linear'), un callable, o None (equivale a 'linear').
            Debe tener la misma longitud que `units`.
        use_bias: Booleano. Si es True (por defecto) la capa incluye un vector
            de sesgos.
        kernel_initializer: Inicializador para la matriz de pesos `kernel`.
            Por defecto 'glorot_uniform'.
        bias_initializer: Inicializador para el vector de sesgos `bias`.
            Por defecto 'zeros'.
        kernel_regularizer: Regularizador aplicado a la matriz de pesos.
        bias_regularizer: Regularizador aplicado al vector de sesgos.
        activity_regularizer: Regularizador aplicado a la salida de la capa.
        kernel_constraint: Restricción aplicada a la matriz de pesos.
        bias_constraint: Restricción aplicada al vector de sesgos.

    Ejemplo de uso:
        >>> layer = MultiDense(
        ...     units=[4, 2],
        ...     activations=['relu', 'sigmoid'],
        ... )
        >>> x = tf.random.normal((8, 16))
        >>> y = layer(x)          # shape: (8, 6)
        >>> y[:, :4]              # activadas con ReLU
        >>> y[:, 4:]              # activadas con Sigmoid
    """

    def __init__(
        self,
        units,
        activations,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        # Validaciones previas a la llamada al constructor padre, igual que
        # hace la capa Dense oficial antes de asignar sus atributos.
        units = list(units)
        activations = list(activations)

        if len(units) != len(activations):
            raise ValueError(
                "Los parámetros `units` y `activations` deben tener la misma "
                f"longitud. Recibido: len(units)={len(units)}, "
                f"len(activations)={len(activations)}."
            )

        if not units:
            raise ValueError(
                "`units` no puede estar vacío. Debe contener al menos una partición."
            )

        for i, u in enumerate(units):
            if not isinstance(u, int) or u <= 0:
                raise ValueError(
                    "Todos los elementos de `units` deben ser enteros "
                    f"positivos. Recibido units[{i}]={u}."
                )

        # activity_regularizer se pasa al padre directamente porque Layer lo
        # gestiona internamente (no se almacena como atributo explícito aquí).
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        self.units = units
        # Resolvemos los identificadores de activación a callables mediante la
        # misma función que usa Keras internamente.
        self._activations_arg = activations  # guardamos el original para get_config
        self.activations = [tf.keras.activations.get(act) for act in activations]

        self.use_bias = use_bias

        # Inicializadores: usamos initializers.get() igual que Dense, para que
        # se acepten tanto cadenas ('glorot_uniform') como instancias
        # (GlorotUniform()) y se serialicen correctamente.
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

        # Regularizadores
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        # Restricciones
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        # InputSpec: exige al menos rango 2 (batch + features), igual que Dense.
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        total_units = sum(self.units)

        # Un único tensor de pesos para todas las particiones.
        # La columna j-ésima de self.kernel corresponde a la j-ésima neurona
        # de la capa, independientemente de la partición a la que pertenezca.
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, total_units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(total_units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
        else:
            self.bias = None

        # Restringir la dimensión de entrada para detección de errores de forma.
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: input_dim})

        # Marcamos la capa como construida (Dense llama a self.built = True
        # en lugar de super().build() en versiones recientes de Keras).
        self.built = True

    def call(self, inputs):
        # Preactivación conjunta: Z tiene shape (..., total_units).
        # Usamos tf.matmul para entradas 2-D; para rangos superiores Dense
        # usa tf.tensordot, pero lo mantenemos simple por claridad.
        z = tf.matmul(inputs, self.kernel)
        if self.bias is not None:
            z = tf.nn.bias_add(z, self.bias)

        # Particionamos Z en bloques según los tamaños indicados en self.units.
        z_splits = tf.split(z, self.units, axis=-1)

        # Aplicamos la función de activación correspondiente a cada bloque.
        a_splits = [
            activation(z_block)
            for activation, z_block in zip(self.activations, z_splits)
        ]

        # Concatenamos las activaciones parciales para obtener la salida final.
        return tf.concat(a_splits, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                # serialize devuelve una cadena o un dict con la configuración
                # completa de la función (nombre + parámetros si los tiene).
                "activations": [
                    tf.keras.activations.serialize(act) for act in self.activations
                ],
                "use_bias": self.use_bias,
                "kernel_initializer": tf.keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": tf.keras.initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": tf.keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": tf.keras.regularizers.serialize(
                    self.bias_regularizer
                ),
                "activity_regularizer": tf.keras.regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": tf.keras.constraints.serialize(
                    self.kernel_constraint
                ),
                "bias_constraint": tf.keras.constraints.serialize(self.bias_constraint),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Las activaciones se deserializan antes de pasarlas al constructor
        # porque get() acepta callables directamente.
        config = config.copy()
        config["activations"] = [
            tf.keras.activations.deserialize(act) for act in config["activations"]
        ]
        # Los inicializadores, regularizadores y restricciones se pasan tal
        # cual: el constructor llama a initializers.get() etc., que sabe
        # manejar tanto dicts serializados como cadenas.
        return cls(**config)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (sum(self.units),)

"""Replay buffer con deduplicacion de frames.

Guardar la observacion apilada completa (4, 84, 84) por transicion desperdicia
4x memoria, porque dos pasos consecutivos del mismo entorno comparten 3 frames.
1M de transiciones costarian 26 GB apilados contra 6.6 GB deduplicados.

La deduplicacion se apoya en una invariante de escritura en lockstep: en cada
paso se escriben exactamente num_envs transiciones, una por entorno, siempre en
el mismo orden. Asi el paso anterior del entorno e esta siempre en el slot
anterior, y el stack se reconstruye leyendo hacia atras. Las posiciones que
cruzan el inicio de un episodio se rellenan con ceros, que es exactamente lo
que hace AtariVectorEnv al reiniciar.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np

# Slots reservados delante del cursor de escritura, para que el muestreo en
# segundo plano nunca lea lo que el hilo principal esta sobrescribiendo.
GUARDA = 4


@dataclass
class Batch:
    obs: np.ndarray          # (B, stack, H, W) uint8
    actions: np.ndarray      # (B,) int64
    returns: np.ndarray      # (B,) float32 - retorno n-step descontado
    next_obs: np.ndarray     # (B, stack, H, W) uint8
    done: np.ndarray         # (B,) bool - hubo terminacion dentro de la ventana
    n_steps: np.ndarray      # (B,) int32 - pasos efectivos (para gamma**n)
    indices: np.ndarray      # (B, 2) int64 - (slot, env), para PER
    weights: np.ndarray | None = None   # (B,) float32 - pesos de importancia (PER)


class ReplayBuffer:
    """Buffer circular uniforme con retornos n-step.

    Escribe en lockstep: cada llamada a `add` recibe la transicion de los
    num_envs entornos a la vez.
    """

    def __init__(self, capacity: int, num_envs: int, *, stack: int = 4,
                 height: int = 84, width: int = 84, gamma: float = 0.99,
                 n_step: int = 1, seed: int = 0):
        if capacity % num_envs != 0:
            capacity = (capacity // num_envs) * num_envs
        self.num_slots = capacity // num_envs
        if self.num_slots <= n_step + stack:
            raise ValueError("capacidad demasiado pequena para el stack y el n-step")

        self.capacity = self.num_slots * num_envs
        self.num_envs = num_envs
        self.stack = stack
        self.height = height
        self.width = width
        self.gamma = gamma
        self.n_step = n_step
        self.rng = np.random.default_rng(seed)

        self.frames = np.zeros((self.num_slots, num_envs, height, width), dtype=np.uint8)
        # Vista plana (slot * num_envs + env): permite reunir todos los frames
        # del lote con un unico np.take, 3x mas rapido que indexar por separado.
        self.frames_flat = self.frames.reshape(self.num_slots * num_envs, height, width)
        self.actions = np.zeros((self.num_slots, num_envs), dtype=np.int64)
        self.rewards = np.zeros((self.num_slots, num_envs), dtype=np.float32)
        self.terminated = np.zeros((self.num_slots, num_envs), dtype=bool)
        self.ep_step = np.zeros((self.num_slots, num_envs), dtype=np.int32)
        self.valid = np.zeros((self.num_slots, num_envs), dtype=bool)

        self.pos = 0
        self.full = False
        self._ep_step_actual = np.zeros(num_envs, dtype=np.int32)

    @property
    def size(self) -> int:
        return self.capacity if self.full else self.pos * self.num_envs

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in
                   (self.frames, self.actions, self.rewards, self.terminated, self.ep_step))

    def add(self, newest_frame: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            terminated: np.ndarray, dummy: np.ndarray | None = None) -> None:
        """Agrega un paso de los num_envs entornos.

        newest_frame: (num_envs, H, W) uint8 - el frame mas reciente del stack
                      en el instante de la observacion, es decir obs[:, -1].
        terminated:   terminacion real del episodio (no perdida de vida).
        dummy:        entornos que venian de terminar. Con AutoresetMode.NEXT_STEP
                      el paso siguiente a una terminacion ignora la accion y solo
                      devuelve la observacion de reinicio, asi que esa transicion
                      se escribe para no romper el lockstep pero se marca invalida.
        """
        if dummy is None:
            dummy = np.zeros(self.num_envs, dtype=bool)

        s = self.pos
        self.frames[s] = newest_frame
        self.actions[s] = actions
        self.rewards[s] = np.where(dummy, 0.0, rewards)
        self.terminated[s] = terminated & ~dummy
        self.ep_step[s] = self._ep_step_actual
        self.valid[s] = ~dummy

        # Tras un dummy el contador sigue en 0: el paso siguiente es el primero
        # real del episodio nuevo.
        self._ep_step_actual = np.where(
            self.terminated[s] | dummy, 0, self._ep_step_actual + 1
        )

        self.pos += 1
        if self.pos >= self.num_slots:
            self.pos = 0
            self.full = True

    def _build_stack(self, slots: np.ndarray, envs: np.ndarray) -> np.ndarray:
        """Reconstruye (B, stack, H, W) leyendo hacia atras desde cada slot."""
        return self._build_stacks((slots,), envs)[0]

    def _build_stacks(self, bases: tuple[np.ndarray, ...], envs: np.ndarray) -> np.ndarray:
        """Reconstruye varios stacks a la vez con un unico np.take.

        Reunir los 2*stack frames del lote en una sola llamada es ~3x mas rapido
        que hacer una indexacion por cada posicion del stack: el muestreo pasaba
        de 13.8 ms a 4.3 ms por lote de 512, y era el 57% del paso de
        entrenamiento.
        """
        n = len(bases)
        b = envs.shape[0]
        idx = np.empty((n, b, self.stack), dtype=np.int64)
        disponible = np.empty((n, b, self.stack), dtype=bool)

        for w, base in enumerate(bases):
            ep = self.ep_step[base, envs]
            for k in range(self.stack):
                atras = self.stack - 1 - k      # 3, 2, 1, 0 frames hacia atras
                origen = (base - atras) % self.num_slots
                idx[w, :, k] = origen * self.num_envs + envs
                disponible[w, :, k] = ep >= atras   # si no, cae antes del reset

        out = np.take(self.frames_flat, idx.reshape(-1), axis=0, mode="clip")
        out = out.reshape(n, b, self.stack, self.height, self.width)
        out[~disponible] = 0
        return out

    def _sample_indices(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Pares (slot, env) validos: escritos, no dummy y con n_step sucesores.

        Se excluye tambien una franja de guarda justo delante del cursor de
        escritura. Sin ella, el muestreo en segundo plano podria leer los slots
        mas antiguos mientras el hilo principal los sobrescribe.
        """
        disponibles = self.num_slots if self.full else self.pos
        margen = self.n_step
        # Antes de la primera vuelta no se sobrescribe nada, asi que no hace
        # falta reservar la franja de guarda.
        guarda = GUARDA if self.full else 0
        if disponibles <= margen + guarda + 1:
            raise ValueError("buffer insuficiente para muestrear")

        limite = self.num_slots - guarda
        slots = np.empty(batch_size, dtype=np.int64)
        envs = np.empty(batch_size, dtype=np.int64)
        escritos = 0
        while escritos < batch_size:
            faltan = batch_size - escritos
            cs = self.rng.integers(0, disponibles, size=faltan * 2)
            ce = self.rng.integers(0, self.num_envs, size=faltan * 2)
            # antiguedad: 0 es el slot recien escrito
            edad = (self.pos - 1 - cs) % self.num_slots
            ok = (edad >= margen) & (edad < limite) & self.valid[cs, ce]
            cs, ce = cs[ok], ce[ok]
            toma = min(faltan, cs.shape[0])
            slots[escritos:escritos + toma] = cs[:toma]
            envs[escritos:escritos + toma] = ce[:toma]
            escritos += toma

        return slots, envs

    def sample(self, batch_size: int) -> Batch:
        slots, envs = self._sample_indices(batch_size)
        return self._make_batch(slots, envs)

    def _make_batch(self, slots: np.ndarray, envs: np.ndarray) -> Batch:
        batch_size = slots.shape[0]

        # Retorno n-step, cortando en la primera terminacion de la ventana.
        retornos = np.zeros(batch_size, dtype=np.float32)
        done = np.zeros(batch_size, dtype=bool)
        pasos = np.zeros(batch_size, dtype=np.int32)
        vivo = np.ones(batch_size, dtype=bool)
        descuento = np.ones(batch_size, dtype=np.float32)

        for k in range(self.n_step):
            s = (slots + k) % self.num_slots
            r = self.rewards[s, envs]
            t = self.terminated[s, envs]
            retornos += np.where(vivo, descuento * r, 0.0)
            pasos += vivo.astype(np.int32)
            done |= vivo & t
            vivo &= ~t
            descuento *= self.gamma

        # El estado siguiente es el del ultimo paso efectivo de la ventana.
        slots_next = (slots + pasos) % self.num_slots
        obs, next_obs = self._build_stacks((slots, slots_next), envs)

        return Batch(
            obs=obs,
            actions=self.actions[slots, envs],
            returns=retornos,
            next_obs=next_obs,
            done=done,
            n_steps=pasos,
            indices=np.stack([slots, envs], axis=1),
        )


class SumTree:
    """Arbol de sumas con actualizacion y descenso vectorizados.

    Un arbol de sumas clasico hace O(log n) por muestra en Python puro, lo que a
    512 muestras por lote domina el paso de entrenamiento. Aqui el descenso se
    hace nivel por nivel sobre el lote completo: son ~19 operaciones de numpy
    sobre arrays de 512, no 512 recorridos independientes.
    """

    def __init__(self, capacity: int):
        self.n = 1
        while self.n < capacity:
            self.n <<= 1
        self.tree = np.zeros(2 * self.n, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def set(self, idx: np.ndarray, valores: np.ndarray) -> None:
        idx = np.asarray(idx, dtype=np.int64)
        valores = np.asarray(valores, dtype=np.float64)
        # Con indices repetidos el delta se calcularia sobre un valor ya pisado.
        idx, primero = np.unique(idx, return_index=True)
        valores = valores[primero]

        p = idx + self.n
        delta = valores - self.tree[p]
        self.tree[p] = valores
        p = p >> 1
        while True:
            np.add.at(self.tree, p, delta)
            if p[0] == 1:
                break
            p = p >> 1

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Muestreo estratificado proporcional. Devuelve (hojas, probabilidades)."""
        total = self.total
        if total <= 0.0:
            raise ValueError("el arbol de prioridades esta vacio")
        bordes = np.linspace(0.0, total, n + 1)
        objetivos = rng.uniform(bordes[:-1], bordes[1:])

        idx = np.ones(n, dtype=np.int64)
        while idx[0] < self.n:
            izq = idx << 1
            v_izq = self.tree[izq]
            derecha = objetivos > v_izq
            objetivos = np.where(derecha, objetivos - v_izq, objetivos)
            idx = izq + derecha

        return idx - self.n, self.tree[idx] / total


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay priorizado proporcional (Schaul et al., 2016).

    Muestrea las transiciones con probabilidad proporcional a |error TD|^alpha y
    corrige el sesgo que eso introduce con pesos de importancia con exponente
    beta, que se recocen hasta 1 al final del entrenamiento.

    Una transicion recien escrita no es muestreable hasta que su ventana n-step
    esta completa: se le da prioridad 0 al escribirla y prioridad maxima n_step
    pasos despues. La franja de guarda delante del cursor tambien se mantiene en
    0 para que el muestreo en segundo plano nunca lea lo que se esta pisando.
    """

    def __init__(self, *args, alpha: float = 0.5, beta_start: float = 0.4,
                 beta_end: float = 1.0, beta_steps: int = 10_000_000,
                 prio_eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.prio_eps = prio_eps
        self.beta = beta_start
        self.max_prio = 1.0
        self.tree = SumTree(self.capacity)
        self._indices_env = np.arange(self.num_envs, dtype=np.int64)
        # El arbol lo tocan el hilo principal (escritura y actualizacion de
        # prioridades) y el de prefetch (muestreo). El candado cubre solo el
        # arbol: la reunion de frames, que es lo caro, sigue en paralelo.
        self._lock = threading.Lock()

    def set_beta_por_paso(self, step: int) -> None:
        frac = min(1.0, step / max(self.beta_steps, 1))
        self.beta = self.beta_start + frac * (self.beta_end - self.beta_start)

    def _hojas(self, slot: int) -> np.ndarray:
        return slot * self.num_envs + self._indices_env

    def add(self, newest_frame, actions, rewards, terminated, dummy=None) -> None:
        s = self.pos
        super().add(newest_frame, actions, rewards, terminated, dummy)

        ceros = np.zeros(self.num_envs)
        listo = (s - self.n_step) % self.num_slots
        with self._lock:
            # el slot recien escrito aun no tiene su ventana n-step completa
            self.tree.set(self._hojas(s), ceros)
            # el slot que acaba de entrar en la franja de guarda
            self.tree.set(self._hojas((s + GUARDA) % self.num_slots), ceros)
            # el slot cuya ventana ya se completo pasa a ser muestreable
            if self.full or listo < s:
                self.tree.set(self._hojas(listo),
                              np.where(self.valid[listo], self.max_prio, 0.0))

    def sample(self, batch_size: int) -> Batch:
        with self._lock:
            hojas, prob = self.tree.sample(batch_size, self.rng)
        slots = hojas // self.num_envs
        envs = hojas % self.num_envs

        lote = self._make_batch(slots, envs)

        n = max(self.size, 1)
        pesos = (n * np.maximum(prob, 1e-12)) ** (-self.beta)
        lote.weights = (pesos / pesos.max()).astype(np.float32)
        return lote

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        hojas = indices[:, 0] * self.num_envs + indices[:, 1]
        prio = (np.abs(td_errors) + self.prio_eps) ** self.alpha
        self.max_prio = max(self.max_prio, float(prio.max()))
        with self._lock:
            self.tree.set(hojas, prio)


class PrefetchSampler:
    """Muestrea lotes en un hilo aparte, solapando con el trabajo de la GPU.

    El muestreo es el componente mas caro del paso de entrenamiento (~4.3 ms de
    los ~14 ms totales) y `np.take` libera el GIL, asi que el hilo corre de
    verdad en paralelo con el update.
    """

    def __init__(self, buffer: ReplayBuffer, batch_size: int, profundidad: int = 3):
        self.buffer = buffer
        self.batch_size = batch_size
        self._cola: queue.Queue = queue.Queue(maxsize=profundidad)
        self._parar = threading.Event()
        self._error: BaseException | None = None
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

    def _bucle(self) -> None:
        while not self._parar.is_set():
            try:
                lote = self.buffer.sample(self.batch_size)
            except BaseException as exc:  # se relanza en el hilo principal
                self._error = exc
                self._parar.set()
                return
            while not self._parar.is_set():
                try:
                    self._cola.put(lote, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def get(self) -> Batch:
        while True:
            if self._error is not None:
                raise self._error
            try:
                return self._cola.get(timeout=0.1)
            except queue.Empty:
                continue

    def close(self) -> None:
        self._parar.set()
        self._hilo.join(timeout=2.0)

from .conn_snn import ConnSNN
from .conn_snn_frozen import ConnSNN_frozen
from .dense_snn import DenseSNN

from .dense_mlp import DenseMLP
from .dense_gru import DenseGRU
from .dense_lstm import DenseLSTM


NETWORKS = {
    "ConnSNN": ConnSNN,
    "ConnSNN_frozen": ConnSNN_frozen,
    # Dense
    "DenseSNN": DenseSNN,

    "DenseMLP": DenseMLP,
    "DenseGRU": DenseGRU,
    "DenseLSTM": DenseLSTM
}

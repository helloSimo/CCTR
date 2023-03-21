import torch
import torch.nn as nn
from torch import Tensor
import logging
from .encoder import EncoderPooler, EncoderModel

logger = logging.getLogger(__name__)


class DensePooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 768, tied=True, normalize=False):
        super(DensePooler, self).__init__()
        self.normalize = normalize
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied, 'normalize': normalize}

    def forward(self, q: Tensor = None, p: Tensor = None, **kwargs):
        if q is not None:
            rep = self.linear_q(q)
        elif p is not None:
            rep = self.linear_p(p)
        else:
            raise ValueError
        if self.normalize:
            rep = nn.functional.normalize(rep, dim=-1)
        return rep


class DenseModel(EncoderModel):
    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(psg)
        if self.pooler is not None:
            psg_out = self.pooler(p=psg_out)  # D * d
        return psg_out

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(qry)
        if self.pooler is not None:
            qry_out = self.pooler(q=qry_out)
        return qry_out

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = DensePooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @staticmethod
    def build_pooler(model_args):
        pooler = DensePooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder,
            normalize=model_args.normalize
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

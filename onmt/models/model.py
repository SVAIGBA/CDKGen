""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class word_memory(nn.Module):

    def __init__(self, embedding_b, all_docs):
        super(word_memory, self).__init__()
        self.word_embedding_a = nn.Parameter(all_docs.permute(1,0))
        self.word_embedding_b = embedding_b
        self.word_embedding_c = nn.Parameter(all_docs)

    def forward(self, word_seq, all_docs):
        '''
            word_seq: [300,13,1] [seq_len, batch_size, 1]
            all_docs: [300,1100,1] [seq_len, all_docs_num, 1]
            all_docs_emb: [550000,768]
            embedding word_seq by sen_transformer   [300,13,1] - > [13,768]
        '''

        usetopk = False  # use only topk similar documents
        u = torch.mean(self.word_embedding_b(word_seq),0)
        p = F.softmax(torch.matmul(u, self.word_embedding_a), dim=1)
        if usetopk:
            #only use topk similar documents
            prob, idx = p.topk(5, dim=1)
            m = m.permute(1, 0)
            topkm = m[idx]
            topkc = c[idx]
            u = u.unsqueeze(2)
            topkp = F.softmax(torch.matmul(topkm, u), dim=1).permute(0, 2, 1)  # [13, 1, 5]
            o = torch.matmul(topkp, topkc).squeeze(1)
        else:
            #use all 550000 documents
            o = torch.matmul(p, self.word_embedding_c)


        return o

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder, all_docs=None, model_opt=None, embedding_b=None):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.all_docs = all_docs
        self.cal_word_memory = word_memory(embedding_b, all_docs)

    def forward(self, src, tgt, lengths, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        # src [300,13,1] [seq_len, batch_size, 1]
        # tgt [5,13,1] [tgt_len, batch, 1]
        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        if self.all_docs is not None:
            o = self.cal_word_memory(src, self.all_docs)
            memory_bank = torch.add(memory_bank, o)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

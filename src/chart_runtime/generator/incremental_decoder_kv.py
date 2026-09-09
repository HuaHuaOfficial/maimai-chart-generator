"""Eval-time TransformerDecoder step with projected self/cross K/V caches."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class LayerCache:
    self_k:torch.Tensor|None=None
    self_v:torch.Tensor|None=None
    cross_k:torch.Tensor|None=None
    cross_v:torch.Tensor|None=None
@dataclass
class DecoderKVState:
    layers:list[LayerCache]
    index:int=0
    decoder_ref:object=None
    memory_ref:object=None
    memory_version:object=None

def new_state(decoder):
    if decoder.training:raise ValueError("KV decoding requires eval mode")
    for layer in decoder.layers:
        for mha in (layer.self_attn,layer.multihead_attn):
            if not mha.batch_first or mha._qkv_same_embed_dim is not True or mha.bias_k is not None or mha.bias_v is not None or mha.add_zero_attn:
                raise ValueError('unsupported MultiheadAttention configuration')
        if not layer.norm_first:raise ValueError('requires norm_first decoder')
    return DecoderKVState([LayerCache() for _ in decoder.layers],decoder_ref=decoder)

def _weights(mha):
    wq,wk,wv=mha.in_proj_weight.chunk(3,dim=0)
    if mha.in_proj_bias is None:return wq,wk,wv,None,None,None
    bq,bk,bv=mha.in_proj_bias.chunk(3,dim=0);return wq,wk,wv,bq,bk,bv

def _heads(x,mha):
    b,l,d=x.shape;h=mha.num_heads;hd=d//h
    return x.view(b,l,h,hd).transpose(1,2)

def _merge(x):
    b,h,l,hd=x.shape;return x.transpose(1,2).contiguous().view(b,l,h*hd)

def _project_qkv(mha,q,k_source=None):
    wq,wk,wv,bq,bk,bv=_weights(mha);source=q if k_source is None else k_source
    return F.linear(q,wq,bq),F.linear(source,wk,bk),F.linear(source,wv,bv)

def _attend(mha,q,k,v):
    out=F.scaled_dot_product_attention(_heads(q,mha),_heads(k,mha),_heads(v,mha),dropout_p=0.0,is_causal=False)
    return mha.out_proj(_merge(out))

def step(decoder,token:torch.Tensor,memory:torch.Tensor,state:DecoderKVState):
    if token.ndim!=3 or token.shape[1]!=1:raise ValueError('token must be [B,1,D]')
    if decoder.training or torch.is_grad_enabled():raise ValueError('KV cache is inference-only')
    if state.decoder_ref is not decoder:raise ValueError('KV cache belongs to another decoder')
    try:version=memory._version
    except RuntimeError:version=None
    if state.memory_ref is None:state.memory_ref=memory;state.memory_version=version
    elif state.memory_ref is not memory or state.memory_version!=version:
        raise ValueError('KV memory changed; start a new decode state')
    x=token
    for i,layer in enumerate(decoder.layers):
        cache=state.layers[i]
        n1=layer.norm1(x);wq,wk,wv,bq,bk,bv=_weights(layer.self_attn)
        q=F.linear(n1,wq,bq);k=F.linear(n1,wk,bk);v=F.linear(n1,wv,bv)
        hk,hv=_heads(k,layer.self_attn),_heads(v,layer.self_attn)
        cache.self_k=hk if cache.self_k is None else torch.cat((cache.self_k,hk),dim=2)
        cache.self_v=hv if cache.self_v is None else torch.cat((cache.self_v,hv),dim=2)
        sq=_heads(q,layer.self_attn)
        sa=F.scaled_dot_product_attention(sq,cache.self_k,cache.self_v,dropout_p=0.0,is_causal=False)
        x=x+layer.dropout1(layer.self_attn.out_proj(_merge(sa)))
        n2=layer.norm2(x);mha=layer.multihead_attn;wq,wk,wv,bq,bk,bv=_weights(mha)
        q=F.linear(n2,wq,bq)
        if cache.cross_k is None:
            cache.cross_k=_heads(F.linear(memory,wk,bk),mha)
            cache.cross_v=_heads(F.linear(memory,wv,bv),mha)
        ca=F.scaled_dot_product_attention(_heads(q,mha),cache.cross_k,cache.cross_v,dropout_p=0.0,is_causal=False)
        x=x+layer.dropout2(mha.out_proj(_merge(ca)))
        ff=layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm3(x)))))
        x=x+layer.dropout3(ff)
    state.index+=1
    return decoder.norm(x) if decoder.norm is not None else x

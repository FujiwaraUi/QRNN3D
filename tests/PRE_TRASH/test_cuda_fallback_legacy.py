import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import hsi_setup


def test_engine_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    opt = argparse.Namespace(
        prefix='gauss',
        arch='qrnn3d',
        batchSize=1,
        lr=1e-3,
        wd=0,
        loss='l2',
        init='kn',
        no_cuda=False,
        no_log=True,
        threads=1,
        seed=2018,
        resume=False,
        no_ropt=False,
        chop=False,
        resumePath=None,
        dataroot='dummy',
        vis_dir=None,
        clip=1e6,
        gpu_ids=[0],
    )

    engine = hsi_setup.Engine(opt)

    assert engine.device == 'cpu'
